"""ST-ACT: Structural and Temporal inductive biases for an Action Expert.

This module contains the paper-faithful decoder used by ST-ACT (Sec. 3 of
``article.tex``):

* relative positional structure through 1-D Rotary Position Embeddings (RoPE,
  Sec. 3.2);
* temporal smoothing through a depth-wise 1-D convolution with a Gaussian
  low-pass initialisation (Sec. 3.3);
* command consistency through weight tying of the command embedding and the
  output projection (Sec. 3.4.1).

The model consumes a control-command history ``H_t`` of shape ``(B, h, A)`` and
the upstream conditioning tokens ``z_t`` of shape ``(B, L, D_cond)``. It returns
one command per history position, i.e. ``(B, k, A)`` with ``k = h``. The first
predicted command corresponds to the most recent history entry ``a_t``, which is
the overlap exploited by the alignment loss in Sec. 3.4.2.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from stact.config import STACTConfig
from stact.utils import left_pad_history

__all__ = [
    "RoPE1D",
    "RoPEMultiheadAttention",
    "TemporalConv1d",
    "STACTDecoderLayer",
    "STACTDecoder",
    "STACTModel",
    "left_pad_history",
]


class RoPE1D(nn.Module):
    """1-D rotary positional embedding (RoPE).

    For a head vector ``x`` at position ``u`` the embedding applies a
    block-diagonal rotation ``R(u)`` with ``||R(u) x|| = ||x||``. Attention
    scores then depend on positions only through their difference, which gives
    the decoder an explicitly continuous dependence on position (Sec. 3.2).

    Tensors are laid out as ``(T, B, n_heads, head_dim)``, matching the
    sequence-first convention used by the decoder.

    Args:
        head_dim: Dimension of a single attention head (must be even).
        max_seq_len: Number of positions to precompute rotations for.
        base: Base of the geometric frequency progression.
    """

    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE requires an even head dimension. Got {head_dim}.")
        if max_seq_len < 1:
            raise ValueError(f"`max_seq_len` must be positive. Got {max_seq_len}.")
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        positions = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)  # (max_seq_len, head_dim / 2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (max_seq_len, head_dim)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    @staticmethod
    def rotate_half(x: Tensor) -> Tensor:
        """Rotate the two halves of the last dimension: ``[x1, x2] -> [-x2, x1]``."""
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, x: Tensor, offset: int = 0) -> Tensor:
        """Apply the rotary transform.

        Args:
            x: ``(T, B, n_heads, head_dim)`` tensor.
            offset: First position index covered by ``x``.

        Returns:
            Tensor with the same shape as ``x``.
        """
        seq_len = x.shape[0]
        if seq_len > self.max_seq_len or offset < 0 or offset + seq_len > self.max_seq_len:
            raise ValueError(
                f"RoPE covers positions [0, {self.max_seq_len}); requested [{offset}, "
                f"{offset + seq_len})."
            )
        cos = self.cos_cached[offset : offset + seq_len].to(dtype=x.dtype)[:, None, None, :]
        sin = self.sin_cached[offset : offset + seq_len].to(dtype=x.dtype)[:, None, None, :]
        return x * cos + self.rotate_half(x) * sin


class RoPEMultiheadAttention(nn.Module):
    """Multi-head attention with optional RoPE on the query (and key) vectors.

    Self-attention rotates both queries and keys (``rope_on_key=True``), so its
    scores depend on the relative offset of the two positions. Cross-attention
    rotates only the queries (``rope_on_key=False``) because the conditioning
    tokens ``z_t`` carry no temporal index; the rotation therefore acts as a
    position-dependent cue on the decoder side instead of a relative one.

    Args:
        embed_dim: Model width.
        num_heads: Number of attention heads.
        dropout: Attention dropout probability.
        rope: Shared :class:`RoPE1D` instance, or ``None`` for plain attention.
        rope_on_key: Whether to rotate keys in addition to queries.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        rope: RoPE1D | None = None,
        rope_on_key: bool = True,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"`embed_dim` must be divisible by `num_heads`. Got {embed_dim}, {num_heads}.")
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.rope = rope
        self.rope_on_key = rope_on_key
        self.dropout = dropout
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Tensor | None = None,
        need_weights: bool = False,
        attn_mask: Tensor | None = None,
    ) -> tuple[Tensor, None]:
        """Run attention.

        Args:
            query: ``(T_q, B, E)``.
            key: ``(T_k, B, E)``.
            value: ``(T_k, B, E)``.
            key_padding_mask: ``(B, T_k)`` boolean mask, ``True`` for padding.
            need_weights: Accepted for API parity with ``nn.MultiheadAttention``
                and ignored; attention weights are never materialised.
            attn_mask: Optional additive bias broadcastable to
                ``(B, n_heads, T_q, T_k)``.

        Returns:
            ``(attended, None)`` where ``attended`` is ``(T_q, B, E)``.
        """
        if need_weights:
            raise NotImplementedError("RoPEMultiheadAttention never materialises attention weights.")

        t_q, batch, embed_dim = query.shape
        t_k = key.shape[0]

        q = self.q_proj(query).view(t_q, batch, self.num_heads, self.head_dim)
        k = self.k_proj(key).view(t_k, batch, self.num_heads, self.head_dim)
        v = self.v_proj(value).view(t_k, batch, self.num_heads, self.head_dim)

        if self.rope is not None:
            q = self.rope(q)
            if self.rope_on_key:
                k = self.rope(k)

        # (B, n_heads, T, head_dim)
        q = q.permute(1, 2, 0, 3)
        k = k.permute(1, 2, 0, 3)
        v = v.permute(1, 2, 0, 3)

        bias = None
        if key_padding_mask is not None:
            bias = torch.zeros(batch, 1, 1, t_k, dtype=q.dtype, device=q.device)
            bias = bias.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
        if attn_mask is not None:
            bias = attn_mask if bias is None else bias + attn_mask

        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=bias,
            dropout_p=self.dropout if self.training else 0.0,
            scale=self.scale,
        )
        attended = attended.permute(2, 0, 1, 3).reshape(t_q, batch, embed_dim)
        return self.out_proj(attended), None


class TemporalConv1d(nn.Module):
    """Depth-wise temporal smoothing filter applied after attention.

    Each feature channel has its own learnable filter, so the convolution acts
    along time only and never mixes channels (Sec. 3.3). The filter is
    initialised as a normalised Gaussian and then rescaled so that its infinity
    norm is bounded by ``sqrt(6 / (D * omega))``, keeping the smoothing branch
    small relative to the identity residual path at the start of training
    (Eqs. 6-7).

    Args:
        num_channels: Number of feature channels ``D``.
        kernel_size: Odd kernel size ``omega``.
        sigma: Standard deviation of the initial Gaussian kernel.
    """

    def __init__(self, num_channels: int, kernel_size: int = 5, sigma: float = 1.5) -> None:
        super().__init__()
        if kernel_size % 2 == 0 or kernel_size < 1:
            raise ValueError(f"`kernel_size` must be a positive odd number. Got {kernel_size}.")
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2
        self.sigma = sigma

        self.conv = nn.Conv1d(
            num_channels,
            num_channels,
            kernel_size,
            padding=0,  # padding is applied explicitly as "replicate"
            bias=False,
            groups=num_channels,
        )
        self._init_lowpass_weights()

    @torch.no_grad()
    def _init_lowpass_weights(self) -> None:
        k = self.kernel_size
        offsets = torch.arange(k, dtype=torch.float32) - k // 2
        kernel = torch.exp(-0.5 * (offsets / self.sigma) ** 2)
        kernel = kernel / kernel.sum()  # Eq. 6

        fan_in = self.conv.in_channels * k
        bound = math.sqrt(6.0 / fan_in)  # Eq. 7
        kernel = kernel * (bound / kernel.abs().max().clamp_min(1e-12))

        weight = kernel.view(1, 1, k).repeat(self.num_channels, 1, 1)
        self.conv.weight.copy_(weight)

    def forward(self, x: Tensor) -> Tensor:
        """Smooth a ``(B, D, T)`` feature sequence along time."""
        if x.dim() != 3:
            raise ValueError(f"Expected (B, D, T) input. Got shape {tuple(x.shape)}.")
        padded = F.pad(x, (self.padding, self.padding), mode="replicate")
        return self.conv(padded)


class STACTDecoderLayer(nn.Module):
    """One decoder layer: self-attention, cross-attention and feed-forward.

    Each attention branch is followed, before its residual addition, by the
    temporal smoothing convolution ("attention-then-smoothing", Sec. 3.3.1).

    Args:
        config: Model configuration.
        rope: Shared RoPE instance used by both attention branches.
    """

    def __init__(self, config: STACTConfig, rope: RoPE1D) -> None:
        super().__init__()
        self.pre_norm = config.pre_norm
        self.activation = F.gelu

        self.self_attn = RoPEMultiheadAttention(
            config.dim_model,
            config.n_heads,
            dropout=config.dropout,
            rope=rope,
            rope_on_key=True,
        )
        self.cross_attn = RoPEMultiheadAttention(
            config.dim_model,
            config.n_heads,
            dropout=config.dropout,
            rope=rope,
            rope_on_key=False,
        )

        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)

        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.norm3 = nn.LayerNorm(config.dim_model)

        self.dropout = nn.Dropout(config.dropout)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.dropout3 = nn.Dropout(config.dropout)

        self.conv1d_self = TemporalConv1d(
            config.dim_model, kernel_size=config.conv_kernel_size, sigma=config.conv_sigma
        )
        self.conv1d_cross = TemporalConv1d(
            config.dim_model, kernel_size=config.conv_kernel_size, sigma=config.conv_sigma
        )

    @staticmethod
    def _smooth(x: Tensor, conv: TemporalConv1d) -> Tensor:
        """Apply a temporal Conv1D to a sequence-first ``(T, B, D)`` tensor."""
        return conv(x.permute(1, 2, 0)).permute(2, 0, 1)

    def forward(self, x: Tensor, cond: Tensor, cond_is_pad: Tensor | None = None) -> Tensor:
        """Run the layer.

        Args:
            x: Decoder features ``(T_dec, B, D)``.
            cond: Conditioning tokens ``z_t`` as ``(T_cond, B, D)``.
            cond_is_pad: ``(B, T_cond)`` boolean padding mask for ``cond``.

        Returns:
            Updated decoder features ``(T_dec, B, D)``.
        """
        # --- Self-attention over the command history, with RoPE on Q and K.
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        x = self.self_attn(x, x, x)[0]
        x = self._smooth(x, self.conv1d_self)
        x = skip + self.dropout1(x)

        # --- Cross-attention against the conditioning tokens, RoPE on Q only.
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = self.cross_attn(x, cond, cond, key_padding_mask=cond_is_pad)[0]
        x = self._smooth(x, self.conv1d_cross)
        x = skip + self.dropout2(x)

        # --- Position-wise feed-forward block.
        if self.pre_norm:
            skip = x
            x = self.norm3(x)
        else:
            x = self.norm2(x)
            skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout3(x)
        if not self.pre_norm:
            x = self.norm3(x)
        return x


class STACTDecoder(nn.Module):
    """Stack of :class:`STACTDecoderLayer` followed by a final LayerNorm."""

    def __init__(self, config: STACTConfig) -> None:
        super().__init__()
        rope = RoPE1D(config.head_dim, max_seq_len=config.max_seq_len, base=config.rope_base)
        self.layers = nn.ModuleList(
            [STACTDecoderLayer(config, rope) for _ in range(config.n_decoder_layers)]
        )
        self.norm = nn.LayerNorm(config.dim_model)

    def forward(
        self,
        x: Tensor,
        cond: Tensor,
        cond_is_pad: Tensor | None = None,
    ) -> Tensor:
        """Run every decoder layer.

        Args:
            x: Decoder features ``(T_dec, B, D)``.
            cond: Conditioning tokens ``(T_cond, B, D)``.
            cond_is_pad: ``(B, T_cond)`` boolean padding mask for ``cond``.

        Returns:
            Decoder features ``(T_dec, B, D)``.
        """
        for layer in self.layers:
            x = layer(x, cond, cond_is_pad=cond_is_pad)
        return self.norm(x)


class STACTModel(nn.Module):
    """The ST-ACT action expert.

    ``f_theta : (z_t, H_{t-1}) -> A_t`` as defined in Eq. 4, implemented as a
    Transformer decoder over the command history with cross-attention to the
    upstream conditioning tokens.

    The input command embedding ``W_e`` and the output projection are tied,
    ``W_o = W_e^T`` (Eq. 8), so the same parameters both encode a command into
    the hidden space and read a command back out of the decoder features. The
    model therefore has no independently learned output matrix and no output
    bias.

    Args:
        config: Model configuration.
    """

    def __init__(self, config: STACTConfig) -> None:
        super().__init__()
        self.config = config

        # Shared command embedding W_e; bias-free so that W_o = W_e^T is exact.
        self.command_embed = nn.Linear(config.action_dim, config.dim_model, bias=False)

        # Optional projection when the upstream tokens are not already dim_model wide.
        self.cond_proj: nn.Module = (
            nn.Identity()
            if config.cond_dim in (None, config.dim_model)
            else nn.Linear(config.cond_dim, config.dim_model)
        )

        self.decoder = STACTDecoder(config)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Xavier-uniform initialisation of the linear layers.

        Only ``nn.Linear`` weights are re-initialised. The temporal Conv1D
        filters keep their Gaussian low-pass initialisation from Eq. 7; a global
        "every tensor with dim > 1" rule would overwrite it.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @property
    def output_projection_weight(self) -> Tensor:
        """Tied output projection ``W_o = W_e^T`` with shape ``(d_action, dim_model)``."""
        return self.command_embed.weight.t()

    def forward(
        self,
        history: Tensor,
        conditioning: Tensor,
        cond_is_pad: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Predict an action chunk from a command history.

        Args:
            history: Control-command history ``H_t`` as ``(B, h, d_action)``.
                Left-pad short histories with :func:`left_pad_history`.
            conditioning: Upstream conditioning tokens ``z_t`` as
                ``(B, L, cond_dim)``, i.e. ``(B, L, dim_model)`` when
                ``config.cond_dim`` is ``None``.
            cond_is_pad: ``(B, L)`` boolean mask marking padded conditioning
                tokens (``True`` = padding).

        Returns:
            ``(actions_hat, features)`` where ``actions_hat`` is
            ``(B, k, d_action)`` and ``features`` is the final decoder feature
            map ``(B, k, dim_model)``. With ``k = h``, ``actions_hat[:, 0]``
            predicts the most recent command ``a_t`` already contained in
            ``history[:, -1]``.
        """
        if history.dim() != 3:
            raise ValueError(f"`history` must be (B, h, d_action). Got {tuple(history.shape)}.")
        if history.shape[1] != self.config.history_len:
            raise ValueError(
                f"`history` must have {self.config.history_len} steps. Got {history.shape[1]} "
                "(use `left_pad_history` to pad short histories)."
            )
        if history.shape[2] != self.config.action_dim:
            raise ValueError(
                f"`history` must have action dimension {self.config.action_dim}. Got {history.shape[2]}."
            )

        # Encode the command history with the shared embedding W_e.
        x = self.command_embed(history).transpose(0, 1)  # (h, B, D)
        cond = self.cond_proj(conditioning).transpose(0, 1)  # (L, B, D)
        if cond.shape[-1] != self.config.dim_model:
            raise ValueError(
                f"Conditioning tokens must be projected to {self.config.dim_model} features. "
                f"Got {cond.shape[-1]} (check `cond_dim`)."
            )

        features = self.decoder(x, cond, cond_is_pad=cond_is_pad).transpose(0, 1)  # (B, k, D)

        # Read commands back out with the tied projection W_o = W_e^T (Eq. 8).
        actions_hat = F.linear(features, self.output_projection_weight)
        return actions_hat, features
