"""Configuration of the ST-ACT action expert.

Defaults follow the real-world setting of Table 1 in ``article.tex``:
history/chunk length 100, decoder 512/7/8, a 5-tap Gaussian Conv1D with
``sigma = 1.5`` and an alignment-loss weight of ``0.1``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class STACTConfig:
    """Hyper-parameters for :class:`stact.model.STACTModel`.

    Attributes:
        action_dim: Number of command coordinates ``d_action``.
        history_len: Length ``h`` of the control-command history ``H_t``.
        chunk_size: Length ``k`` of the predicted action chunk ``A_t``.
        cond_dim: Feature width of the upstream conditioning tokens ``z_t``.
            ``None`` means the tokens are already ``dim_model`` wide.
        dim_model: Decoder width.
        n_heads: Number of decoder attention heads.
        n_decoder_layers: Number of decoder layers (paper: 7).
        dim_feedforward: Hidden width of the decoder feed-forward block.
        dropout: Dropout probability used throughout the decoder.
        pre_norm: Use pre-layer-norm instead of the paper's post-layer-norm.
        rope_base: Base of the RoPE frequency progression.
        conv_kernel_size: Temporal Conv1D kernel size ``omega`` (odd; paper: 5).
        conv_sigma: Standard deviation of the initial Gaussian kernel.
        align_loss_weight: Weight ``lambda`` of the alignment loss in Eq. 9.

    Note:
        The paper uses ``h = k = 100`` (real world) and ``h = k = 32``
        (LIBERO). The decoder predicts one command per input position, so the
        two lengths must match.
    """

    # Input / output structure.
    action_dim: int = 7
    history_len: int = 100
    chunk_size: int = 100
    cond_dim: int | None = None

    # Transformer decoder.
    dim_model: int = 512
    n_heads: int = 8
    n_decoder_layers: int = 7
    dim_feedforward: int = 3200
    dropout: float = 0.1
    pre_norm: bool = False

    # Relative positional encoding (Sec. 3.2).
    rope_base: float = 10000.0

    # Temporal smoothing (Sec. 3.3).
    conv_kernel_size: int = 5
    conv_sigma: float = 1.5

    # Command consistency (Sec. 3.4.2).
    align_loss_weight: float = 0.1

    def __post_init__(self) -> None:
        if self.history_len != self.chunk_size:
            raise ValueError(
                "`history_len` and `chunk_size` must be equal because the decoder predicts one "
                f"command per input position. Got history_len={self.history_len} and "
                f"chunk_size={self.chunk_size}."
            )
        if self.history_len < 1:
            raise ValueError(f"`history_len` must be positive. Got {self.history_len}.")
        if self.action_dim < 1:
            raise ValueError(f"`action_dim` must be positive. Got {self.action_dim}.")
        if self.n_heads < 1 or self.dim_model % self.n_heads != 0:
            raise ValueError(
                f"`dim_model` must be divisible by `n_heads`. Got dim_model={self.dim_model} "
                f"and n_heads={self.n_heads}."
            )
        if self.head_dim % 2 != 0:
            raise ValueError(
                "RoPE rotates halves of each head, so the head dimension must be even. Got "
                f"head_dim={self.head_dim} (dim_model={self.dim_model}, n_heads={self.n_heads})."
            )
        if self.conv_kernel_size < 1 or self.conv_kernel_size % 2 == 0:
            raise ValueError(
                "`conv_kernel_size` must be a positive odd number so that the smoothing filter is "
                f"centred. Got {self.conv_kernel_size}."
            )
        if self.conv_sigma <= 0:
            raise ValueError(f"`conv_sigma` must be positive. Got {self.conv_sigma}.")
        if self.align_loss_weight < 0:
            raise ValueError(f"`align_loss_weight` must be non-negative. Got {self.align_loss_weight}.")

    @property
    def head_dim(self) -> int:
        """Dimension of a single attention head."""
        return self.dim_model // self.n_heads

    @property
    def max_seq_len(self) -> int:
        """Longest sequence the decoder RoPE table has to cover."""
        return max(self.history_len, self.chunk_size)

    @classmethod
    def real_world(cls, **overrides) -> STACTConfig:
        """Preset for the SO-ARM101 experiments (h = k = 100, 60 Hz)."""
        return cls(**overrides)

    @classmethod
    def libero(cls, **overrides) -> STACTConfig:
        """Preset for the LIBERO experiments (h = k = 32, 20 Hz, sigma = 0.5)."""
        kwargs = {"history_len": 32, "chunk_size": 32, "conv_sigma": 0.5}
        kwargs.update(overrides)
        return cls(**kwargs)
