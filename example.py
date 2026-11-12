"""Minimal smoke test for the standalone ST-ACT implementation.

This script builds a small ST-ACT model on random tensors, runs a forward pass,
computes the total loss, and back-propagates. It verifies the structural claims
of the paper that are cheap to check:

* the input command embedding and the output projection are tied (Eq. 8);
* the decoder uses the configured number of layers and a 5-tap smoothing filter;
* the first predicted command aligns with the last history command (Eq. 9);
* padded chunk steps are excluded from the prediction loss;
* RoPE is norm-preserving.

Run with ``python example.py``. This is not a training script.
"""

from __future__ import annotations

import torch

from stact import (
    RoPE1D,
    STACTConfig,
    STACTModel,
    left_pad_history,
    masked_l1_loss,
    stact_loss,
)


def main() -> None:
    torch.manual_seed(0)

    cfg = STACTConfig(
        action_dim=7,
        history_len=32,
        chunk_size=32,
        cond_dim=64,
        dim_model=64,
        n_heads=4,
        n_decoder_layers=2,
        dim_feedforward=128,
        conv_kernel_size=5,
        conv_sigma=0.5,
    )
    model = STACTModel(cfg)

    batch, cond_len = 2, 16

    # Build a command stream once, then cut the overlapping history/chunk pair:
    #   H_t = (a_{t-h+1}, ..., a_t)      -> full[:, :h]
    #   A_t = (a_t, ..., a_{t+k-1})      -> full[:, h-1 : h-1+k]
    full = torch.randn(batch, cfg.history_len + cfg.chunk_size - 1, cfg.action_dim) * 0.05
    history = full[:, : cfg.history_len]
    target = full[:, cfg.history_len - 1 : cfg.history_len - 1 + cfg.chunk_size]
    assert torch.allclose(history[:, -1], target[:, 0]), "history/chunk overlap is broken"

    conditioning = torch.randn(batch, cond_len, cfg.cond_dim)

    actions_hat, features = model(history, conditioning)
    assert actions_hat.shape == (batch, cfg.chunk_size, cfg.action_dim), actions_hat.shape
    assert features.shape == (batch, cfg.chunk_size, cfg.dim_model), features.shape

    total, components = stact_loss(
        actions_hat,
        target,
        history,
        align_weight=cfg.align_loss_weight,
    )
    total.backward()

    # --- Eq. 8: the output projection is the transpose of the command embedding.
    assert model.output_projection_weight.shape == (cfg.action_dim, cfg.dim_model)
    assert torch.allclose(model.output_projection_weight, model.command_embed.weight.t())
    assert model.command_embed.bias is None
    assert model.command_embed.weight.grad is not None and model.command_embed.weight.grad.abs().sum() > 0

    # --- Sec. 3.3: 5-tap depth-wise filters in every attention branch.
    assert len(model.decoder.layers) == cfg.n_decoder_layers
    for layer in model.decoder.layers:
        for conv in (layer.conv1d_self, layer.conv1d_cross):
            assert conv.conv.kernel_size[0] == cfg.conv_kernel_size
            assert conv.conv.groups == cfg.dim_model == conv.conv.in_channels

    # --- Eq. 9: alignment loss is the squared error at the chunk boundary.
    manual_alignment = (actions_hat[:, 0] - history[:, -1]).pow(2).sum(dim=-1).mean().detach()
    assert torch.allclose(components["alignment_loss"], manual_alignment, atol=1e-6)
    expected_total = components["prediction_loss"] + cfg.align_loss_weight * manual_alignment
    assert torch.allclose(components["loss"], expected_total, atol=1e-6)

    # --- Padding is excluded from the prediction loss: huge errors on padded
    # steps must not leak into the masked average.
    noisy_prediction = target.clone()
    noisy_prediction[:, -4:] += 100.0
    is_pad = torch.zeros(batch, cfg.chunk_size, dtype=torch.bool)
    is_pad[:, -4:] = True
    assert masked_l1_loss(noisy_prediction, target, is_pad=is_pad).item() < 1e-6
    assert masked_l1_loss(noisy_prediction, target, is_pad=is_pad) < masked_l1_loss(
        noisy_prediction, target
    )

    # --- RoPE preserves the norm of every head vector.
    rope = RoPE1D(head_dim=cfg.head_dim, max_seq_len=cfg.max_seq_len)
    x = torch.randn(cfg.history_len, batch, cfg.n_heads, cfg.head_dim)
    assert torch.allclose(rope(x).norm(dim=-1), x.norm(dim=-1), atol=1e-5)

    # --- Left-padding repeats the earliest available command.
    short = full[:, :5]
    padded = left_pad_history(short, cfg.history_len)
    assert padded.shape == (batch, cfg.history_len, cfg.action_dim)
    assert torch.allclose(padded[:, 0], short[:, 0])
    assert torch.allclose(padded[:, -1], short[:, -1])

    print("ST-ACT smoke test passed")
    print(f"  actions_hat      : {tuple(actions_hat.shape)}")
    print(f"  decoder features : {tuple(features.shape)}")
    print(f"  total loss       : {total.item():.6f}")
    for name, value in components.items():
        print(f"  {name:17s}: {value.item():.6f}")


if __name__ == "__main__":
    main()
