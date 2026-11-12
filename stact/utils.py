"""Small helpers shared by the ST-ACT reference implementation."""

from __future__ import annotations

import torch
from torch import Tensor


def left_pad_history(commands: Tensor, history_len: int) -> Tensor:
    """Left-pad a command sequence with its earliest available command.

    The paper (Sec. 3.1) builds ``H_t`` from the ``h`` most recent commands and
    left-pads with the earliest available command when fewer than ``h``
    commands exist, e.g. at the beginning of an episode.

    Args:
        commands: ``(B, n, d_action)`` tensor of ``n`` consecutive commands in
            chronological order, with ``n <= history_len``.
        history_len: Target history length ``h``.

    Returns:
        ``(B, history_len, d_action)`` tensor. The first ``history_len - n``
        entries replicate ``commands[:, 0]``.
    """
    if commands.dim() != 3:
        raise ValueError(f"`commands` must be (B, n, d_action). Got shape {tuple(commands.shape)}.")
    n_steps = commands.shape[1]
    if n_steps == 0:
        raise ValueError("`commands` must contain at least one command.")
    if n_steps >= history_len:
        return commands[:, -history_len:]
    pad = commands[:, :1].expand(-1, history_len - n_steps, -1)
    return torch.cat([pad, commands], dim=1)
