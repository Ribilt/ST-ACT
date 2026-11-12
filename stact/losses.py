"""Training objectives of the ST-ACT action expert (Sec. 3.4.2).

The paper minimises the negative log-likelihood in Eq. 3, which action-chunking
policies realise as an L1 regression over the predicted chunk. On top of that,
the command-consistency bias adds the alignment term of Eq. 9 between the first
predicted command and the most recent history command, so the predicted chunk
starts from the command that is already known at decision time:

``L = L_pred + lambda * L_align``.
"""

from __future__ import annotations

import torch.nn.functional as F  # noqa: N812
from torch import Tensor


def masked_l1_loss(prediction: Tensor, target: Tensor, is_pad: Tensor | None = None) -> Tensor:
    """L1 prediction loss averaged over valid chunk steps and coordinates.

    Args:
        prediction: ``(B, k, d_action)`` predicted commands.
        target: ``(B, k, d_action)`` demonstrated commands.
        is_pad: ``(B, k)`` boolean mask, ``True`` where ``target`` is padding.

    Returns:
        Scalar loss.
    """
    if prediction.shape != target.shape:
        raise ValueError(
            f"`prediction` and `target` must have the same shape. Got {tuple(prediction.shape)} "
            f"and {tuple(target.shape)}."
        )
    elementwise = F.l1_loss(prediction, target, reduction="none")
    if is_pad is None:
        return elementwise.mean()

    weights = (~is_pad).unsqueeze(-1).to(elementwise.dtype)  # (B, k, 1)
    denominator = weights.sum().clamp_min(1.0) * prediction.shape[-1]
    return (elementwise * weights).sum() / denominator


def alignment_loss(actions_hat: Tensor, history: Tensor, history_is_pad: Tensor | None = None) -> Tensor:
    """Boundary alignment between the predicted chunk and the command history.

    Implements Eq. 9, ``L_align = || a_hat_t - a_t ||^2``, where ``a_t`` is both
    the last entry of the history and the first entry of the target chunk.

    Args:
        actions_hat: ``(B, k, d_action)`` predicted commands.
        history: ``(B, h, d_action)`` input command history.
        history_is_pad: Optional ``(B, h)`` boolean mask; samples whose last
            history entry is padding are ignored.

    Returns:
        Scalar loss.
    """
    boundary_prediction = actions_hat[:, 0]  # a_hat_t
    boundary_reference = history[:, -1]  # a_t
    squared = (boundary_prediction - boundary_reference).pow(2).sum(dim=-1)  # (B,)

    if history_is_pad is None:
        return squared.mean()

    valid = (~history_is_pad[:, -1]).to(squared.dtype)
    return (squared * valid).sum() / valid.sum().clamp_min(1.0)


def stact_loss(
    actions_hat: Tensor,
    target_actions: Tensor,
    history: Tensor,
    target_is_pad: Tensor | None = None,
    history_is_pad: Tensor | None = None,
    align_weight: float = 0.1,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Total training loss and its components.

    Args:
        actions_hat: ``(B, k, d_action)`` predicted commands.
        target_actions: ``(B, k, d_action)`` demonstrated chunk ``A_t``.
        history: ``(B, h, d_action)`` command history ``H_t``.
        target_is_pad: ``(B, k)`` padding mask for ``target_actions``.
        history_is_pad: ``(B, h)`` padding mask for ``history``.
        align_weight: Weight ``lambda`` of the alignment loss (paper: ``0.1``).

    Returns:
        ``(total_loss, components)`` where ``components`` holds detached
        scalars for logging.
    """
    prediction_loss = masked_l1_loss(actions_hat, target_actions, is_pad=target_is_pad)
    boundary_loss = alignment_loss(actions_hat, history, history_is_pad=history_is_pad)
    total = prediction_loss + align_weight * boundary_loss
    components = {
        "loss": total.detach(),
        "prediction_loss": prediction_loss.detach(),
        "alignment_loss": boundary_loss.detach(),
    }
    return total, components
