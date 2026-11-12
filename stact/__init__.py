"""ST-ACT: structural and temporal inductive biases for an action expert.

Standalone reference implementation extracted from the ST-ACT paper
(``article.tex``). Public API:

* :class:`STACTConfig` -- hyper-parameters and experimental presets.
* :class:`STACTModel` -- the action expert ``f_theta(z_t, H_{t-1}) -> A_t``.
* :func:`stact_loss` -- prediction loss plus the boundary alignment loss.
* :func:`left_pad_history` -- left-padding of short command histories.
"""

from stact.config import STACTConfig
from stact.losses import alignment_loss, masked_l1_loss, stact_loss
from stact.model import (
    RoPE1D,
    RoPEMultiheadAttention,
    STACTDecoder,
    STACTDecoderLayer,
    STACTModel,
    TemporalConv1d,
    left_pad_history,
)

__all__ = [
    "STACTConfig",
    "STACTModel",
    "STACTDecoder",
    "STACTDecoderLayer",
    "RoPE1D",
    "RoPEMultiheadAttention",
    "TemporalConv1d",
    "masked_l1_loss",
    "alignment_loss",
    "stact_loss",
    "left_pad_history",
]

__version__ = "1.0.0"
