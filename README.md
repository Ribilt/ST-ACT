# ST-ACT: Structural and Temporal Inductive Biases for an Action Expert

A minimal, standalone PyTorch reference implementation of **ST-ACT**, the action
expert described in

> *ST-ACT: Embedding Structural and Temporal Inductive Biases into Action Experts
> for Kinematically Consistent Robot Manipulation*

This repository is an extraction of the core model: it contains only the
decoder architecture, its configuration, and the training objective. It does not
include a training loop, dataset loaders, robot code, image encoders, or a
vision-language backbone. It is not wired to run any experiment out of the box.

## Contents

```
stact/
  __init__.py     public API
  config.py       STACTConfig (defaults + real-world / LIBERO presets)
  model.py        RoPE, RoPE attention, temporal Conv1D, decoder, STACTModel
  losses.py       masked L1 prediction loss + boundary alignment loss
  utils.py        left_pad_history
example.py        random-tensor forward/backward smoke test with assertions
requirements.txt  torch only
```

## Quick start

```bash
pip install -r requirements.txt
python example.py
```

```python
import torch
from stact import STACTConfig, STACTModel, stact_loss

cfg = STACTConfig(cond_dim=1024)          # real-world defaults: h = k = 100, 512/7/8
model = STACTModel(cfg)

history = torch.randn(8, cfg.history_len, cfg.action_dim)      # H_t
z_t = torch.randn(8, 128, cfg.cond_dim)                        # upstream VLM tokens
target = torch.randn(8, cfg.chunk_size, cfg.action_dim)        # A_t

actions_hat, features = model(history, z_t)                    # (8, 100, 7), (8, 100, 512)
loss, parts = stact_loss(actions_hat, target, history, align_weight=cfg.align_loss_weight)
loss.backward()
```

## Interface

`STACTModel.forward(history, conditioning, cond_is_pad=None) -> (actions_hat, features)`

| Argument | Shape | Meaning |
| --- | --- | --- |
| `history` | `(B, h, d_action)` | control-command history `H_t`, oldest first |
| `conditioning` | `(B, L, cond_dim)` | high-level condition `z_t` (keys/values of cross-attention) |
| `cond_is_pad` | `(B, L)` bool | optional mask for padded conditioning tokens |
| `actions_hat` | `(B, k, d_action)` | predicted chunk `A_t` |
| `features` | `(B, k, d_model)` | final decoder features |

`h == k` is enforced: the decoder emits one command per history position, and
`actions_hat[:, 0]` is the prediction of `a_t`, which is already the last entry
of `history`. Short histories are prepared with `left_pad_history(commands, h)`.

The model consumes **commands**, not measured joint states. Normalisation of
commands and construction of `z_t` are the caller's responsibility.

## Paper-to-code map

| Paper | Equation / section | Code |
| --- | --- | --- |
| Action expert `f_theta : (z_t, H_{t-1}) -> A_t` | Eq. 4 | `STACTModel.forward` |
| Command history, left padding with `a_1` | Sec. 3.1 | `left_pad_history`, `cond_is_pad` analogue |
| RoPE continuous in position, norm preserving | Sec. 3.2.1, Eq. 2 | `RoPE1D` |
| Relative scores `Q^T R(u_s - u_t) K` | Sec. 3.2.2, Eqs. 3-5 | `RoPEMultiheadAttention` (`rope_on_key=True` in self-attention) |
| Depth-wise smoothing `O~_d,t = sum_j G_d,j O_d,t+j` | Sec. 3.3.1, Eq. 5 | `TemporalConv1d` |
| Normalised Gaussian kernel | Eq. 6 | `TemporalConv1d._init_lowpass_weights` |
| Kaiming rescaling, `inf`-norm bound `sqrt(6/(D*omega))` | Eq. 7 | `TemporalConv1d._init_lowpass_weights` |
| Smoothing before the residual addition | Sec. 3.3.1 | `STACTDecoderLayer.forward` |
| Tied output projection `W_o = W_e^T` | Sec. 3.4.1, Eq. 8 | `STACTModel.output_projection_weight` |
| Boundary alignment `L_align = ||a_hat_t - a_t||^2` | Sec. 3.4.2, Eq. 9 | `alignment_loss` |
| Total loss `L = L_pred + lambda * L_align` | Sec. 3.4.2 | `stact_loss` |

## Defaults

`STACTConfig()` reproduces the real-world column of Table 1. `STACTConfig.libero()`
returns the LIBERO variant.

| Parameter | Real world | LIBERO |
| --- | --- | --- |
| History length `h` / chunk size `k` | 100 / 100 | 32 / 32 |
| Decoder width / layers / heads | 512 / 7 / 8 | 512 / 7 / 8 |
| Feed-forward width | 3200 | 3200 |
| Conv1D kernel / Gaussian `sigma` | 5 / 1.5 | 5 / 0.5 |
| Conv1D padding | replicate, 2 per side | replicate, 2 per side |
| Alignment weight `lambda` | 0.1 | 0.1 |
| Dropout | 0.1 | 0.1 |

The paper's `L_pred` (Eq. 3) is a negative log-likelihood. As in action-chunking
policies, this implementation realises it as an L1 regression over the chunk,
with padded steps excluded from the average.

## Design notes and deliberate choices

1. **History input is always used.** The extracted model always conditions on
   `H_t`; there is no "no-history" switch.
2. **Weight tying is mandatory.** `command_embed` is bias-free and its transpose
   is the output projection, so Eq. 8 holds exactly. There is no separate action
   head and no output bias.
3. **Alignment loss is implemented.** The original training code did not contain
   Eq. 9; it is added here as `alignment_loss`.
4. **Seven decoder layers.** The paper uses seven layers. The source ACT
   implementation defaults to one layer because of an upstream bug; that
   workaround is intentionally not carried over.
5. **Replicate padding and a 5-tap kernel.** The smoothing filter uses
   `kernel_size=5`, `sigma` from the preset, and explicit replicate padding of
   two samples per side, matching Table 1.
6. **Gaussian initialisation survives.** Xavier initialisation is applied to
   `nn.Linear` layers only. A blanket "dim > 1" rule would overwrite the
   Eq. 7 filter initialisation, defeating the low-pass prior.
7. **Cross-attention RoPE.** As in the source model, cross-attention rotates
   queries only: the conditioning tokens have no temporal index, so rotating
   keys would not yield a meaningful relative offset. Self-attention rotates
   both queries and keys, giving relative positional scores as in Eq. 3.

## Scope

Included: decoder architecture, configuration, losses, padding helper, example.

Not included: VAE branch, ResNet image backbone, observation/action
normalisation, policy wrapper with action queues or temporal ensembling,
training/evaluation scripts, LIBERO or SO-ARM101 integration.

## Attribution

The architecture follows the ST-ACT paper. The reference code base was derived
from LeRobot's ACT implementation, which is licensed under Apache-2.0; the
original `LICENSE` is retained.

## 中文说明

本仓库从 ST-ACT 论文与现有实现中提取了动作专家的核心代码：命令历史输入、
`z_t` 交叉注意力条件、RoPE 相对位置编码、深度可分离时序卷积平滑、命令嵌入与
输出权重绑定，以及论文式 (9) 的边界对齐损失。运行 `python example.py` 可在随机
张量上完成前向、损失与反向传播，并校验上述结构性质。

代码不包含训练脚本、数据加载、图像编码器与机器人接口；命令归一化和 `z_t`
的生成由调用方负责。
