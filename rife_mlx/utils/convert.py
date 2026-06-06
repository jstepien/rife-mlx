"""flownet.pkl (torch state_dict) -> MLX weights. P4 deliverable.

Conv weight (O,I,kH,kW) -> (O,kH,kW,I). ConvTranspose2d weight has its own rule
(torch stores (I,O,kH,kW)) — handled per-key once the pinned arch is known.
PReLU weight (C,) passthrough. `mx.eval` before save (lazy-zero guard).
upstream keys are prefixed `module.` (DataParallel) — stripped on load.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np


def strip_module_prefix(sd: dict) -> dict:
    return {k.replace("module.", "", 1): v for k, v in sd.items()}


def convert_state_dict(sd: dict[str, np.ndarray]) -> dict[str, mx.array]:
    raise NotImplementedError("P4: per-key conv vs conv-transpose transpose, "
                              "after the v4.25 arch + key set are pinned in P1")
