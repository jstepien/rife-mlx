"""Model build + weight load (HF auto-download / local override). P4 deliverable.

Resolution: weights_dir arg | $RIFE_MLX_WEIGHTS_DIR | dist/<hf_name> | HF
mlx-community/<hf_name>. Each dir: model.safetensors + config.json.
"""

from __future__ import annotations

import os

from ..config import VERSIONS

WEIGHTS_DIR_ENV = "RIFE_MLX_WEIGHTS_DIR"
HF_ORG = "mlx-community"


def resolve_repo_id(version: str) -> str:
    return f"{HF_ORG}/{VERSIONS[version].hf_name}"


def build_model(version: str = "4.25", weights_dir: str | None = None):
    raise NotImplementedError("P4: instantiate RIFE_HDv3.Model + load safetensors + eval")
