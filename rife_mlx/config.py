"""Version registry — pinned config for the RIFE checkpoint(s) we port.

Source of truth: the bundled `IFNet_HDv3.py` + `flownet.pkl` that ship together
per version (Google Drive), NOT the GitHub repo (which omits the version arch).
CONFIRM gate #3: pin the exact arch from the downloaded package, never guess.

This round pins **RIFE 4.25** (upstream's recommended default; "anime scenes
significantly improved"). NOTE the 4.25 release note: "I am trying using more
flow blocks, so the scale_list will change accordingly" — so `block_channels`
and `scale_list` here are PLACEHOLDERS to be overwritten in P1 from the actual
bundled IFNet_HDv3.py before any parity work.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class VersionConfig:
    name: str               # internal key
    hf_name: str            # published HF repo name (mlx-community/<hf_name>)
    gdrive_id: str          # per-version package (model .py + flownet.pkl)
    # arch params — PLACEHOLDER until pinned from the bundled IFNet_HDv3.py (P1)
    scale_list: tuple[float, ...] = (8, 4, 2, 1)
    block_channels: tuple[int, ...] = (192, 128, 96, 64)
    # warp / interpolate semantics (verified from warplayer.py)
    grid_align_corners: bool = True
    grid_padding_mode: str = "border"


VERSIONS: dict[str, VersionConfig] = {
    "4.25": VersionConfig(
        name="4.25",
        hf_name="RIFE-4.25",
        gdrive_id="1ZKjcbmt1hypiFprJPIKW0Tt0lr_2i7bg",
    ),
}

DEFAULT_VERSION = "4.25"
