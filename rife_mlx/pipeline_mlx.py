"""High-level interpolation + CLI. P5 deliverable (full video pipeline).

    rife-mlx -i in.mp4 -o out.mp4 --multi 2 --scale 1.0
    rife-mlx --img0 a.png --img1 b.png -t 0.5 -o mid.png

--multi N: insert N-1 frames between each pair (recursive midpoint at scale).
--scale {0.25,0.5,1,2,4}: coarse-to-fine pyramid scale (0.5 for 4K).
Video path (rife_mlx/video.py): decode -> interpolate -> encode + audio passthrough.
"""

from __future__ import annotations

from .config import DEFAULT_VERSION


def interpolate_pair(img0, img1, timestep: float = 0.5, scale: float = 1.0,
                     version: str = DEFAULT_VERSION):
    raise NotImplementedError("P5: build_model -> Model.inference")


def cli_main() -> None:
    raise NotImplementedError("P5: argparse (img-pair + video modes) mirroring "
                              "inference_img.py / inference_video.py")


if __name__ == "__main__":
    cli_main()
