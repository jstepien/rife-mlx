"""Model wrapper — MLX port of train_log/RIFE_HDv3.py.

Upstream `Model` wraps `IFNet` and exposes `inference(img0, img1, timestep, scale)`
returning the interpolated middle frame. P3/P5 deliverable.
"""

from __future__ import annotations

import mlx.core as mx

from .IFNet_HDv3 import IFNet


class Model:
    def __init__(self) -> None:
        self.flownet = IFNet()

    def inference(self, img0: mx.array, img1: mx.array,
                  timestep: float = 0.5, scale: float = 1.0) -> mx.array:
        """img0,img1: [N,H,W,3] in [0,1]. Returns the frame at `timestep`."""
        raise NotImplementedError("P3/P5: scale_list = [8,4,2,1]/scale; pad to /64; "
                                  "flownet(cat[img0,img1], timestep, scale_list) -> merged")
