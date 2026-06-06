"""IFNet (RIFE 4.x) — MLX port, isomorphic to the bundled train_log/IFNet_HDv3.py.

STATUS: skeleton. The exact arch (number of IFBlocks, per-block channels,
scale_list, Head encoder presence, ResConv definition, align_corners flags) is
pinned in P1 from the v4.25 package downloaded by scripts/fetch_weights.py, then
translated + parity-locked in P3 (Gate B).

Building blocks (4.x, to confirm against the pinned source):
  conv(in,out,k=3,s,p) -> Conv2d + PReLU
  ResConv(c)           -> conv(c,c,3,1,1) + PReLU, residual * beta
  IFBlock(in_planes,c) -> conv0 (2x stride-2 downsample) + N ResConv +
                          ConvTranspose2d -> PixelShuffle, emits flow(4) + mask(1)
  IFNet.forward(x, timestep, scale_list): coarse-to-fine over blocks; each block
    warps img0/img1 by the running flow, refines (flow,mask); final fuse
    merged = warp(img0,flow0)*mask + warp(img1,flow1)*(1-mask).

Reuse: ops.warp, ops.interpolate_bilinear, ops.grid_sample; PReLU + pixel-shuffle
patterns carried from the realesrgan-mlx port.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class IFNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        raise NotImplementedError("P3: translate from pinned v4.25 IFNet_HDv3.py")

    def __call__(self, x: mx.array, timestep: float = 0.5,
                 scale_list=(8, 4, 2, 1)):  # -> (flow_list, mask, merged)
        raise NotImplementedError("P3")
