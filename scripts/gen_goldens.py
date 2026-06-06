"""P1: PyTorch parity oracle. Requires the [parity] extra + refs/<v>/ package.

Captures (seed 1234, CPU fp32):
  - warp: random img + random flow -> warped (Gate A oracle)
  - bilinear interpolate: up/down at a few scales
  - IFBlock: per-block flow/mask output
  - full IFNet.inference(img0,img1,t=0.5,scale=1): merged middle frame (Gate B)
Saves goldens/<v>.npz. Also reads the bundled IFNet_HDv3.py to print the pinned
arch (block count, channels, scale_list, align_corners) for config.py.
"""

raise NotImplementedError("P1 — after fetch_weights.py")
