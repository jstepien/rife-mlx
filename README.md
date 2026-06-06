# rife-mlx

Apple **MLX** port of [Practical-RIFE](https://github.com/hzwer/Practical-RIFE) 4.x —
real-time video **frame interpolation** on Apple Silicon. **MIT.**

The first MLX / Apple-Silicon-native RIFE: torch-free inference, HF auto-download
(upstream ships weights via Google Drive only), arbitrary-timestep interpolation,
`--multi` Nx frame-rate, `--scale` pyramid for 4K, and audio-preserving video output.

> **Status: P0 (scaffold).** The two crux ops (`grid_sample` warp + bilinear
> `interpolate`) are written; IFNet arch translation, parity, weight conversion,
> and the video pipeline are in progress. Pinned version: **RIFE 4.25**. See
> [`docs/PREFLIGHT.md`](docs/PREFLIGHT.md).

## Planned usage

```bash
pip install rife-mlx

# 2× frame rate, keep audio
rife-mlx -i input.mp4 -o output.mp4 --multi 2

# interpolate a single middle frame
rife-mlx --img0 a.png --img1 b.png -t 0.5 -o mid.png

# 4K: halve the pyramid scale for memory
rife-mlx -i 4k.mp4 -o out.mp4 --multi 2 --scale 0.5
```

## Why a port

- **No MLX RIFE exists** — only ncnn-Vulkan / CoreML community builds.
- Upstream weights are **Google-Drive-only**; this re-hosts RIFE 4.25 on
  `mlx-community` (MIT + attribution) with HF auto-download and torch-free load.
- The backward-`warp` (`grid_sample`) and bilinear `interpolate` are hand-rolled
  in MLX/NHWC and parity-locked vs PyTorch.

## License

MIT, inherited from upstream Practical-RIFE (© hzwer). Weights are the official
RIFE 4.25 release, converted to MLX.
