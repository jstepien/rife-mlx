"""Video decode/encode with audio passthrough. P5 deliverable.

Plan: PyAV (or imageio-ffmpeg) for frame decode + encode; ffmpeg for audio
passthrough (copy the original audio stream onto the interpolated video, with
the output fps scaled by --multi). Mirrors upstream inference_video.py behaviour
(skvideo/ffmpeg) but with a cleaner macOS-friendly dep.
"""

from __future__ import annotations


def read_frames(path: str):
    raise NotImplementedError("P5")


def write_video(path: str, frames, fps: float, audio_from: str | None = None):
    raise NotImplementedError("P5")
