"""CLI progress reporting without model weights or video encoding."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from rife_mlx import pipeline_mlx, video


@pytest.mark.parametrize("frame_count,multi,total", [(3, 2, 5), (3, 1, 3), (None, 2, None)])
@pytest.mark.parametrize("is_terminal", [True, False])
def test_progress_counts_written_frames_and_finalises(
        monkeypatch, capsys, frame_count, multi, total, is_terminal):
    import tqdm

    monkeypatch.setattr(pipeline_mlx.sys, "argv",
                        ["rife-mlx", "-i", "input.mp4", "-o", "output.mp4", "--multi", str(multi)])
    monkeypatch.setattr(pipeline_mlx.sys.stderr, "isatty", lambda: is_terminal)
    monkeypatch.setattr(pipeline_mlx, "build_model", lambda *args, **kwargs: object())
    monkeypatch.setattr(video, "probe_video", lambda path: (30.0, frame_count))
    monkeypatch.setattr(video, "iter_frames", lambda path: iter(range(3)))
    events = []

    def interpolate(_model, frames, multiplier, scale):
        assert list(frames) == [0, 1, 2]
        yield from range((3 - 1) * multiplier + 1)

    monkeypatch.setattr(pipeline_mlx, "interpolate_stream", interpolate)

    class Writer:
        frames_written = 0

        def __init__(self, path, fps, audio_from):
            assert (path, fps, audio_from) == ("output.mp4", 30 * multi, "input.mp4")

        def __enter__(self):
            return self

        def write(self, frame):
            events.append("write")
            self.frames_written += 1

        def __exit__(self, *args):
            events.append("finalise")

    monkeypatch.setattr(video, "VideoWriter", Writer)
    progress = SimpleNamespace(
        update=lambda count: events.append(("update", count)),
        set_description=lambda desc: events.append(desc))
    factory = MagicMock(return_value=nullcontext(progress))
    monkeypatch.setattr(tqdm, "tqdm", factory)

    pipeline_mlx.cli_main()

    settings = factory.call_args.kwargs
    assert settings["total"] == total
    assert settings["disable"] is (not is_terminal)
    assert settings["file"] is pipeline_mlx.sys.stderr
    output_count = (3 - 1) * multi + 1
    assert events == [event for _ in range(output_count) for event in ("write", ("update", 1))] + [
        "Finalising audio", "finalise", "Done"]
    assert f"({output_count}f @{30 * multi:.2f}, audio passthrough)" in capsys.readouterr().out


@pytest.mark.parametrize("reported_count,expected", [(7, 7), (0, None)])
def test_probe_video_uses_metadata_without_decoding(monkeypatch, reported_count, expected):
    import av

    stream = SimpleNamespace(average_rate=24, frames=reported_count)
    container = SimpleNamespace(streams=SimpleNamespace(video=[stream]))
    monkeypatch.setattr(av, "open", lambda path: nullcontext(container))
    assert video.probe_video("input.mp4") == (24.0, expected)
    assert video.probe_fps("input.mp4") == 24.0
