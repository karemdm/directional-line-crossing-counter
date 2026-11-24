"""Tests for the processing loop with a synthetic video and a fake detector (no real model)."""

import json
import sys
import types

import cv2
import numpy as np
import pytest

from line_counter.config import ProcessingConfig
from line_counter.counter import CountingLine
from line_counter.pipeline import FALLBACK_FPS, _detect, process_video
from line_counter.tracks_io import load_tracks, track_record, write_frame_tracks

W, H, N_FRAMES, FPS = 160, 120, 12, 10.0


def make_video(path, n=N_FRAMES, fps=FPS):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for _ in range(n):
        writer.write(np.full((H, W, 3), 90, np.uint8))
    writer.release()
    return path


def write_tracks(path, frames):
    """frames: {frame_index: [(id, cls, name, box), ...]}"""
    with open(path, "w", encoding="utf-8") as fh:
        for f, tracks in frames.items():
            write_frame_tracks(fh, f, [track_record(i, c, n, 0.9, b) for i, c, n, b in tracks])


def config(**kw):
    kw.setdefault("display", False)
    kw.setdefault("layout", "overlay")
    return ProcessingConfig(**kw)


def horizontal_line():
    return [CountingLine("Line_1", (0, 60), (W, 60))]


def walker(y_bottoms, start=1, step=1):
    """Box tracks of person #1 whose bottom edge is at each given y."""
    return {start + i * step: [(1, 0, "person", (70, y - 20, 90, y))] for i, y in enumerate(y_bottoms)}


def video_props(path):
    cap = cv2.VideoCapture(str(path))
    props = (cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS), cap.get(3), cap.get(4))
    cap.release()
    return props


def test_replay_counts_and_writes_every_frame(tmp_path):
    video = make_video(tmp_path / "v.mp4")
    tracks = tmp_path / "t.jsonl"
    write_tracks(tracks, walker([40, 50, 70, 80], start=3))
    out = tmp_path / "out.mp4"
    counter = process_video(video, out, horizontal_line(), config(replay_tracks=str(tracks)))
    assert counter.lines[0].counts["person"] == {"IN": 1, "OUT": 0}
    frames, _, w, h = video_props(out)
    assert (w, h) == (W, H) and frames == 4  # only the frames present in the tracks file


def test_replay_stops_decoding_after_the_last_track_frame(tmp_path, monkeypatch):
    video = make_video(tmp_path / "v.mp4", n=40)
    tracks = tmp_path / "t.jsonl"
    write_tracks(tracks, walker([40, 70]))
    reads = []
    real_capture = cv2.VideoCapture

    class Spy:  # a proxy, not a subclass: subclassing cv2.VideoCapture corrupts memory
        def __init__(self, *args):
            self._cap = real_capture(*args)

        def read(self):
            reads.append(1)
            return self._cap.read()

        def __getattr__(self, name):
            return getattr(self._cap, name)

    monkeypatch.setattr(cv2, "VideoCapture", Spy)
    process_video(video, tmp_path / "o.mp4", horizontal_line(), config(replay_tracks=str(tracks)))
    assert len(reads) <= 3  # no point decoding the 38 remaining frames


def test_replay_frame_step_is_taken_from_the_file_and_divides_output_fps(tmp_path):
    video = make_video(tmp_path / "v.mp4")
    tracks = tmp_path / "t.jsonl"
    write_tracks(tracks, walker([40, 50, 70, 80], start=1, step=3))
    out = tmp_path / "o.mp4"
    process_video(video, out, horizontal_line(), config(replay_tracks=str(tracks)))
    frames, fps, _, _ = video_props(out)
    assert frames == 4 and fps == pytest.approx(FPS / 3, rel=0.05)


def test_zero_length_or_outside_lines_never_count(tmp_path):
    video = make_video(tmp_path / "v.mp4")
    tracks = tmp_path / "t.jsonl"
    write_tracks(tracks, walker([40, 70, 90]))
    lines = [CountingLine("dot", (50, 60), (50, 60)), CountingLine("far", (0, 500), (W, 500))]
    counter = process_video(video, tmp_path / "o.mp4", lines, config(replay_tracks=str(tracks)))
    assert counter.summary() == {"dot": {"person": {"IN": 0, "OUT": 0}},
                                 "far": {"person": {"IN": 0, "OUT": 0}}}


def test_empty_tracks_file_is_not_an_error(tmp_path):
    video = make_video(tmp_path / "v.mp4")
    tracks = tmp_path / "t.jsonl"
    tracks.write_text("", encoding="utf-8")
    counter = process_video(video, tmp_path / "o.mp4", horizontal_line(), config(replay_tracks=str(tracks)))
    assert counter.summary() == {"Line_1": {}}


def test_video_without_frames_is_an_error(tmp_path):
    video = tmp_path / "empty.mp4"
    cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (W, H)).release()
    tracks = tmp_path / "t.jsonl"
    tracks.write_text("", encoding="utf-8")
    with pytest.raises(RuntimeError):
        process_video(video, tmp_path / "o.mp4", horizontal_line(), config(replay_tracks=str(tracks)))


def test_missing_video_is_an_error(tmp_path):
    tracks = tmp_path / "t.jsonl"
    tracks.write_text("", encoding="utf-8")
    with pytest.raises(RuntimeError):
        process_video(tmp_path / "nope.mp4", tmp_path / "o.mp4", horizontal_line(),
                      config(replay_tracks=str(tracks)))


def test_class_names_with_the_same_label_are_listed_once(tmp_path):
    video = make_video(tmp_path / "v.mp4")
    tracks = tmp_path / "t.jsonl"
    write_tracks(tracks, {1: [(1, 2, "vehicle", (10, 10, 30, 30)), (2, 5, "vehicle", (50, 10, 70, 30))]})
    counter = process_video(video, tmp_path / "o.mp4", horizontal_line(), config(replay_tracks=str(tracks)))
    assert list(counter.lines[0].counts) == ["vehicle"]


def test_fallback_fps_is_positive():
    assert FALLBACK_FPS > 0


# ---- live path with a fake detector -------------------------------------------------------------

class _Arr:
    def __init__(self, a):
        self.a = np.asarray(a)

    def cpu(self):
        return self

    def numpy(self):
        return self.a

    def __len__(self):
        return len(self.a)


class _Boxes:
    def __init__(self, xyxy, ids, cls, conf):
        self.xyxy, self.cls, self.conf = _Arr(xyxy), _Arr(cls), _Arr(conf)
        self.id = None if ids is None else _Arr(ids)

    def __len__(self):
        return len(self.xyxy)


class _Result:
    def __init__(self, boxes):
        self.boxes = boxes


class FakeModel:
    """Returns the scripted detections of call n (0-based) from ``script``."""

    names = {0: "person", 2: "car"}

    def __init__(self, script):
        self.script, self.calls = script, 0

    def track(self, frame, **kwargs):
        res = self.script[self.calls] if self.calls < len(self.script) else None
        self.calls += 1
        return [_Result(res)] if res is not None else []


def fake_ultralytics(monkeypatch, model):
    module = types.ModuleType("ultralytics")
    module.YOLO = lambda path: model
    monkeypatch.setitem(sys.modules, "ultralytics", module)


def person_script(y_bottoms):
    return [_Boxes([[70, y - 20, 90, y]], [1], [0], [0.9]) for y in y_bottoms]


def test_detect_handles_no_result_no_boxes_and_missing_ids():
    cfg = config()
    assert _detect(FakeModel([None]), None, cfg, None) == []
    assert _detect(FakeModel([_Boxes(np.zeros((0, 4)), [], [], [])]), None, cfg, None) == []
    assert _detect(FakeModel([_Boxes([[1, 2, 3, 4]], None, [0], [0.5])]), None, cfg, None) == []


def test_detect_returns_id_class_name_conf_and_box():
    out = _detect(FakeModel(person_script([50])), None, config(), None)
    (tid, cid, name, conf, box), = out
    assert (tid, cid, name) == (1, 0, "person") and conf == pytest.approx(0.9)
    assert list(box) == [70, 30, 90, 50]


def test_live_run_with_frame_step_saves_tracks_counts_and_fps(tmp_path, monkeypatch):
    video = make_video(tmp_path / "v.mp4")
    fake_ultralytics(monkeypatch, FakeModel(person_script([40, 50, 70, 80])))
    out, tracks = tmp_path / "o.mp4", tmp_path / "t.jsonl"
    counter = process_video(video, out, horizontal_line(),
                            config(frame_step=3, save_tracks=str(tracks)))
    assert counter.lines[0].counts["person"] == {"IN": 1, "OUT": 0}
    assert sorted(load_tracks(tracks)) == [1, 4, 7, 10]  # 1-based source frame indices
    frames, fps, _, _ = video_props(out)
    assert frames == 4 and fps == pytest.approx(FPS / 3, rel=0.05)


def test_replay_reproduces_the_live_run(tmp_path, monkeypatch):
    video = make_video(tmp_path / "v.mp4")
    script = person_script([40, 50, 70, 80, 70, 40]) + [
        _Boxes([[10, 30, 30, 50], [100, 70, 120, 90]], [2, 3], [2, 0], [0.8, 0.7])]
    fake_ultralytics(monkeypatch, FakeModel(script))
    tracks = tmp_path / "t.jsonl"
    live = process_video(video, tmp_path / "live.mp4", horizontal_line(),
                         config(frame_step=2, save_tracks=str(tracks)))
    replay = process_video(video, tmp_path / "replay.mp4", horizontal_line(),
                           config(replay_tracks=str(tracks)))
    # Replay only knows the classes present in the file, so the zero-count buckets of
    # classes that never appeared differ; the real counts must be identical.
    assert _nonzero(live.summary()) == _nonzero(replay.summary())
    assert _nonzero(live.summary())["Line_1"]  # something was actually counted
    assert video_props(tmp_path / "live.mp4") == video_props(tmp_path / "replay.mp4")
    live_frames, replay_frames = (_read_all(tmp_path / n) for n in ("live.mp4", "replay.mp4"))
    assert len(live_frames) == len(replay_frames)
    for a, b in zip(live_frames, replay_frames):
        assert np.array_equal(a, b)


def _nonzero(summary):
    return {line: {c: d for c, d in per.items() if d["IN"] or d["OUT"]} for line, per in summary.items()}


def _read_all(path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames


def test_tracks_file_is_closed_when_the_writer_fails(tmp_path, monkeypatch):
    video = make_video(tmp_path / "v.mp4")
    fake_ultralytics(monkeypatch, FakeModel(person_script([40])))
    bad_dir = tmp_path / "blocker"
    bad_dir.write_text("a file, not a folder", encoding="utf-8")
    with pytest.raises(OSError):
        process_video(video, tmp_path / "o.mp4", horizontal_line(),
                      config(save_tracks=str(bad_dir / "t.jsonl")))
    # the source video must still be deletable (capture released)
    video.unlink()


def test_counts_json_matches_documented_shape(tmp_path):
    video = make_video(tmp_path / "v.mp4")
    tracks = tmp_path / "t.jsonl"
    write_tracks(tracks, walker([40, 70]))
    counter = process_video(video, tmp_path / "o.mp4", horizontal_line(), config(replay_tracks=str(tracks)))
    assert json.loads(json.dumps(counter.summary())) == {"Line_1": {"person": {"IN": 1, "OUT": 0}}}


def test_rider_overlap_skips_a_person_on_a_motorcycle_in_replay(tmp_path):
    video = make_video(tmp_path / "v.mp4")
    tracks = tmp_path / "t.jsonl"
    frames = {}
    for i, y in enumerate([40, 50, 70, 80], start=1):
        frames[i] = [(1, 0, "person", (70, y - 30, 90, y)), (2, 3, "motorcycle", (60, y - 18, 100, y + 5))]
    write_tracks(tracks, frames)
    plain = process_video(video, tmp_path / "a.mp4", horizontal_line(), config(replay_tracks=str(tracks)))
    assert plain.lines[0].counts["person"] == {"IN": 1, "OUT": 0}
    riders = process_video(video, tmp_path / "b.mp4", horizontal_line(),
                           config(replay_tracks=str(tracks), rider_overlap=0.5))
    assert riders.lines[0].counts["person"] == {"IN": 0, "OUT": 0}
    assert riders.lines[0].counts["motorcycle"] == {"IN": 1, "OUT": 0}


def test_min_cross_distance_in_replay_ignores_jitter_on_the_line(tmp_path):
    video = make_video(tmp_path / "v.mp4")
    tracks = tmp_path / "t.jsonl"
    write_tracks(tracks, walker([58, 62, 59, 63, 58, 62], start=1))  # boxes 20 px high
    plain = process_video(video, tmp_path / "a.mp4", horizontal_line(), config(replay_tracks=str(tracks)))
    assert plain.lines[0].counts["person"]["IN"] == 1
    hyst = process_video(video, tmp_path / "b.mp4", horizontal_line(),
                         config(replay_tracks=str(tracks), min_cross_distance=0.5))
    assert hyst.lines[0].counts["person"] == {"IN": 0, "OUT": 0}
