"""Tests for face blurring geometry with a fake face detector (no weights needed)."""

import numpy as np

from line_counter.privacy import FACE_BATCH_SIZE, FaceBlurrer


class _XYXY:
    def __init__(self, a):
        self.a = np.asarray(a, dtype=float).reshape(-1, 4)

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class _Res:
    def __init__(self, faces):
        self.boxes = None if faces is None else types_namespace(_XYXY(faces))


def types_namespace(xyxy):
    class B:
        pass
    b = B()
    b.xyxy = xyxy
    return b


class FakeFaceModel:
    """Reports the same face boxes (crop coordinates) for every crop and records the batch sizes."""

    def __init__(self, faces):
        self.faces, self.batches = faces, []

    def __call__(self, crops, verbose=False, conf=0.3):
        self.batches.append(len(crops))
        return [_Res(self.faces) for _ in crops]


def make_blurrer(faces):
    b = object.__new__(FaceBlurrer)  # skip loading real weights
    b.model, b.conf, b.kernel, b.sigma = FakeFaceModel(faces), 0.3, (9, 9), 3.0
    return b


def noisy_frame(h=100, w=100):
    return np.random.default_rng(0).integers(0, 255, (h, w, 3), dtype=np.uint8)


def test_blur_only_changes_the_face_region():
    frame = noisy_frame()
    before = frame.copy()
    n = make_blurrer([[10, 10, 30, 30]]).blur_in_boxes(frame, [(40, 20, 90, 80)])
    assert n == 1
    changed = np.argwhere((frame != before).any(axis=2))
    assert changed[:, 0].min() >= 20 + 10 and changed[:, 0].max() < 20 + 30
    assert changed[:, 1].min() >= 40 + 10 and changed[:, 1].max() < 40 + 30


def test_person_box_outside_or_partly_outside_the_frame_is_clipped():
    frame = noisy_frame()
    b = make_blurrer([[0, 0, 500, 500]])  # face box larger than the crop: clipped to it
    assert b.blur_in_boxes(frame, [(-30, -30, 20, 20), (90, 90, 400, 400), (200, 200, 300, 300)]) == 2


def test_degenerate_boxes_are_skipped():
    frame = noisy_frame()
    before = frame.copy()
    b = make_blurrer([[0, 0, 5, 5]])
    assert b.blur_in_boxes(frame, [(10, 10, 10, 50), (30, 30, 20, 40)]) == 0
    assert np.array_equal(frame, before) and b.model.batches == []


def test_empty_and_degenerate_face_boxes_blur_nothing():
    frame = noisy_frame()
    before = frame.copy()
    assert make_blurrer([[5, 5, 5, 9], [8, 8, 2, 2]]).blur_in_boxes(frame, [(0, 0, 50, 50)]) == 0
    assert np.array_equal(frame, before)
    assert make_blurrer(np.zeros((0, 4))).blur_in_boxes(frame, [(0, 0, 50, 50)]) == 0


def test_crops_are_sent_in_batches():
    b = make_blurrer([])
    b.blur_in_boxes(noisy_frame(), [(0, 0, 10, 10)] * (FACE_BATCH_SIZE * 2 + 3))
    assert b.model.batches == [FACE_BATCH_SIZE, FACE_BATCH_SIZE, 3]


def test_a_result_without_boxes_is_ignored():
    b = make_blurrer(None)
    assert b.blur_in_boxes(noisy_frame(), [(0, 0, 50, 50)]) == 0


def test_colour_order_is_preserved():
    frame = np.zeros((40, 40, 3), np.uint8)
    frame[..., 0] = 200  # pure blue in BGR
    make_blurrer([[0, 0, 40, 40]]).blur_in_boxes(frame, [(0, 0, 40, 40)])
    assert frame[..., 0].min() == 200 and frame[..., 1:].max() == 0
