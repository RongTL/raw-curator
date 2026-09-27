"""Identity guard: revert a restored face whose ArcFace embedding drifted.

The guard compares the *before* vs *after* crop through the same embedder, so
the cosine reflects only what CodeFormer changed. A fake embedder stands in for
ArcFace: it maps dark crops and bright crops to orthogonal (drift) or identical
(match) vectors, so no model weights are needed.
"""

from __future__ import annotations

import numpy as np

from app.arrays import Array


def _dark_bright_orthogonal(crop: Array) -> Array:
    """Dark crops -> [1, 0]; bright crops -> [0, 1] (cosine 0 => drift)."""
    if float(crop.mean()) < 128:
        return np.array([1.0, 0.0], dtype=np.float32)
    return np.array([0.0, 1.0], dtype=np.float32)


def _always_same(crop: Array) -> Array:
    """Every crop -> [1, 0] (cosine 1 => match)."""
    return np.array([1.0, 0.0], dtype=np.float32)


def test_identity_guard_reverts_a_face_that_drifted() -> None:
    from app.enhancement.face_restore import guard_faces

    before = np.zeros((100, 100, 3), dtype=np.uint8)
    after = before.copy()
    after[10:50, 10:50] = 255
    out = guard_faces(
        before, after, [(10, 10, 40, 40)], embed=_dark_bright_orthogonal, min_similarity=0.5
    )
    assert int(out[20, 20, 0]) == 0  # reverted to the original (dark) crop


def test_identity_guard_keeps_a_face_that_matches() -> None:
    from app.enhancement.face_restore import guard_faces

    before = np.zeros((100, 100, 3), dtype=np.uint8)
    after = before.copy()
    after[10:50, 10:50] = 255
    out = guard_faces(before, after, [(10, 10, 40, 40)], embed=_always_same, min_similarity=0.5)
    assert int(out[20, 20, 0]) == 255  # kept the restored (bright) crop


def test_identity_guard_skips_a_box_outside_the_image() -> None:
    from app.enhancement.face_restore import guard_faces

    before = np.zeros((100, 100, 3), dtype=np.uint8)
    after = before.copy()
    after[10:50, 10:50] = 255
    calls: list[Array] = []

    def counting_embed(crop: Array) -> Array:
        calls.append(crop)
        return np.array([1.0, 0.0], dtype=np.float32)

    out = guard_faces(before, after, [(200, 200, 40, 40)], embed=counting_embed, min_similarity=0.5)
    assert calls == []  # empty crop is skipped; embed is never called
    assert np.array_equal(out, after)


def test_identity_guard_does_not_modify_after_in_place() -> None:
    from app.enhancement.face_restore import guard_faces

    before = np.zeros((100, 100, 3), dtype=np.uint8)
    after = before.copy()
    after[10:50, 10:50] = 255
    snapshot = after.copy()
    out = guard_faces(
        before, after, [(10, 10, 40, 40)], embed=_dark_bright_orthogonal, min_similarity=0.5
    )
    assert np.array_equal(after, snapshot)  # input `after` untouched
    assert int(out[20, 20, 0]) == 0  # the copy was reverted
