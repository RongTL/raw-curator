from app.enhancement.geometry import scale_boxes


def test_scale_boxes_maps_preview_to_native_and_clamps() -> None:
    boxes = [(300, 200, 100, 50), (2900, 1950, 200, 100)]
    out = scale_boxes(boxes, src_size=(3000, 2000), dst_size=(6000, 4000))
    assert out[0] == (600, 400, 200, 100)
    assert out[1] == (5800, 3900, 200, 100)  # clamped to the frame


def test_scale_boxes_identity_when_sizes_match() -> None:
    assert scale_boxes([(1, 2, 3, 4)], (10, 10), (10, 10)) == [(1, 2, 3, 4)]
