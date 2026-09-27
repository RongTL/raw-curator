"""What the planner must decide for specific corpus frames. Extend as rules land."""

EXPECT: dict[str, dict[str, object]] = {
    "IMG_0030": {"present": {"highlight_recover"}, "absent": {"tone_map_final"}},  # backlit sunset
    "IMG_1177": {"absent": {"tone_map_final"}},  # bright monument
}
