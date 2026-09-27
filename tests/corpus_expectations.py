"""What the planner must decide for specific corpus frames. Extend as rules land."""

EXPECT: dict[str, dict[str, object]] = {
    "IMG_0030": {
        "absent": {"highlight_recover", "tone_map_final"}
    },  # backlit sunset: sigmoid rolls the sun off, nothing left to recover (filmic needed highlight_recover)
    "IMG_1177": {"absent": {"tone_map_final"}},  # bright monument
}
