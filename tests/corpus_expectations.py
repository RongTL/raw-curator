"""What the planner must decide for specific corpus frames. Extend as rules land."""

EXPECT: dict[str, dict[str, object]] = {
    "IMG_0030": {
        "absent": {"highlight_recover", "tone_map_final"}
    },  # backlit sunset: sigmoid rolls the sun off, nothing left to recover (filmic needed highlight_recover)
    "IMG_1177": {
        "absent": {"tone_map_final", "realesrgan_upscale"}
    },  # bright monument; 6000x4000 at native target -> no SR
    "IMG_0037": {"absent": {"scunet_denoise"}},  # ISO 100: mild noise, no denoise
    "IMG_0098": {"present": {"scunet_denoise"}},  # ISO 12800: high ISO, denoise
    "IMG_0958": {
        "absent": {"white_balance"}
    },  # near-neutral fraction 0.0014 < 2% -> no neutral estimate, no WB
    "IMG_1176": {
        "absent": {"white_balance"}
    },  # neutral cast 0.039 < 0.12 over 6.5% neutral -> mood preserved, no WB
}
