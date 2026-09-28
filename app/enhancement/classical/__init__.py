"""Classical (non-AI) image-processing primitives used by the engine.

All functions take float32 RGB in [0, 1] and return float32 RGB in [0, 1].
Operating in float keeps each step lossless relative to its predecessor;
the only quantisation step is the final write to the display-referred render JPEG.
"""
