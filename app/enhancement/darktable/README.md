# darktable baseline sidecar

`raw-curator-base.xmp` is the **baseline develop** the enhancement engine applies
when a photo has no user-authored sidecar. `app/enhancement/sidecar.py` resolves
it as `BASELINE_XMP` and `app/enhancement/batch.py` passes it to `darktable-cli`;
a user's own `<name>.<ext>.xmp` (mirrored under `xmp/`) always wins over it.

It is a look-neutral, camera-agnostic starting point — the corrections that
belong on every frame regardless of subject — not a develop of any one image.
The engine's own rule table (`app/enhancement/engine/decision.py`) does the
per-photo, subject-aware work on top of this.

## Modules it sets

Authored in the darktable darkroom on `IMG_1177.CR3` (bright, ISO 100) under the
`scene-referred (sigmoid)` workflow, then saved as a sidecar:

| Module | Setting |
|---|---|
| lens correction | method *lensfun*, camera/lens *auto*, corrections *all* |
| chromatic aberrations | `cacorrectrgb`, defaults |
| highlight reconstruction | method *inpaint opposed* |
| denoise (profiled) | *wavelets auto*, profile *auto* (ISO-dependent), default strength |
| color calibration | adaptation *CAT16*, illuminant *as shot in camera* |
| exposure | scene-referred default (+0.7 EV with compensation), not hand-tuned |
| sigmoid | contrast *1.5*, skew *0*, preservation *per channel* |
| white balance (temperature) | *as shot* — left untouched |

Crop, rotate, local edits and the white-balance sliders are deliberately not
touched. The `lensfun/` subfolder ships extra lens calibration XML that
`liblensfun-data-v1` lacks; the `Containerfile` copies it into the image so both
the GUI and `darktable-cli` resolve those lenses.

## Re-authoring it

Launch darktable from the image with the corpus and this folder mounted (needs
the host X socket; run as yourself):

```bash
xhost +local: && podman run --rm -it --entrypoint darktable \
  -e DISPLAY=$DISPLAY -e LIBGL_ALWAYS_SOFTWARE=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v "$PWD/tests/data/corpus":/corpus:z \
  -v "$PWD/app/enhancement/darktable":/styles:z \
  localhost/raw-curator:latest /corpus/IMG_1177.CR3 \
  --conf plugins/darkroom/workflow="scene-referred (sigmoid)"
```

Set the modules above, then in lighttable select the frame and **write sidecar
files** (writes `IMG_1177.CR3.xmp` next to the RAW). Copy that file over
`raw-curator-base.xmp`, remove the image-specific attributes
(`xmpMM:DerivedFrom`, `darktable:import_timestamp`, `darktable:change_timestamp`),
and confirm it develops a *different* frame without error:

```bash
darktable-cli /corpus/IMG_0098.CR3 /styles/raw-curator-base.xmp /tmp/t.tif \
  --icc-type LIN_REC2020 --core
```

The result should be a full-resolution 16-bit TIFF with lifted, clean shadows.
