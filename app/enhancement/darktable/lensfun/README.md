# Extra lensfun calibration data

The image installs `liblensfun-data-v1` **0.3.4**, whose bundled database
(`/usr/share/lensfun/version_1/`) predates some recent lenses. The `*.xml` files
here fill those gaps; the `Containerfile` copies them into
`/usr/share/lensfun/version_1/` so **both** the darktable GUI and the
`darktable-cli` develop step (`app/enhancement/develop_full.py`) resolve them.
lensfun loads every `*.xml` in that directory, so no index needs updating.

## Files

- `mil-canon-rf24.xml` — Canon RF 24mm F1.8 MACRO IS STM (both calibrated
  variants: cropfactor 1.613 for APS-C bodies, 1.0 for full-frame).

## Compatibility

The bundled data package ships DB format **version 1**, while lensfun's upstream
`master` is **version 2**. These files are therefore wrapped as
`<lensdatabase version="1">` — verified to load cleanly against the installed
liblensfun 0.3.4 (`lf_db_load` returns `LF_NO_ERROR` and the lens resolves).
Only use v1-compatible calibration elements here (`distortion`, `tca`,
`vignetting`, `cropfactor`); do not paste blocks that rely on v2-only features.

## Refreshing / adding a lens

Lens data lives in lensfun's upstream DB. To add or update a lens:

1. Fetch the relevant file, e.g.
   `curl -fsSL https://raw.githubusercontent.com/lensfun/lensfun/master/data/db/mil-canon.xml`
   (mirrorless Canon; SLR lenses are in `slr-canon.xml`, other makers analogous).
2. Copy the matching `<lens>...</lens>` block(s) into a file here, wrapped in
   `<!DOCTYPE lensdatabase SYSTEM "lensfun-database.dtd">` +
   `<lensdatabase version="1"> ... </lensdatabase>`. Keep every calibrated
   variant of the model (they differ by `<cropfactor>`). The `<mount>` (e.g.
   `Canon RF`) is already defined by the shipped DB — do not redefine it.
3. Rebuild the image (`make image-warm`) and confirm the lens resolves.
