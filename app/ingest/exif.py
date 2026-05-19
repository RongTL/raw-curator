"""EXIF extraction via PyExifTool daemon mode."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import exiftool


@dataclass
class ExifData:
    camera_make: str | None = None
    camera_body: str | None = None
    lens: str | None = None
    captured_at: datetime | None = None
    width: int | None = None
    height: int | None = None
    iso: int | None = None
    shutter: float | None = None
    aperture: float | None = None
    focal_length: float | None = None
    orientation: int | None = None


_DATETIME_FORMATS = ("%Y:%m:%d %H:%M:%S.%f", "%Y:%m:%d %H:%M:%S")


def _parse_dt(value: Any) -> datetime | None:
    if value is None:
        return None
    text = str(value).split("+", 1)[0].strip()
    for fmt in _DATETIME_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _parse_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if isinstance(value, str) and "/" in value:
            num, denom = value.split("/", 1)
            denom_v = float(denom)
            return float(num) / denom_v if denom_v else None
        return float(value)
    except (ValueError, ZeroDivisionError):
        return None


def _parse_int(value: Any) -> int | None:
    try:
        return int(float(value)) if value is not None else None
    except (TypeError, ValueError):
        return None


class ExifReader:
    def __init__(self) -> None:
        self._et: exiftool.ExifToolHelper | None = None

    def __enter__(self) -> ExifReader:
        self._et = exiftool.ExifToolHelper()
        self._et.__enter__()
        return self

    def __exit__(self, *exc: object) -> None:
        if self._et is not None:
            self._et.__exit__(*exc)
            self._et = None

    def read(self, path: Path) -> ExifData:
        assert self._et is not None, "use ExifReader as a context manager"
        rows = self._et.get_metadata([str(path)])
        if not rows:
            return ExifData()
        m = rows[0]
        return ExifData(
            camera_make=m.get("EXIF:Make") or m.get("Make"),
            camera_body=m.get("EXIF:Model") or m.get("Model"),
            lens=m.get("EXIF:LensModel") or m.get("Composite:LensID"),
            captured_at=_parse_dt(
                m.get("EXIF:DateTimeOriginal")
                or m.get("EXIF:CreateDate")
                or m.get("DateTimeOriginal")
            ),
            width=_parse_int(m.get("EXIF:ImageWidth") or m.get("File:ImageWidth")),
            height=_parse_int(m.get("EXIF:ImageHeight") or m.get("File:ImageHeight")),
            iso=_parse_int(m.get("EXIF:ISO") or m.get("ISO")),
            shutter=_parse_float(m.get("EXIF:ExposureTime") or m.get("ExposureTime")),
            aperture=_parse_float(m.get("EXIF:FNumber") or m.get("FNumber")),
            focal_length=_parse_float(m.get("EXIF:FocalLength") or m.get("FocalLength")),
            orientation=_parse_int(m.get("EXIF:Orientation") or m.get("Orientation")),
        )
