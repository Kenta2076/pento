"""Image loading and preprocessing utilities for pentomino recognition."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

try:  # pragma: no cover - OpenCV is optional at this stage
    import cv2
except Exception:  # pragma: no cover - gracefully degrade when OpenCV is missing
    cv2 = None  # type: ignore


class PreprocessingError(RuntimeError):
    """Raised when the preprocessing stage fails."""


def load_image(path: Path) -> np.ndarray:
    """Load an image from ``path`` into a NumPy array.

    The implementation attempts to use OpenCV when available; otherwise
    it falls back to Pillow.  For a template project the returned array is
    guaranteed to be ``float32`` regardless of the backend.
    """

    if cv2 is not None:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise PreprocessingError(f"Unable to read image: {path}")
        return image.astype("float32") / 255.0

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise PreprocessingError(
            "Neither OpenCV nor Pillow is available to load images"
        ) from exc

    with Image.open(path) as img:
        return np.asarray(img, dtype="float32") / 255.0


def extract_board_region(image: np.ndarray) -> np.ndarray:
    """Return an image focused on the 6x10 pentomino board region.

    The template implementation simply returns the original image.  When
    replacing this placeholder, consider applying perspective correction,
    contour detection, and color normalization to isolate the board from
    the background.
    """

    if image.ndim != 3:
        raise PreprocessingError("Expected a color image array")
    return image
