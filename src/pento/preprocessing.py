"""Image loading and preprocessing utilities for pentomino recognition."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

try:  # pragma: no cover - OpenCV is optional at this stage
    import cv2
except Exception:  # pragma: no cover - gracefully degrade when OpenCV is missing
    cv2 = None  # type: ignore


logger = logging.getLogger(__name__)


class PreprocessingError(RuntimeError):
    """Raised when the preprocessing stage fails."""


def _describe_image(array: np.ndarray) -> dict[str, float | tuple[int, ...] | str]:
    """Return a dictionary with basic statistics about an image array."""

    stats: dict[str, float | tuple[int, ...] | str] = {
        "shape": array.shape,
        "dtype": str(array.dtype),
    }

    if array.size:
        stats["min"] = float(np.min(array))
        stats["max"] = float(np.max(array))
        stats["mean"] = float(np.mean(array))
    return stats


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
        result = image.astype("float32") / 255.0
        logger.info("Loaded image %s with stats %s", path, _describe_image(result))
        return result

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise PreprocessingError(
            "Neither OpenCV nor Pillow is available to load images"
        ) from exc

    with Image.open(path) as img:
        array = np.asarray(img, dtype="float32") / 255.0
        logger.info("Loaded image %s with stats %s", path, _describe_image(array))
        return array


def _order_corners(points: np.ndarray) -> np.ndarray:
    """Return ``points`` ordered as top-left, top-right, bottom-right, bottom-left."""

    pts = points.reshape(4, 2).astype("float32")
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).ravel()

    ordered = np.zeros((4, 2), dtype="float32")
    ordered[0] = pts[np.argmin(sums)]  # top-left
    ordered[2] = pts[np.argmax(sums)]  # bottom-right
    ordered[1] = pts[np.argmin(diffs)]  # top-right
    ordered[3] = pts[np.argmax(diffs)]  # bottom-left
    return ordered


def extract_board_region(image: np.ndarray) -> np.ndarray:
    """Return an image focused on the 6x10 pentomino board region.

    The board is isolated via contour detection and perspective correction.
    The resulting view is contrast-enhanced to highlight unit cells.
    """

    if image.ndim != 3:
        logger.error("extract_board_region expected a color image array, got %s dimensions", image.ndim)
        raise PreprocessingError("Expected a color image array")

    if cv2 is None:  # pragma: no cover - OpenCV is optional
        logger.error("OpenCV is required for board extraction but is not available")
        raise PreprocessingError("OpenCV is required for board extraction")

    # Ensure we operate on 8-bit values for OpenCV routines.
    if image.dtype != np.uint8:
        image_uint8 = np.clip(image * 255.0, 0, 255).astype("uint8")
    else:
        image_uint8 = image

    blurred = cv2.GaussianBlur(image_uint8, (5, 5), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        logger.error("No contours detected while attempting to locate the board")
        raise PreprocessingError("Unable to detect board contour")

    board_contour: Optional[np.ndarray] = None
    max_area = 0.0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area <= max_area:
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        if len(approx) == 4 and cv2.isContourConvex(approx):
            max_area = area
            board_contour = approx

    if board_contour is None:
        logger.error("Unable to locate a quadrilateral contour representing the board")
        raise PreprocessingError("Unable to locate board region")

    ordered = _order_corners(board_contour)

    board_width = 1000
    board_height = int(board_width * 6 / 10)
    destination = np.array(
        [
            [0, 0],
            [board_width - 1, 0],
            [board_width - 1, board_height - 1],
            [0, board_height - 1],
        ],
        dtype="float32",
    )

    transform = cv2.getPerspectiveTransform(ordered, destination)
    warped = cv2.warpPerspective(image_uint8, transform, (board_width, board_height))

    # Enhance contrast to make unit cells easier to segment.
    lab = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    lab = cv2.merge([l_channel, a_channel, b_channel])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    enhanced = cv2.bilateralFilter(enhanced, d=9, sigmaColor=75, sigmaSpace=75)

    result = enhanced.astype("float32") / 255.0
    logger.info("Extracted board region with stats %s", _describe_image(result))
    return result
