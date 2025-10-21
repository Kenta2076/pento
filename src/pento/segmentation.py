"""Grid segmentation utilities for pentomino boards."""

from __future__ import annotations

import numpy as np

GRID_HEIGHT = 6
GRID_WIDTH = 10


class SegmentationError(RuntimeError):
    """Raised when the grid segmentation step cannot succeed."""


def segment_grid(board_image: np.ndarray) -> np.ndarray:
    """Segment the board into ``GRID_HEIGHT``×``GRID_WIDTH`` cells.

    The template implementation simply creates an empty grid with the
    expected shape.  Replace this placeholder with logic that detects the
    board boundaries, applies perspective correction, and extracts each
    cell as an image tile.
    """

    if board_image.ndim != 3:
        raise SegmentationError("Board image must be an H×W×C array")

    return np.zeros((GRID_HEIGHT, GRID_WIDTH, *board_image.shape[2:]), dtype=board_image.dtype)
