"""Grid segmentation utilities for pentomino boards."""

from __future__ import annotations

import logging
from typing import List

import numpy as np

GRID_HEIGHT = 6
GRID_WIDTH = 10

logger = logging.getLogger(__name__)


class SegmentationError(RuntimeError):
    """Raised when the grid segmentation step cannot succeed."""


def segment_grid(board_image: np.ndarray) -> np.ndarray:
    """Segment the board into ``GRID_HEIGHT``×``GRID_WIDTH`` cells.

    The segmentation assumes that ``board_image`` already represents a
    top-down view of the board.  The board may contain small margins around
    the playable area – these are trimmed automatically so that each cell
    has the same dimensions.
    """

    if board_image.ndim != 3:
        raise SegmentationError("Board image must be an H×W×C array")

    height, width, channels = board_image.shape

    if height < GRID_HEIGHT or width < GRID_WIDTH:
        raise SegmentationError("Board image is smaller than the expected grid dimensions")

    cell_height = height // GRID_HEIGHT
    cell_width = width // GRID_WIDTH

    if cell_height == 0 or cell_width == 0:
        raise SegmentationError("Board image is too small to segment into cells")

    margin_y = (height - cell_height * GRID_HEIGHT) // 2
    margin_x = (width - cell_width * GRID_WIDTH) // 2

    cells: List[np.ndarray] = []

    for row in range(GRID_HEIGHT):
        row_cells: List[np.ndarray] = []
        y_start = margin_y + row * cell_height
        y_end = y_start + cell_height

        if y_end > height:
            logger.error("Row %d exceeds board height (start=%d, end=%d, height=%d)", row, y_start, y_end, height)
            raise SegmentationError("Failed to determine vertical bounds for cell segmentation")

        for col in range(GRID_WIDTH):
            x_start = margin_x + col * cell_width
            x_end = x_start + cell_width

            if x_end > width:
                logger.error(
                    "Cell (%d, %d) exceeds board width (start=%d, end=%d, width=%d)",
                    row,
                    col,
                    x_start,
                    x_end,
                    width,
                )
                raise SegmentationError("Failed to determine horizontal bounds for cell segmentation")

            cell = board_image[y_start:y_end, x_start:x_end]

            if cell.shape[0] != cell_height or cell.shape[1] != cell_width:
                logger.error(
                    "Cell (%d, %d) has unexpected shape %s (expected %dx%d)",
                    row,
                    col,
                    cell.shape,
                    cell_height,
                    cell_width,
                )
                raise SegmentationError("Cell dimensions do not match expected size")

            row_cells.append(cell)

        cells.append(np.stack(row_cells, axis=0))

    grid = np.stack(cells, axis=0)

    expected_shape = (GRID_HEIGHT, GRID_WIDTH, cell_height, cell_width, channels)
    if grid.shape != expected_shape:
        logger.error("Segmented grid has unexpected shape %s (expected %s)", grid.shape, expected_shape)
        raise SegmentationError("Segmented grid has unexpected dimensions")

    return grid
