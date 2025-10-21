import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from pento.segmentation import GRID_HEIGHT, GRID_WIDTH, SegmentationError, segment_grid


def test_segment_grid_returns_expected_number_of_cells():
    cell_height = 12
    cell_width = 15
    margin_y = 2
    margin_x = 3

    height = GRID_HEIGHT * cell_height + 2 * margin_y
    width = GRID_WIDTH * cell_width + 2 * margin_x

    board = np.zeros((height, width, 3), dtype=np.uint8)

    for row in range(GRID_HEIGHT):
        for col in range(GRID_WIDTH):
            y_start = margin_y + row * cell_height
            y_end = y_start + cell_height
            x_start = margin_x + col * cell_width
            x_end = x_start + cell_width
            color = np.array([(row + 1) * 10, (col + 1) * 5, (row + col + 1) * 3], dtype=np.uint8)
            board[y_start:y_end, x_start:x_end] = color

    grid = segment_grid(board)

    assert grid.shape == (GRID_HEIGHT, GRID_WIDTH, cell_height, cell_width, 3)

    for row in range(GRID_HEIGHT):
        for col in range(GRID_WIDTH):
            expected = np.array([(row + 1) * 10, (col + 1) * 5, (row + col + 1) * 3], dtype=np.uint8)
            cell = grid[row, col]
            assert np.all(cell == expected)


def test_segment_grid_raises_error_for_invalid_input():
    board = np.zeros((10, 10), dtype=np.uint8)

    with pytest.raises(SegmentationError):
        segment_grid(board)
