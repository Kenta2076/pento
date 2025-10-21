from __future__ import annotations

import numpy as np

from pento import segmentation


def test_segment_grid_returns_expected_shape():
    cell_height, cell_width = 10, 12
    board_image = np.random.rand(cell_height * segmentation.GRID_HEIGHT, cell_width * segmentation.GRID_WIDTH, 3)

    grid = segmentation.segment_grid(board_image)

    assert grid.shape == (
        segmentation.GRID_HEIGHT,
        segmentation.GRID_WIDTH,
        cell_height,
        cell_width,
        3,
    )
    np.testing.assert_allclose(grid[0, 0], board_image[:cell_height, :cell_width])
