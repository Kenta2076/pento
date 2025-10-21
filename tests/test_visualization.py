from __future__ import annotations

import numpy as np

from pento.classification import LabeledPiece
from pento.visualization import format_label_grid, grid_to_labeled_pieces, labeled_pieces_to_grid


def test_format_label_grid_round_trip():
    grid = np.array([
        list("FILPN" + "T" * 5),
        list("UVWXZ" + "Y" * 5),
        list("F" * 10),
        list("I" * 10),
        list("L" * 10),
        list("P" * 10),
    ], dtype="<U1")

    formatted = format_label_grid(grid)
    assert formatted.splitlines()[0].startswith("F I L")

    pieces = grid_to_labeled_pieces(grid)
    reconstructed = labeled_pieces_to_grid(pieces)
    assert np.array_equal(reconstructed, grid)


def test_labeled_pieces_to_grid_handles_partial_masks():
    mask = np.zeros((6, 10), dtype=bool)
    mask[:5, 0] = True
    other_mask = np.zeros_like(mask)
    other_mask[0, :5] = True

    pieces = [
        LabeledPiece(name="X", mask=mask),
        LabeledPiece(name="Y", mask=other_mask),
    ]

    grid = labeled_pieces_to_grid(pieces)
    assert grid[0, 0] == "Y"
    assert grid[1, 0] == "X"
