from __future__ import annotations

import random
from typing import List

import numpy as np
import pytest

from pento import classification, normalization


def _build_board() -> np.ndarray:
    """Return a board with 12 labelled pentomino regions."""

    return np.array(
        [
            list("ZZZZZYYYYY"),
            list("XXXXXWWWWW"),
            list("VVVVVUUUUU"),
            list("TTTTTNNNNN"),
            list("PPPPPLLLLL"),
            list("IIIIIFFFFF"),
        ]
    )


def test_label_pieces_extracts_all_masks() -> None:
    board = _build_board()
    pieces = classification.label_pieces(board)

    assert {piece.name for piece in pieces} == set(classification.PIECE_NAMES)
    for piece in pieces:
        assert piece.mask.shape == (classification.GRID_HEIGHT, classification.GRID_WIDTH)
        assert piece.mask.dtype == bool
        assert int(np.sum(piece.mask)) == 5

    combined = sum((piece.mask.astype(int) for piece in pieces))
    assert np.all(combined == 1)


def test_label_pieces_rejects_unknown_labels() -> None:
    board = _build_board()
    board[0, 0] = "A"

    with pytest.raises(classification.ClassificationError):
        classification.label_pieces(board)


def test_to_canonical_solution_picks_minimal_orientation() -> None:
    board = _build_board()
    pieces = classification.label_pieces(board)

    # Shuffle to ensure ordering does not affect the outcome.
    shuffled: List[classification.LabeledPiece] = pieces.copy()
    random.Random(1234).shuffle(shuffled)

    canonical = normalization.to_canonical_solution(shuffled)
    expected = np.rot90(board, 2)

    assert canonical.grid.shape == (classification.GRID_HEIGHT, classification.GRID_WIDTH)
    assert np.array_equal(canonical.grid, expected)
