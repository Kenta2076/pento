"""Piece classification utilities for pentomino recognition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .segmentation import GRID_HEIGHT, GRID_WIDTH

PIECE_NAMES = [
    "F",
    "I",
    "L",
    "P",
    "N",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
]


@dataclass
class LabeledPiece:
    """Information about a recognized pentomino piece."""

    name: str
    mask: np.ndarray


class ClassificationError(RuntimeError):
    """Raised when piece classification fails."""


def label_pieces(grid_cells: np.ndarray) -> List[LabeledPiece]:
    """Assign pentomino labels to the segmented grid cells.

    The template implementation returns an empty list but performs basic
    validation on the grid structure.  Downstream components expect the
    result to contain exactly 12 entries, one for each pentomino type.
    """

    if grid_cells.shape[0:2] != (GRID_HEIGHT, GRID_WIDTH):
        raise ClassificationError(
            f"Grid must have shape ({GRID_HEIGHT}, {GRID_WIDTH}, ...), got {grid_cells.shape}"
        )

    return []
