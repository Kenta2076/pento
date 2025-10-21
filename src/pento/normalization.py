"""Utilities for normalizing pentomino solutions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np

from .classification import LabeledPiece
from .segmentation import GRID_HEIGHT, GRID_WIDTH


@dataclass
class CanonicalSolution:
    """Canonical representation of a pentomino solution."""

    grid: np.ndarray


class NormalizationError(RuntimeError):
    """Raised when a solution cannot be normalized."""


def to_canonical_solution(pieces: Iterable[LabeledPiece]) -> CanonicalSolution:
    """Convert labeled pieces into a canonical grid representation.

    The placeholder implementation returns an empty grid.  Replace this
    with logic that constructs a 6×10 grid of piece identifiers and then
    normalizes it by applying rotations/reflections to choose a canonical
    representative.
    """

    grid = np.full((GRID_HEIGHT, GRID_WIDTH), fill_value=".", dtype=object)
    for piece in pieces:
        if piece.mask.shape != grid.shape:
            raise NormalizationError(
                "Template expects piece masks to have board shape; replace with real logic"
            )
        grid = np.where(piece.mask, piece.name, grid)

    return CanonicalSolution(grid=grid)
