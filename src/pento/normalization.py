"""Utilities for normalizing pentomino solutions."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List

import numpy as np

from .classification import LabeledPiece, PIECE_NAMES
from .segmentation import GRID_HEIGHT, GRID_WIDTH


logger = logging.getLogger(__name__)


@dataclass
class CanonicalSolution:
    """Canonical representation of a pentomino solution."""

    grid: np.ndarray


class NormalizationError(RuntimeError):
    """Raised when a solution cannot be normalized."""


def _validate_piece(piece: LabeledPiece) -> np.ndarray:
    """Validate an individual piece and return its boolean mask."""

    if piece.name not in PIECE_NAMES:
        raise NormalizationError(f"Unknown pentomino name: {piece.name}")

    mask = np.asarray(piece.mask, dtype=bool)
    if mask.shape != (GRID_HEIGHT, GRID_WIDTH):
        raise NormalizationError(
            f"Mask for '{piece.name}' must have shape {(GRID_HEIGHT, GRID_WIDTH)}, got {mask.shape}"
        )

    cell_count = int(np.sum(mask))
    if cell_count != 5:
        raise NormalizationError(
            f"Pentomino '{piece.name}' covers {cell_count} cells (expected 5)"
        )

    return mask


def _assemble_grid(pieces: List[LabeledPiece]) -> np.ndarray:
    """Convert the labeled pieces into a 6×10 grid of identifiers."""

    grid = np.full((GRID_HEIGHT, GRID_WIDTH), fill_value=".", dtype="<U1")

    for piece in pieces:
        mask = _validate_piece(piece)
        if np.any(grid[mask] != "."):
            raise NormalizationError(
                f"Pentomino '{piece.name}' overlaps another piece",
            )
        grid[mask] = piece.name

    if np.any(grid == "."):
        raise NormalizationError("Incomplete tiling: some cells are not covered by a piece")

    return grid


def _transforms(grid: np.ndarray) -> List[np.ndarray]:
    """Generate symmetry-equivalent grids (rotations/reflections)."""

    candidates: List[np.ndarray] = []

    for flip_ud in (False, True):
        for flip_lr in (False, True):
            transformed = grid
            if flip_ud:
                transformed = np.flipud(transformed)
            if flip_lr:
                transformed = np.fliplr(transformed)

            candidates.append(np.array(transformed, copy=True))
            candidates.append(np.rot90(transformed, 2))

    unique: List[np.ndarray] = []
    seen = set()

    for candidate in candidates:
        key = tuple(candidate.flatten())
        if key in seen:
            continue
        seen.add(key)
        unique.append(np.array(candidate, copy=True))

    return unique


def _choose_canonical(grid: np.ndarray) -> np.ndarray:
    """Select the lexicographically smallest representative."""

    candidates = _transforms(grid)
    if not candidates:
        raise NormalizationError("No grid candidates generated during normalization")

    best = min(candidates, key=lambda arr: tuple(arr.flatten()))
    return np.array(best, copy=True)


def to_canonical_solution(pieces: Iterable[LabeledPiece]) -> CanonicalSolution:
    """Convert labeled pieces into a canonical grid representation."""

    piece_list = list(pieces)

    if len(piece_list) != len(PIECE_NAMES):
        raise NormalizationError(
            f"Expected {len(PIECE_NAMES)} pentominoes, received {len(piece_list)}",
        )

    seen_names = set()
    for piece in piece_list:
        if piece.name in seen_names:
            raise NormalizationError(f"Duplicate pentomino name: {piece.name}")
        seen_names.add(piece.name)

    if set(seen_names) != set(PIECE_NAMES):
        missing = set(PIECE_NAMES) - set(seen_names)
        raise NormalizationError(
            f"Missing pentominoes in solution: {', '.join(sorted(missing))}"
        )

    grid = _assemble_grid(piece_list)
    canonical_grid = _choose_canonical(grid)
    logger.info("Normalized solution to canonical grid:\n%s", "\n".join("".join(row) for row in canonical_grid))
    return CanonicalSolution(grid=canonical_grid)
