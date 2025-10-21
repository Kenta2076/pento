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


def _squeeze_grid(grid_cells: np.ndarray) -> np.ndarray:
    """Reduce ``grid_cells`` to a 2-D array while keeping the leading axes."""

    if grid_cells.ndim < 2:
        raise ClassificationError("Grid must be at least two-dimensional")

    if grid_cells.ndim == 2:
        return grid_cells

    trailing = grid_cells.shape[2:]
    if any(size != 1 for size in trailing):
        raise ClassificationError(
            "Each cell must resolve to a single label; trailing dimensions must be singleton",
        )

    axes = tuple(range(2, grid_cells.ndim))
    return np.squeeze(grid_cells, axis=axes)


def _labels_from_grid(grid: np.ndarray) -> np.ndarray:
    """Convert ``grid`` into a 2-D array of pentomino labels."""

    if grid.shape != (GRID_HEIGHT, GRID_WIDTH):
        raise ClassificationError(
            f"Grid must have shape ({GRID_HEIGHT}, {GRID_WIDTH}), got {grid.shape}"
        )

    if grid.dtype.kind in {"U", "S", "O"}:
        labels = grid.astype(str)
    elif np.issubdtype(grid.dtype, np.integer):
        unique_values = sorted(int(value) for value in np.unique(grid))

        if len(unique_values) != len(PIECE_NAMES):
            raise ClassificationError(
                "Integer labeled grid must contain exactly 12 unique values",
            )

        if unique_values[0] == 0 and unique_values[-1] == len(PIECE_NAMES) - 1:
            mapping = {value: PIECE_NAMES[value] for value in unique_values}
        elif unique_values[0] == 1 and unique_values[-1] == len(PIECE_NAMES):
            mapping = {value: PIECE_NAMES[value - 1] for value in unique_values}
        else:
            raise ClassificationError(
                "Cannot infer pentomino names from integer labels; expected contiguous range",
            )

        labels = np.empty_like(grid, dtype="<U1")
        for value, name in mapping.items():
            labels[grid == value] = name
    else:
        raise ClassificationError("Unsupported grid dtype for classification")

    return labels


def label_pieces(grid_cells: np.ndarray) -> List[LabeledPiece]:
    """Assign pentomino labels and masks to the segmented grid cells."""

    if grid_cells.shape[0:2] != (GRID_HEIGHT, GRID_WIDTH):
        raise ClassificationError(
            f"Grid must have shape ({GRID_HEIGHT}, {GRID_WIDTH}, ...), got {grid_cells.shape}"
        )

    squeezed = _squeeze_grid(np.asarray(grid_cells))
    label_grid = _labels_from_grid(squeezed)

    unique_labels = sorted(label for label in np.unique(label_grid) if label != ".")
    missing = set(PIECE_NAMES) - set(unique_labels)

    if missing:
        raise ClassificationError(
            f"Missing pentomino labels: {', '.join(sorted(missing))}",
        )

    if len(unique_labels) != len(PIECE_NAMES):
        raise ClassificationError("Expected exactly 12 unique pentomino labels")

    pieces: List[LabeledPiece] = []
    coverage = np.zeros_like(label_grid, dtype=int)

    for name in PIECE_NAMES:
        mask = label_grid == name
        if not np.any(mask):
            raise ClassificationError(f"Pentomino '{name}' is not present in the grid")

        cell_count = int(np.sum(mask))
        if cell_count != 5:
            raise ClassificationError(
                f"Pentomino '{name}' covers {cell_count} cells (expected 5)"
            )

        coverage[mask] += 1
        pieces.append(LabeledPiece(name=name, mask=mask.astype(bool)))

    if not np.all(coverage == 1):
        raise ClassificationError("Piece masks overlap or leave gaps in the grid")

    return pieces
