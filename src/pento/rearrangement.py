"""Utilities for exploring local rearrangements of pentomino solutions."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy.typing as npt

import numpy as np

from .classification import PIECE_NAMES
from .segmentation import GRID_HEIGHT, GRID_WIDTH


_PIECE_PATTERNS: Mapping[str, Sequence[str]] = {
    "F": (".XX", "XX.", ".X."),
    "I": ("X", "X", "X", "X", "X"),
    "L": ("X.", "X.", "X.", "XX"),
    "P": ("XX", "XX", "X."),
    "N": ("X.", "XX", ".X", ".X"),
    "T": ("XXX", ".X.", ".X."),
    "U": ("X.X", "XXX"),
    "V": ("X..", "X..", "XXX"),
    "W": ("X..", "XX.", ".XX"),
    "X": (".X.", "XXX", ".X."),
    "Y": ("X.", "XX", "X.", "X."),
    "Z": ("XX.", ".X.", ".XX"),
}


def _parse_pattern(pattern: Sequence[str]) -> List[Tuple[int, int]]:
    cells: List[Tuple[int, int]] = []
    for row, line in enumerate(pattern):
        for col, char in enumerate(line):
            if char == "X":
                cells.append((row, col))
    return cells


def _rotate(cells: Iterable[Tuple[int, int]]) -> List[Tuple[int, int]]:
    return [(-col, row) for row, col in cells]


def _flip(cells: Iterable[Tuple[int, int]]) -> List[Tuple[int, int]]:
    return [(row, -col) for row, col in cells]


def _normalize(cells: Iterable[Tuple[int, int]]) -> Tuple[Tuple[int, int], ...]:
    as_list = list(cells)
    min_row = min(row for row, _ in as_list)
    min_col = min(col for _, col in as_list)
    normalized = sorted((row - min_row, col - min_col) for row, col in as_list)
    return tuple(normalized)


def _generate_orientations(pattern: Sequence[str]) -> List[Tuple[Tuple[int, int], ...]]:
    base = _parse_pattern(pattern)
    seen: set[Tuple[Tuple[int, int], ...]] = set()
    orientations: List[Tuple[Tuple[int, int], ...]] = []

    current = base
    rotations: List[List[Tuple[int, int]]] = [current]
    for _ in range(3):
        current = _rotate(current)
        rotations.append(current)

    for rotated in rotations:
        for variant in (rotated, _flip(rotated)):
            normalized = _normalize(variant)
            if normalized not in seen:
                seen.add(normalized)
                orientations.append(normalized)

    return orientations


Placement = frozenset[Tuple[int, int]]


def _placements_for_piece(pattern: Sequence[str]) -> List[Placement]:
    placements: set[Placement] = set()

    for orientation in _generate_orientations(pattern):
        max_row = max(row for row, _ in orientation)
        max_col = max(col for _, col in orientation)

        for base_row in range(GRID_HEIGHT - max_row):
            for base_col in range(GRID_WIDTH - max_col):
                placed = frozenset((row + base_row, col + base_col) for row, col in orientation)
                placements.add(placed)

    return sorted(placements, key=lambda cells: sorted(cells))


_ALL_PLACEMENTS: Dict[str, List[frozenset[Tuple[int, int]]]] = {
    name: _placements_for_piece(pattern)
    for name, pattern in _PIECE_PATTERNS.items()
}


@dataclass(frozen=True)
class LocalRearrangement:
    """Description of an alternative placement for a subset of pieces."""

    pieces: Tuple[str, ...]
    alternative_grid: npt.NDArray[np.str_]
    placements: Dict[str, Placement]


def _validate_solution_grid(grid: npt.NDArray[np.str_]) -> Dict[str, Placement]:
    if grid.shape != (GRID_HEIGHT, GRID_WIDTH):
        raise ValueError(
            f"Solution grid must have shape {(GRID_HEIGHT, GRID_WIDTH)}, got {grid.shape}"
        )

    cells_by_piece: Dict[str, set[Tuple[int, int]]] = {name: set() for name in PIECE_NAMES}

    for row in range(GRID_HEIGHT):
        for col in range(GRID_WIDTH):
            label = str(grid[row, col])
            if label not in cells_by_piece:
                raise ValueError(f"Unexpected label '{label}' at ({row}, {col}) in solution grid")
            cells_by_piece[label].add((row, col))

    placements: Dict[str, Placement] = {}

    for name, cells in cells_by_piece.items():
        if len(cells) != 5:
            raise ValueError(f"Piece '{name}' covers {len(cells)} cells (expected 5)")

        placement = frozenset(cells)
        if placement not in _ALL_PLACEMENTS[name]:
            raise ValueError(f"Placement for piece '{name}' does not match a valid pentomino shape")

        placements[name] = placement

    return placements


def _search_alternative(
    subset: Sequence[str],
    options: Mapping[str, List[Placement]],
    original: Mapping[str, Placement],
) -> Dict[str, Placement] | None:
    names = tuple(sorted(subset, key=lambda name: len(options[name])))
    used: set[Tuple[int, int]] = set()
    assignment: Dict[str, Placement] = {}

    def backtrack(index: int) -> Dict[str, frozenset[Tuple[int, int]]] | None:
        if index == len(names):
            if any(assignment[name] != original[name] for name in names):
                return dict(assignment)
            return None

        name = names[index]
        for placement in options[name]:
            if placement == original[name]:
                continue

            if any(cell in used for cell in placement):
                continue

            assignment[name] = placement
            used.update(placement)

            result = backtrack(index + 1)
            if result is not None:
                return result

            used.difference_update(placement)

        return None

    return backtrack(0)


def _canonicalize_grid(grid: npt.NDArray[np.str_]) -> Tuple[str, ...]:
    return tuple("".join(row) for row in grid)


def find_local_rearrangements(
    solution_grid: npt.NDArray[np.str_],
    *,
    min_subset_size: int = 2,
    max_subset_size: int | None = None,
    max_results: int | None = 1,
) -> List[LocalRearrangement]:
    """Search for alternative placements by rearranging subsets of pieces.

    Args:
        solution_grid: 6×10 array containing the piece labels of a known solution.
        min_subset_size: Minimum number of pieces that may be rearranged together.
        max_subset_size: Optional upper bound on the number of pieces to rearrange.
        max_results: Optional limit on the number of rearrangements to return. If
            ``None``, all rearrangements discovered within the search limits are
            returned.

    Returns:
        A list of :class:`LocalRearrangement` instances describing the discovered
        alternative placements.  The list is sorted by increasing subset size and
        lexicographic order of the involved piece names.
    """

    grid = np.asarray(solution_grid, dtype="<U1")
    placements = _validate_solution_grid(grid)

    min_size = max(2, min_subset_size)
    max_size = len(PIECE_NAMES) if max_subset_size is None else max(min_size, max_subset_size)

    results: List[LocalRearrangement] = []
    seen: set[Tuple[str, ...]] = set()

    def result_key(alt: LocalRearrangement) -> Tuple[int, Tuple[str, ...], Tuple[str, ...]]:
        return (len(alt.pieces), alt.pieces, _canonicalize_grid(alt.alternative_grid))

    for size in range(min_size, max_size + 1):
        for subset in combinations(sorted(PIECE_NAMES), size):
            region_cells = frozenset().union(*(placements[name] for name in subset))

            candidate_options: Dict[str, List[Placement]] = {}
            viable = True

            for name in subset:
                options = [
                    placement
                    for placement in _ALL_PLACEMENTS[name]
                    if placement.issubset(region_cells)
                ]

                if not any(placement != placements[name] for placement in options):
                    viable = False
                    break

                candidate_options[name] = options

            if not viable:
                continue

            alternative = _search_alternative(subset, candidate_options, placements)
            if alternative is None:
                continue

            new_grid = np.array(grid, copy=True)
            for name in subset:
                for cell in placements[name]:
                    new_grid[cell] = "."
            for name, cells in alternative.items():
                for row, col in cells:
                    new_grid[row, col] = name

            rearrangement = LocalRearrangement(
                pieces=tuple(sorted(subset)),
                alternative_grid=new_grid,
                placements=alternative,
            )

            key = _canonicalize_grid(new_grid)

            if key in seen:
                continue

            seen.add(key)
            results.append(rearrangement)

            if max_results is not None and len(results) >= max_results:
                return sorted(results, key=result_key)

    return sorted(results, key=result_key)
