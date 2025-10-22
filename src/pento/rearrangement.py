"""Utilities for exploring local rearrangements of pentomino solutions."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Dict, Iterable, Iterator, List, Mapping, Sequence, Tuple

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


def _placements_for_piece(pattern: Sequence[str]) -> List[frozenset[Tuple[int, int]]]:
    placements: set[frozenset[Tuple[int, int]]] = set()

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
class SearchStep:
    """Snapshot emitted while exploring a local rearrangement search."""

    subset: Tuple[str, ...]
    depth: int
    piece: str | None
    placement: frozenset[Tuple[int, int]] | None
    status: str


@dataclass
class LocalRearrangement:
    """Description of an alternative placement for a subset of pieces."""

    pieces: Tuple[str, ...]
    alternative_grid: np.ndarray
    placements: Dict[str, frozenset[Tuple[int, int]]]


def _validate_solution_grid(grid: np.ndarray) -> Dict[str, frozenset[Tuple[int, int]]]:
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

    placements: Dict[str, frozenset[Tuple[int, int]]] = {}

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
    options: Mapping[str, List[frozenset[Tuple[int, int]]]],
    original: Mapping[str, frozenset[Tuple[int, int]]],
    *,
    record_step: Callable[[SearchStep], None] | None = None,
) -> Dict[str, frozenset[Tuple[int, int]]] | None:
    names = tuple(sorted(subset))
    used: set[Tuple[int, int]] = set()
    assignment: Dict[str, frozenset[Tuple[int, int]]] = {}

    def emit(step: SearchStep) -> None:
        if record_step is not None:
            record_step(step)

    def backtrack(index: int) -> Dict[str, frozenset[Tuple[int, int]]] | None:
        if index == len(names):
            if any(assignment[name] != original[name] for name in names):
                emit(
                    SearchStep(
                        subset=names,
                        depth=index,
                        piece=None,
                        placement=None,
                        status="complete",
                    )
                )
                return dict(assignment)
            return None

        name = names[index]
        for placement in options[name]:
            if any(cell in used for cell in placement):
                emit(
                    SearchStep(
                        subset=names,
                        depth=index,
                        piece=name,
                        placement=placement,
                        status="conflict",
                    )
                )
                continue

            emit(
                SearchStep(
                    subset=names,
                    depth=index,
                    piece=name,
                    placement=placement,
                    status="try",
                )
            )

            assignment[name] = placement
            used.update(placement)

            result = backtrack(index + 1)
            if result is not None:
                emit(
                    SearchStep(
                        subset=names,
                        depth=index,
                        piece=name,
                        placement=placement,
                        status="accept",
                    )
                )
                return result

            used.difference_update(placement)
            emit(
                SearchStep(
                    subset=names,
                    depth=index,
                    piece=name,
                    placement=placement,
                    status="backtrack",
                )
            )

        return None

    return backtrack(0)


def _candidate_options(
    placements: Mapping[str, frozenset[Tuple[int, int]]],
    *,
    min_size: int,
    max_size: int,
) -> Iterator[Tuple[Tuple[str, ...], Dict[str, List[frozenset[Tuple[int, int]]]]]]:
    all_names = tuple(sorted(PIECE_NAMES))

    for size in range(min_size, max_size + 1):
        for subset in combinations(all_names, size):
            region_cells = frozenset().union(*(placements[name] for name in subset))

            candidate_options: Dict[str, List[frozenset[Tuple[int, int]]]] = {}
            viable = True

            for name in subset:
                options = [
                    placement
                    for placement in _ALL_PLACEMENTS[name]
                    if placement.issubset(region_cells)
                ]

                if len(options) <= 1:
                    viable = False
                    break

                candidate_options[name] = options

            if viable:
                yield subset, candidate_options


def _build_rearranged_grid(
    grid: np.ndarray,
    *,
    subset: Sequence[str],
    placements: Mapping[str, frozenset[Tuple[int, int]]],
    alternative: Mapping[str, frozenset[Tuple[int, int]]],
) -> np.ndarray:
    new_grid = np.array(grid, copy=True)
    for name in subset:
        for cell in placements[name]:
            new_grid[cell] = "."
    for name, cells in alternative.items():
        for row, col in cells:
            new_grid[row, col] = name
    return new_grid


def find_local_rearrangements(
    solution_grid: np.ndarray,
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
        max_results: Optional limit on the number of rearrangements to return.

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
    seen_grids: set[bytes] = set()

    for subset, candidate_options in _candidate_options(
        placements, min_size=min_size, max_size=max_size
    ):
        alternative = _search_alternative(subset, candidate_options, placements)
        if alternative is None:
            continue

        new_grid = _build_rearranged_grid(
            grid, subset=subset, placements=placements, alternative=alternative
        )
        grid_signature = new_grid.tobytes()
        if grid_signature in seen_grids:
            continue

        seen_grids.add(grid_signature)

        rearrangement = LocalRearrangement(
            pieces=tuple(sorted(subset)),
            alternative_grid=new_grid,
            placements=alternative,
        )
        results.append(rearrangement)

        if max_results is not None and len(results) >= max_results:
            return results

    return results


def trace_local_rearrangements(
    solution_grid: np.ndarray,
    *,
    min_subset_size: int = 2,
    max_subset_size: int | None = None,
    max_results: int | None = 1,
) -> List[Tuple[LocalRearrangement, List[SearchStep]]]:
    """Run the rearrangement search while recording the explored placements."""

    grid = np.asarray(solution_grid, dtype="<U1")
    placements = _validate_solution_grid(grid)

    min_size = max(2, min_subset_size)
    max_size = len(PIECE_NAMES) if max_subset_size is None else max(min_size, max_subset_size)

    traced_results: List[Tuple[LocalRearrangement, List[SearchStep]]] = []
    seen_grids: set[bytes] = set()

    for subset, candidate_options in _candidate_options(
        placements, min_size=min_size, max_size=max_size
    ):
        steps: List[SearchStep] = []
        alternative = _search_alternative(
            subset,
            candidate_options,
            placements,
            record_step=steps.append,
        )

        if alternative is None:
            continue

        new_grid = _build_rearranged_grid(
            grid, subset=subset, placements=placements, alternative=alternative
        )
        grid_signature = new_grid.tobytes()
        if grid_signature in seen_grids:
            continue

        seen_grids.add(grid_signature)

        rearrangement = LocalRearrangement(
            pieces=tuple(sorted(subset)),
            alternative_grid=new_grid,
            placements=alternative,
        )
        traced_results.append((rearrangement, steps))

        if max_results is not None and len(traced_results) >= max_results:
            break

    return traced_results
