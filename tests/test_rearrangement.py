from __future__ import annotations

import numpy as np

from pento import rearrangement


def _sample_solution() -> np.ndarray:
    rows = [
        "PPIIIIIVVV",
        "PPYYYYTTTV",
        "PXFFYZZTNV",
        "XXXFFWZTNN",
        "UXUFWWZZLN",
        "UUUWWLLLLN",
    ]
    return np.array([[cell for cell in row] for row in rows], dtype="<U1")


def test_find_local_rearrangements_discovers_alternative() -> None:
    grid = _sample_solution()

    results = rearrangement.find_local_rearrangements(grid)

    assert results, "Expected at least one alternative rearrangement"

    alt = results[0]
    assert alt.pieces == ("F", "X")
    assert not np.array_equal(alt.alternative_grid, grid)

    region_cells = {(r, c) for r, c in zip(*np.where(np.isin(grid, alt.pieces)))}
    for name, cells in alt.placements.items():
        assert name in alt.pieces
        assert set(cells).issubset(region_cells)

    mask = np.ones_like(grid, dtype=bool)
    for row, col in region_cells:
        mask[row, col] = False
    assert np.array_equal(alt.alternative_grid[mask], grid[mask])


def test_find_local_rearrangements_respects_minimum_subset_size() -> None:
    grid = _sample_solution()

    results = rearrangement.find_local_rearrangements(grid, min_subset_size=3)

    assert results
    assert all(len(alt.pieces) >= 3 for alt in results)


def test_find_local_rearrangements_deduplicates_alternative_grids(monkeypatch) -> None:
    grid = _sample_solution()

    original_candidate_options = rearrangement._candidate_options

    def duplicate_first(*args, **kwargs):
        iterator = iter(original_candidate_options(*args, **kwargs))
        subset, options = next(iterator)
        yield subset, options
        duplicate_options = {name: list(placements) for name, placements in options.items()}
        yield subset, duplicate_options

    monkeypatch.setattr(rearrangement, "_candidate_options", duplicate_first)

    results = rearrangement.find_local_rearrangements(grid, max_results=2)

    signatures = {alt.alternative_grid.tobytes() for alt in results}

    assert len(signatures) == len(results)
