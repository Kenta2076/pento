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

    ordering_keys = [
        (
            len(alt.pieces),
            alt.pieces,
            tuple("".join(row) for row in alt.alternative_grid),
        )
        for alt in results
    ]
    assert ordering_keys == sorted(ordering_keys)

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


def test_find_local_rearrangements_avoids_duplicate_grids() -> None:
    grid = _sample_solution()

    results = rearrangement.find_local_rearrangements(
        grid, max_subset_size=4, max_results=None
    )

    assert results

    unique_grids = {tuple("".join(row) for row in alt.alternative_grid) for alt in results}
    assert len(unique_grids) == len(results)
