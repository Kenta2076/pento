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
    assert all(len(alt.pieces) >= 2 for alt in results)

    traced = rearrangement.trace_local_rearrangements(
        grid, min_subset_size=3, max_results=len(results)
    )

    assert traced

    result_signatures = {alt.alternative_grid.tobytes() for alt in results}

    traced_signatures = set()
    for rearrangement_result, steps in traced:
        traced_signatures.add(rearrangement_result.alternative_grid.tobytes())
        assert all(len(step.subset) >= 3 for step in steps)

    assert result_signatures.issubset(traced_signatures)


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


def test_find_local_rearrangements_ignore_unchanged_pieces(monkeypatch) -> None:
    grid = _sample_solution()

    target_subset = ("F", "I", "L", "N", "X")

    placements = rearrangement._validate_solution_grid(grid)

    original_candidate_options = rearrangement._candidate_options
    original_search_alternative = rearrangement._search_alternative

    fx_options = None
    for subset, options in original_candidate_options(placements, min_size=2, max_size=2):
        if subset == ("F", "X"):
            fx_options = options
            break

    assert fx_options is not None

    fx_alternative = original_search_alternative(("F", "X"), fx_options, placements)
    assert fx_alternative is not None

    superset_alternative = {name: placements[name] for name in target_subset}
    superset_alternative.update(fx_alternative)

    def yield_superset_first(placements_arg, *, min_size, max_size):
        if min_size <= len(target_subset) <= max_size:
            yield target_subset, {}

    def fake_search_alternative(subset, options, original, *, record_step=None):
        if tuple(sorted(subset)) == target_subset:
            return superset_alternative
        return original_search_alternative(
            subset, options, original, record_step=record_step
        )

    monkeypatch.setattr(rearrangement, "_candidate_options", yield_superset_first)
    monkeypatch.setattr(rearrangement, "_search_alternative", fake_search_alternative)

    result = rearrangement.find_local_rearrangements(
        grid, max_subset_size=5, max_results=1
    )[0]

    assert set(result.pieces) == set(result.placements)
    for name in result.pieces:
        assert result.placements[name] != placements[name]
