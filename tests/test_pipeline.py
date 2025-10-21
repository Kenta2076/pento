from __future__ import annotations

from pathlib import Path

import numpy as np

from pento import classification, normalization, pipeline, preprocessing, segmentation
from pento.storage import SolutionArchive


class InMemoryArchive(SolutionArchive):
    def __init__(self) -> None:
        self.stored: list[normalization.CanonicalSolution] = []

    def store_solution(self, solution: normalization.CanonicalSolution) -> None:  # pragma: no cover - invoked in tests
        self.stored.append(solution)


def _make_piece(name: str, index: int) -> classification.LabeledPiece:
    mask = np.zeros((segmentation.GRID_HEIGHT, segmentation.GRID_WIDTH), dtype=bool)
    row = index % segmentation.GRID_HEIGHT
    col_block = index // segmentation.GRID_HEIGHT
    start = col_block * 5
    mask[row, start : start + 5] = True
    return classification.LabeledPiece(name=name, mask=mask)


def test_pentomino_archiver_process(monkeypatch, tmp_path) -> None:
    image_path = tmp_path / "board.png"
    image_path.write_bytes(b"")

    board_image = np.zeros((10, 10, 3), dtype=np.uint8)
    grid_cells = np.zeros((segmentation.GRID_HEIGHT, segmentation.GRID_WIDTH), dtype=object)

    pieces = [_make_piece(name, idx) for idx, name in enumerate(classification.PIECE_NAMES)]

    expected_grid = np.full((segmentation.GRID_HEIGHT, segmentation.GRID_WIDTH), fill_value="F", dtype="<U1")
    canonical = normalization.CanonicalSolution(grid=expected_grid)

    monkeypatch.setattr(preprocessing, "load_image", lambda path: board_image)
    monkeypatch.setattr(preprocessing, "extract_board_region", lambda image: image)
    monkeypatch.setattr(segmentation, "segment_grid", lambda board: grid_cells)
    monkeypatch.setattr(classification, "label_pieces", lambda grid: pieces)
    monkeypatch.setattr(normalization, "to_canonical_solution", lambda pcs: canonical)

    archive = InMemoryArchive()
    archiver = pipeline.PentominoArchiver(archive=archive)

    result = archiver.process(image_path)

    assert isinstance(result, pipeline.PipelineResult)
    assert result.image_path == Path(image_path)
    assert np.array_equal(result.board_image, board_image)
    assert np.array_equal(result.grid_cells, grid_cells)
    assert result.labeled_pieces == pieces
    assert result.canonical_solution is canonical
    assert archive.stored == [canonical]
