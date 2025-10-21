"""High-level pipeline for recognizing and archiving pentomino solutions."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np

from . import classification, normalization, preprocessing, segmentation, storage


logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Container for the intermediate artifacts produced by the pipeline."""

    image_path: Path
    board_image: np.ndarray
    grid_cells: np.ndarray
    labeled_pieces: List[classification.LabeledPiece]
    canonical_solution: normalization.CanonicalSolution


class PentominoArchiver:
    """Entry point that runs the complete archiving pipeline.

    The pipeline is composed of the following stages:

    1. Image preprocessing (deskewing, perspective correction, color balancing).
    2. Grid segmentation into 6x10 cells.
    3. Piece classification for each connected component.
    4. Solution normalization to obtain a canonical representation.
    5. Persistence in the solution archive.

    Each stage is implemented in a dedicated module so that the
    implementation can evolve independently while the high-level flow
    remains stable.
    """

    def __init__(self, archive: storage.SolutionArchive | None = None) -> None:
        self.archive = archive or storage.JsonSolutionArchive()

    def process(self, image_path: str | Path) -> PipelineResult:
        """Run the pipeline on the provided image.

        Parameters
        ----------
        image_path:
            Path to an image that contains a completed 6x10 pentomino board.
        """

        image_path = Path(image_path)
        logger.info("Starting pipeline for %s", image_path)
        original = preprocessing.load_image(image_path)
        board = preprocessing.extract_board_region(original)
        grid = segmentation.segment_grid(board)
        pieces = classification.label_pieces(grid)
        canonical = normalization.to_canonical_solution(pieces)
        self.archive.store_solution(canonical)
        logger.info("Completed pipeline for %s", image_path)
        return PipelineResult(
            image_path=image_path,
            board_image=board,
            grid_cells=grid,
            labeled_pieces=pieces,
            canonical_solution=canonical,
        )

    def batch_process(self, image_paths: Iterable[str | Path]) -> List[PipelineResult]:
        """Process a collection of images and return the pipeline results."""

        results: List[PipelineResult] = []
        for path in image_paths:
            results.append(self.process(path))
        return results
