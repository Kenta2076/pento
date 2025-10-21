"""Persistence layer for pentomino solutions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

from .normalization import CanonicalSolution


@dataclass
class PiecePlacement:
    """Placeholder structure that mirrors expected downstream data."""

    name: str
    cells: List[tuple[int, int]]


class SolutionArchive:
    """Abstract base class for persisting solutions."""

    def store_solution(self, solution: CanonicalSolution) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class JsonSolutionArchive(SolutionArchive):
    """Store canonical solutions in a JSON Lines file."""

    def __init__(self, path: str | Path = "solutions.jsonl") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def store_solution(self, solution: CanonicalSolution) -> None:
        """Append the solution to the archive file."""

        record = {
            "grid": [[str(cell) for cell in row] for row in solution.grid.tolist()],
        }
        with self.path.open("a", encoding="utf-8") as fh:
            json.dump(record, fh, ensure_ascii=False)
            fh.write("\n")
