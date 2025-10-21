"""Command line interface for the pentomino archiver template."""

from __future__ import annotations

import argparse

from .pipeline import PentominoArchiver
from .storage import JsonSolutionArchive


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", help="Paths to pentomino board images")
    parser.add_argument(
        "--archive",
        default="solutions.jsonl",
        help="Path to the JSON Lines file where solutions will be appended",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    archive = JsonSolutionArchive(args.archive)
    archiver = PentominoArchiver(archive=archive)
    for image_path in args.images:
        result = archiver.process(image_path)
        print(f"Processed {image_path} -> canonical solution stored at {args.archive}")
        print(result.canonical_solution.grid)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
