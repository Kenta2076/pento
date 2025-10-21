# Pento Archiver Template

This repository provides a Python template for building a photo-based
pentomino solution archiver.  The goal is to take a picture of a completed
6×10 pentomino board, recognize the placement of the 12 pentomino pieces,
normalize the solution, and append it to an archive for later retrieval.

## Project layout

```
src/pento/
├── __init__.py            # Package entry point
├── cli.py                 # Minimal command line interface
├── classification.py      # Pentomino piece identification stubs
├── normalization.py       # Solution normalization placeholder
├── pipeline.py            # High-level orchestration logic
├── preprocessing.py       # Image loading and board extraction helpers
├── segmentation.py        # Grid segmentation scaffolding
└── storage.py             # JSON Lines archive implementation
```

Each module contains a documented stub implementation that outlines the
expected behavior of the corresponding stage in the pipeline.  Developers
can progressively replace these placeholders with production-grade logic
while preserving the overall structure.

## Usage

Install the package in editable mode and invoke the CLI on one or more
images:

```
python -m pip install -e .
python -m pento.cli path/to/board.jpg
```

The current template does not perform real image processing; instead it
illustrates the data flow required to load an image, segment the board,
label the pieces, convert the result into a canonical representation, and
store it in a JSON Lines archive.
