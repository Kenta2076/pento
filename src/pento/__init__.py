"""Pento module for archiving pentomino puzzle solutions.

This package provides a pipeline that takes a photo of a completed
pentomino puzzle, extracts the board state, converts it into a canonical
representation, and stores it in a local archive.
"""

from .pipeline import PentominoArchiver

__all__ = ["PentominoArchiver"]
