"""Visualization helpers for inspecting the pentomino pipeline in notebooks."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np

from .classification import LabeledPiece, PIECE_NAMES
from .segmentation import GRID_HEIGHT, GRID_WIDTH


def _ensure_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Matplotlib is required for visualization helpers. Install it with `pip install matplotlib`."
        ) from exc

    return plt


def _prepare_image(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[2] == 3:
        # Heuristic: assume OpenCV style BGR ordering when max value <= 1.0.
        if np.issubdtype(array.dtype, np.floating) and array.max() <= 1.0 + 1e-6:
            return np.clip(array[..., ::-1], 0.0, 1.0)
    return array


def show_board_image(board_image: np.ndarray, *, title: str | None = None, ax=None):
    """Display the extracted board image using Matplotlib."""

    plt = _ensure_matplotlib()
    prepared = _prepare_image(board_image)

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    ax.imshow(prepared)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title or "Extracted board")
    return ax


def plot_grid_cells(grid_cells: np.ndarray, *, figsize: Tuple[int, int] = (12, 7)):
    """Plot each segmented cell in a grid of subplots."""

    plt = _ensure_matplotlib()
    grid = np.asarray(grid_cells)

    if grid.ndim < 4:
        raise ValueError("Expected grid_cells to have shape (rows, cols, h, w, c)")

    rows, cols = grid.shape[:2]
    fig, axes = plt.subplots(rows, cols, figsize=figsize, constrained_layout=True)

    for row in range(rows):
        for col in range(cols):
            ax = axes[row][col]
            ax.imshow(_prepare_image(grid[row, col]))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{row},{col}", fontsize=8)

    fig.suptitle("Segmented grid cells", fontsize=14)
    return fig, axes


def labeled_pieces_to_grid(pieces: Iterable[LabeledPiece]) -> np.ndarray:
    """Convert ``LabeledPiece`` objects into a 6×10 label grid."""

    grid = np.full((GRID_HEIGHT, GRID_WIDTH), fill_value=".", dtype="<U1")
    for piece in pieces:
        mask = np.asarray(piece.mask, dtype=bool)
        grid[mask] = piece.name
    return grid


def grid_to_labeled_pieces(label_grid: np.ndarray) -> list[LabeledPiece]:
    """Convert a label grid into ``LabeledPiece`` instances."""

    grid = np.asarray(label_grid)
    if grid.shape != (GRID_HEIGHT, GRID_WIDTH):
        raise ValueError(
            f"Expected grid shape {(GRID_HEIGHT, GRID_WIDTH)}, received {grid.shape}"
        )

    pieces: list[LabeledPiece] = []
    for label in np.unique(grid):
        if str(label) == ".":
            continue
        mask = grid == label
        pieces.append(LabeledPiece(name=str(label), mask=mask))
    return pieces


def format_label_grid(label_grid: np.ndarray) -> str:
    """Return a monospace-friendly string representation of ``label_grid``."""

    grid = np.asarray(label_grid)
    if grid.shape != (GRID_HEIGHT, GRID_WIDTH):
        raise ValueError(
            f"Expected grid shape {(GRID_HEIGHT, GRID_WIDTH)}, received {grid.shape}"
        )

    lines = [" ".join(str(cell) for cell in row) for row in grid]
    return "\n".join(lines)


def plot_label_grid(label_grid: np.ndarray, *, ax=None, title: str | None = None):
    """Visualize the label grid as an annotated heatmap."""

    plt = _ensure_matplotlib()
    grid = np.asarray(label_grid)

    if grid.shape != (GRID_HEIGHT, GRID_WIDTH):
        raise ValueError(
            f"Expected grid shape {(GRID_HEIGHT, GRID_WIDTH)}, received {grid.shape}"
        )

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    ax.imshow(np.zeros_like(grid, dtype=float), cmap="Greys", alpha=0.0)
    for (row, col), value in np.ndenumerate(grid):
        ax.text(col, row, str(value), ha="center", va="center", fontsize=12, family="monospace")

    ax.set_xticks(range(GRID_WIDTH))
    ax.set_yticks(range(GRID_HEIGHT))
    ax.set_xlim(-0.5, GRID_WIDTH - 0.5)
    ax.set_ylim(GRID_HEIGHT - 0.5, -0.5)
    ax.grid(True, which="both", color="lightgray", linewidth=0.5)
    ax.set_title(title or "Pentomino labels")
    return ax


def plot_colored_label_grid(
    label_grid: np.ndarray,
    *,
    ax=None,
    title: str | None = None,
    highlight: Iterable[str] | None = None,
    show_labels: bool = True,
):
    """Render a label grid with per-piece colors for visual exploration."""

    plt = _ensure_matplotlib()
    from matplotlib.colors import to_rgba  # Imported lazily with Matplotlib.

    grid = np.asarray(label_grid)
    if grid.shape != (GRID_HEIGHT, GRID_WIDTH):
        raise ValueError(
            f"Expected grid shape {(GRID_HEIGHT, GRID_WIDTH)}, received {grid.shape}"
        )

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    palette = plt.get_cmap("tab20").colors
    color_lookup = {
        name: to_rgba(palette[index % len(palette)])
        for index, name in enumerate(sorted(PIECE_NAMES))
    }

    highlight_set = {str(name) for name in highlight or []}

    rgba_grid = np.zeros((GRID_HEIGHT, GRID_WIDTH, 4), dtype=float)
    for (row, col), value in np.ndenumerate(grid):
        label = str(value)
        if label == ".":
            rgba_grid[row, col] = (0.95, 0.95, 0.95, 1.0)
            continue

        base = color_lookup.get(label)
        if base is None:
            base = (0.7, 0.7, 0.7, 1.0)

        if highlight_set and label not in highlight_set:
            softened = tuple(0.75 + 0.25 * channel for channel in base[:3]) + (1.0,)
            rgba_grid[row, col] = softened
        else:
            rgba_grid[row, col] = base[:3] + (1.0,)

    ax.imshow(rgba_grid, interpolation="none")

    ax.set_xticks(range(GRID_WIDTH))
    ax.set_yticks(range(GRID_HEIGHT))
    ax.set_xlim(-0.5, GRID_WIDTH - 0.5)
    ax.set_ylim(GRID_HEIGHT - 0.5, -0.5)
    ax.grid(True, which="both", color="white", linewidth=1.0, alpha=0.6)

    if show_labels:
        for (row, col), value in np.ndenumerate(grid):
            ax.text(
                col,
                row,
                str(value),
                ha="center",
                va="center",
                fontsize=12,
                family="monospace",
                color="black",
            )

    if highlight_set:
        ax.set_title(title or f"Highlighted: {', '.join(sorted(highlight_set))}")
    else:
        ax.set_title(title or "Pentomino labels")

    return ax


__all__ = [
    "format_label_grid",
    "grid_to_labeled_pieces",
    "labeled_pieces_to_grid",
    "plot_colored_label_grid",
    "plot_grid_cells",
    "plot_label_grid",
    "show_board_image",
]
