"""Decision boundaries, drawn by classifying a grid of points."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

POSITIVE = "#1a7f37"
NEGATIVE = "#cf222e"


def _boundary(axis, model, x, y, title):
    margin = 1.0
    grid_x, grid_y = np.meshgrid(
        np.linspace(x[:, 0].min() - margin, x[:, 0].max() + margin, 300),
        np.linspace(x[:, 1].min() - margin, x[:, 1].max() + margin, 300),
    )
    grid = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    axis.contourf(grid_x, grid_y, model.predict(grid).reshape(grid_x.shape),
                  levels=[-2, 0, 2], colors=[NEGATIVE, POSITIVE], alpha=0.16)
    for label, colour in ((1, POSITIVE), (-1, NEGATIVE)):
        axis.scatter(x[y == label, 0], x[y == label, 1], s=8, color=colour,
                     label=f"class {label}")
    axis.set_title(f"{title}\naccuracy {model.score(x, y):.0%}", fontsize=11)
    axis.set_xticks([]), axis.set_yticks([])


def boundaries(panels: list[tuple], output_path: Path, suptitle: str = "") -> Path:
    """Each panel is (fitted model, x, y, title)."""
    figure, axes = plt.subplots(1, len(panels), figsize=(4.3 * len(panels), 4.3))
    axes = [axes] if len(panels) == 1 else list(axes)
    for axis, (model, x, y, title) in zip(axes, panels):
        _boundary(axis, model, x, y, title)
    axes[0].legend(loc="upper left", fontsize=8)
    if suptitle:
        figure.suptitle(suptitle, fontsize=12, y=1.03)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(figure)
    return output_path
