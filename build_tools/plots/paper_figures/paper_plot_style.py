from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

COLORS = {
    "merlin": "#0072B2",
    "executorch": "#D55E00",
    "xnnpack": "#009E73",
    "openblas": "#CC79A7",
    "pending": "#777777",
}

mpl.rcParams.update(
    {
        "font.size": 10,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def save_figure(fig: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="pdf")
    print(f"saved {output}")


def figure_dir(set_name: str) -> Path:
    """Where a figure set renders: ``out/artifacts/paper-figures/<set_name>/``, created on demand.

    Renders are generated output, so they belong under the single out/ root -- never beside the
    generator, which is how 33 renders once accumulated in a top-level ``figures/`` directory.
    """
    from merlin.common.paths import artifacts_dir  # noqa: PLC0415

    out = artifacts_dir() / "paper-figures" / set_name
    out.mkdir(parents=True, exist_ok=True)
    return out
