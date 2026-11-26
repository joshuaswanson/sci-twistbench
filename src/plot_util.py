"""Plotting utilities adapted directly from ethz-spylab/modal-aphasia."""

import math
import os

os.environ["PATH"] = "/Library/TeX/texbin:" + os.environ.get("PATH", "")

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors
import matplotlib.figure
import matplotlib.legend
import matplotlib.patches
import matplotlib.pyplot

TEX_PT_PER_IN = 72.27
DEFAULT_PPI = 300.0

GOLDEN_RATIO = 0.5 * (1.0 + math.sqrt(5))

# Reference font sizes from LaTeX (all in pt, rendered 1:1 with zero downscaling)
_FONT_SIZE_MAIN_PT = 10.0
_FONT_SIZE_SMALL_PT = 9.0
_FONT_SIZE_SCRIPTSIZE_PT = 7.0
_FONT_SIZE_IN_BETWEEN = 8.0

# Exact ICLR textwidth physical figure sizes (golden ratio)
FIGURE_WIDTH_FULL_PT = 396.97516
FIGURE_WIDTH_HALF_PT = 193.48758
FIGURE_WIDTH_NINECOL_PT = 295.23137

FIGURE_SIZE_FULL_IN = (
    FIGURE_WIDTH_FULL_PT / TEX_PT_PER_IN,
    (FIGURE_WIDTH_FULL_PT / TEX_PT_PER_IN) / GOLDEN_RATIO,
)
FIGURE_SIZE_HALF_IN = (
    FIGURE_WIDTH_HALF_PT / TEX_PT_PER_IN,
    (FIGURE_WIDTH_HALF_PT / TEX_PT_PER_IN) / GOLDEN_RATIO,
)
FIGURE_SIZE_NINECOL_IN = (
    FIGURE_WIDTH_NINECOL_PT / TEX_PT_PER_IN,
    (FIGURE_WIDTH_NINECOL_PT / TEX_PT_PER_IN) / GOLDEN_RATIO,
)

LINE_WIDTH_PT = 1.25
BAR_WIDTH = 1.0 / GOLDEN_RATIO

DEFAULT_COLORMAP = matplotlib.colors.ListedColormap(
    colors=("#139FCD", "#FFD166", "#CE123E", "#0D6382", "#03D55F", "#D3B6F6"),
    name="cvd_friendly",
)


def setup_matplotlib():
    matplotlib.pyplot.rcdefaults()
    matplotlib.colormaps.register(cmap=DEFAULT_COLORMAP, force=True)

    matplotlib.rcParams.update(
        {
            "text.usetex": True,
            "text.latex.preamble": r"\renewcommand{\familydefault}{\sfdefault}\usepackage{sansmath}\sansmath",
            "image.cmap": DEFAULT_COLORMAP.name,
            "axes.prop_cycle": matplotlib.rcsetup.cycler("color", DEFAULT_COLORMAP.colors),
            "font.family": "sans-serif",
            "font.sans-serif": ["Computer Modern Sans Serif"],
            "font.size": _FONT_SIZE_MAIN_PT,
            "figure.dpi": DEFAULT_PPI,
            "axes.titlesize": _FONT_SIZE_IN_BETWEEN,
            "axes.labelsize": _FONT_SIZE_IN_BETWEEN,
            "legend.fontsize": _FONT_SIZE_IN_BETWEEN,
            "xtick.labelsize": _FONT_SIZE_SCRIPTSIZE_PT,
            "ytick.labelsize": _FONT_SIZE_SCRIPTSIZE_PT,
            "lines.linewidth": LINE_WIDTH_PT,
            "patch.linewidth": LINE_WIDTH_PT,
            "lines.markersize": 4.5,
            "errorbar.capsize": 2.0,
            "legend.frameon": False,
            "legend.handlelength": 1.4,
            "legend.borderpad": 0.1,
            "legend.borderaxespad": 0.2,
            "legend.labelspacing": 0.2,
            "legend.columnspacing": 1.0,
            "legend.handletextpad": 0.4,
            "legend.loc": "center",
            "savefig.dpi": "figure",
            "savefig.pad_inches": 0.0,
            "savefig.transparent": True,
            "figure.constrained_layout.use": True,
            "figure.figsize": FIGURE_SIZE_FULL_IN,
            "grid.color": "#c0c0c0",
            "grid.linestyle": "-",
            "grid.linewidth": 0.25,
            "grid.alpha": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.6,
            "xtick.minor.visible": False,
            "ytick.minor.visible": False,
        }
    )


def resize_figure_with_legend(fig: matplotlib.figure.Figure) -> None:
    fig.canvas.draw()
    legends = [child for child in fig.get_children() if isinstance(child, matplotlib.legend.Legend)]
    if not legends:
        return
    legend = legends[0]
    fig.set_size_inches(
        fig.get_size_inches()[0],
        fig.get_size_inches()[1] + legend.get_window_extent().height / fig.dpi,
    )
