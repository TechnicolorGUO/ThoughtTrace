"""Small plotting helpers. matplotlib is imported lazily so pure-compute code
and tests run on machines without it; figure functions raise a clear error if
called without it installed."""

from __future__ import annotations

from pathlib import Path


def _plt():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "matplotlib is required to render figures; install it or use the "
            "JSON/CSV outputs instead."
        ) from e


def barh_counts(
    counts: dict[str, int],
    title: str,
    out_path: str | Path,
    *,
    pct: bool = True,
    color: str = "#4C72B0",
) -> Path:
    """Horizontal bar chart of a label->count mapping, largest at top."""
    plt = _plt()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    items = sorted(counts.items(), key=lambda kv: kv[1])
    labels = [k for k, _ in items]
    values = [v for _, v in items]
    total = sum(values) or 1

    fig, ax = plt.subplots(figsize=(8, max(2.5, 0.5 * len(labels) + 1)))
    bars = ax.barh(labels, values, color=color)
    ax.set_title(title)
    for bar, v in zip(bars, values):
        txt = f"{v} ({100 * v / total:.1f}%)" if pct else f"{v}"
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, " " + txt,
                va="center", fontsize=9)
    ax.margins(x=0.15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
