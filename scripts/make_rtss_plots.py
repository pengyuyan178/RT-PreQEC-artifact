"""Generate RTSS-focused figures from real-stream evaluation outputs."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
import typer

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rt_preqec.utils import ensure_parent

app = typer.Typer(add_completion=False)

PAPER_FONT_SIZE = 8
LER_BREAK_LOW_MAX = 0.02
LER_BREAK_HIGH_MIN = 0.16
LER_BREAK_HIGH_MAX = 0.19
LER_BREAK_LOW_SPAN = 0.78
LER_BREAK_HIGH_START = 0.88

plt.rcParams.update(
    {
        "font.size": PAPER_FONT_SIZE,
        "axes.labelsize": PAPER_FONT_SIZE,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

MODE_LABELS = {
    "accurate_only": "Accurate-only",
    "fast_only": "Fast-only",
    "edf": "EDF",
    "heuristic_pre_fixed": "Front-end only",
    "risk_heuristic": "Rule-risk only",
    "rt_qec": "Heuristic runtime",
    "rt_qec_ai": "RT-PreQEC",
    "ai_risk": "Learned risk only",
    "rt_qec_without_validation": "No validation",
    "rt_qec_without_abstention": "No abstention",
    "rt_qec_without_scheduler": "No scheduler",
    "oracle_predecoder": "Oracle front-end",
    "oracle_risk": "Oracle risk",
}

MODE_COLORS = {
    "accurate_only": "#4c78a8",
    "fast_only": "#f58518",
    "edf": "#54a24b",
    "heuristic_pre_fixed": "#b279a2",
    "rt_qec": "#9d755d",
    "rt_qec_ai": "#e45756",
    "oracle_predecoder": "#72b7b2",
    "oracle_risk": "#6f4e7c",
}

MODE_MARKERS = {
    "accurate_only": "o",
    "fast_only": "s",
    "edf": "^",
    "rt_qec": "s",
    "rt_qec_ai": "D",
    "oracle_predecoder": "P",
    "oracle_risk": "X",
}

MAIN_PLOT_MODES = [
    "accurate_only",
    "fast_only",
    "edf",
    "rt_qec_ai",
    "oracle_predecoder",
    "oracle_risk",
]

CDF_PLOT_MODES = [
    "accurate_only",
    "fast_only",
    "edf",
    "heuristic_pre_fixed",
    "rt_qec",
    "rt_qec_ai",
]

THRESHOLD_FRONTIER_DISPLAY_THRESHOLDS = [0.10, 0.15, 0.25, 0.30, 0.35, 0.40, 0.50]

MODE_LINESTYLES = {
    "accurate_only": "-",
    "fast_only": "--",
    "edf": ":",
    "heuristic_pre_fixed": "-.",
    "rt_qec": "--",
    "rt_qec_ai": "-",
}


def _mode_label(mode: str) -> str:
    return MODE_LABELS.get(str(mode), str(mode))


def _ordered_available_modes(frame: pd.DataFrame, preferred: list[str]) -> list[str]:
    available = set(frame["mode"].astype(str)) if "mode" in frame else set()
    return [mode for mode in preferred if mode in available]


def _broken_ler_position(values: float | pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    clipped = np.clip(arr, 0.0, LER_BREAK_HIGH_MAX)
    low = clipped <= LER_BREAK_LOW_MAX
    out = np.empty_like(clipped, dtype=float)
    out[low] = clipped[low] / LER_BREAK_LOW_MAX * LER_BREAK_LOW_SPAN
    high_span = max(LER_BREAK_HIGH_MAX - LER_BREAK_HIGH_MIN, 1e-9)
    out[~low] = LER_BREAK_HIGH_START + (
        (clipped[~low] - LER_BREAK_HIGH_MIN)
        / high_span
        * (1.0 - LER_BREAK_HIGH_START)
    )
    return out


def _rate_tick_label(value: float) -> str:
    if value == 0:
        return "0"
    if value < 0.01:
        return f"{value * 100:.1f}%"
    return f"{value * 100:.0f}%"


def _draw_y_axis_break(ax: plt.Axes) -> None:
    y = (LER_BREAK_LOW_SPAN + LER_BREAK_HIGH_START) / 2
    for offset in [-0.014, 0.014]:
        ax.plot(
            [-0.018, 0.018],
            [y + offset - 0.018, y + offset + 0.018],
            transform=ax.transAxes,
            color="#333333",
            linewidth=0.75,
            clip_on=False,
        )


def _draw_x_axis_break(ax: plt.Axes) -> None:
    x = (LER_BREAK_LOW_SPAN + LER_BREAK_HIGH_START) / 2
    for offset in [-0.014, 0.014]:
        ax.plot(
            [x + offset - 0.018, x + offset + 0.018],
            [-0.018, 0.018],
            transform=ax.transAxes,
            color="#333333",
            linewidth=0.75,
            clip_on=False,
        )


def _apply_ler_axis_break(ax: plt.Axes, axis: str) -> None:
    tick_values = [0.0, 0.005, 0.01, 0.02, 0.18]
    tick_positions = _broken_ler_position(np.array(tick_values))
    tick_labels = [_rate_tick_label(value) for value in tick_values]
    if axis == "x":
        ax.set_xlim(-0.04, 1.04)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        _draw_x_axis_break(ax)
    elif axis == "y":
        ax.set_ylim(-0.04, 1.04)
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels)
        _draw_y_axis_break(ax)


def _savefig(path: str | Path) -> None:
    plt.tight_layout()
    plt.savefig(ensure_parent(path), dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close()


def _diagram_box(ax, x: float, y: float, w: float, h: float, text: str, *, fc: str = "#f4f7fb") -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.0,
        edgecolor="#333333",
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8)


def _diagram_arrow(
    ax,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    text: str | None = None,
    *,
    dx: float = 0.0,
    dy: float = 0.0,
    rad: float = 0.0,
) -> None:
    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=9,
        linewidth=0.9,
        color="#333333",
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arrow)
    if text:
        ax.text((x1 + x2) / 2 + dx, (y1 + y2) / 2 + dy, text, ha="center", va="center", fontsize=7)


def _plot_lag_model_diagram(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.4, 1.75))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.2)
    ax.axis("off")
    for idx, label in enumerate(["$J_t$", "$J_{t+1}$", "$J_{t+2}$", "$J_b$"]):
        _diagram_box(ax, 0.35 + idx * 0.65, 1.85, 0.48, 0.38, label, fc="#ffffff")
    ax.plot([0.35, 2.78], [1.62, 1.62], color="#555555", linewidth=0.8)
    ax.text(1.55, 1.36, "arrivals at period $T$", ha="center", va="center", fontsize=7)
    ax.plot([2.32, 2.32], [2.27, 2.85], linestyle=":", color="#333333", linewidth=0.9)
    ax.text(2.32, 3.02, "boundary", ha="center", va="center", fontsize=7)
    _diagram_box(ax, 3.25, 1.7, 1.2, 0.62, "decode\nqueue", fc="#eef5ff")
    _diagram_box(ax, 5.05, 1.7, 1.35, 0.62, "lag-aware\nscheduler", fc="#eefaf1")
    _diagram_box(ax, 6.95, 1.7, 1.15, 0.62, "ordered\ncommit", fc="#fff8e8")
    _diagram_box(ax, 6.95, 0.42, 1.15, 0.62, "Pauli\nframe", fc="#fff8e8")
    _diagram_arrow(ax, 2.78, 2.04, 3.25, 2.01)
    _diagram_arrow(ax, 4.45, 2.01, 5.05, 2.01)
    _diagram_arrow(ax, 6.4, 2.01, 6.95, 2.01, "$f(t)$", dy=0.18)
    _diagram_arrow(ax, 7.52, 1.7, 7.52, 1.04)
    _diagram_arrow(ax, 6.95, 0.74, 4.05, 1.68, "$L_t$", dx=-0.25, dy=-0.25, rad=-0.25)
    ax.text(5.75, 2.88, "$L_t > L_{\\max}$ triggers\noverload routing", ha="center", va="center", fontsize=7)
    _savefig(out_path)


def _plot_architecture_diagram(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.4, 2.0))
    ax.set_xlim(0, 10.6)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    _diagram_box(ax, 0.25, 1.75, 1.05, 0.62, "syndrome\njob", fc="#ffffff")
    _diagram_box(ax, 1.75, 1.75, 1.45, 0.62, "front-end\ncertificate", fc="#eef5ff")
    _diagram_box(ax, 3.75, 1.75, 1.45, 0.62, "lag-aware\nscheduler", fc="#eefaf1")
    _diagram_box(ax, 5.85, 2.65, 1.25, 0.62, "fast\nbackend", fc="#fff8e8")
    _diagram_box(ax, 5.85, 0.85, 1.25, 0.62, "accurate\nbackend", fc="#f7f0ff")
    _diagram_box(ax, 7.65, 2.65, 1.3, 0.62, "validation\ngate", fc="#fdeff1")
    _diagram_box(ax, 9.55, 1.75, 0.78, 0.62, "commit", fc="#ffffff")
    _diagram_arrow(ax, 1.3, 2.06, 1.75, 2.06)
    _diagram_arrow(ax, 3.2, 2.06, 3.75, 2.06, "risk, conf.", dy=0.2)
    _diagram_arrow(ax, 5.2, 2.22, 5.85, 2.88, "strong cert.", dx=-0.05, dy=0.25)
    _diagram_arrow(ax, 5.2, 1.9, 5.85, 1.16, "weak/no cert.", dy=-0.28)
    _diagram_arrow(ax, 7.1, 2.96, 7.65, 2.96)
    _diagram_arrow(ax, 8.95, 2.96, 9.55, 2.2, "pass", dx=0.12, dy=0.18)
    _diagram_arrow(ax, 8.3, 2.65, 6.52, 1.47, "reject", dy=-0.22, rad=-0.12)
    _diagram_arrow(ax, 7.1, 1.16, 9.55, 1.88)
    ax.text(2.48, 1.16, "weak cert.:\nshape only", ha="center", va="center", fontsize=7)
    ax.text(4.48, 2.95, "learned risk\nestimate", ha="center", va="center", fontsize=7)
    _savefig(out_path)


def _annotated_scatter(
    frame: pd.DataFrame,
    x_column: str,
    y_column: str,
    x_label: str,
    y_label: str,
    out_path: Path,
) -> None:
    if frame.empty or x_column not in frame or y_column not in frame:
        return
    fig, ax = plt.subplots(figsize=(2.25, 1.85))
    for _, row in frame.iterrows():
        mode = str(row["mode"])
        ax.scatter(
            row[x_column],
            row[y_column],
            s=18,
            marker=MODE_MARKERS.get(mode, "o"),
            color=MODE_COLORS.get(mode, "#4c78a8"),
            linewidths=0.4,
            edgecolors="white",
            zorder=3,
        )
        ax.annotate(
            _mode_label(mode),
            (row[x_column], row[y_column]),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=6.2,
        )
    ax.set_xlabel(x_label, labelpad=1.5)
    ax.set_ylabel(y_label, labelpad=1.5)
    ax.grid(True, linewidth=0.35, alpha=0.28)
    ax.tick_params(width=0.6, length=2.5, pad=1.5)
    _savefig(out_path)


def _scatter_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    x_column: str,
    y_column: str,
    x_label: str,
    y_label: str,
    ler_break_axis: str | None = None,
) -> list[plt.Artist]:
    handles: list[plt.Artist] = []
    for _, row in frame.iterrows():
        mode = str(row["mode"])
        x_value = float(row[x_column])
        y_value = float(row[y_column])
        if ler_break_axis == "x":
            x_value = float(_broken_ler_position(np.array([x_value]))[0])
        elif ler_break_axis == "y":
            y_value = float(_broken_ler_position(np.array([y_value]))[0])
        artist = ax.scatter(
            x_value,
            y_value,
            s=22,
            marker=MODE_MARKERS.get(mode, "o"),
            color=MODE_COLORS.get(mode, "#4c78a8"),
            linewidths=0.45,
            edgecolors="white",
            zorder=3,
        )
        handles.append(artist)
    ax.set_xlabel(x_label, labelpad=1.0)
    ax.set_ylabel(y_label, labelpad=1.0)
    ax.set_box_aspect(1.0)
    ax.margins(x=0.08, y=0.12)
    if ler_break_axis:
        _apply_ler_axis_break(ax, ler_break_axis)
    ax.grid(True, linewidth=0.35, alpha=0.28)
    ax.tick_params(width=0.6, length=2.5, pad=1.2)
    return handles


def _plot_main_pareto_frontier(frame: pd.DataFrame, out_path: Path) -> None:
    needed = {
        "mode",
        "pauli_frame_lag_violation_ratio",
        "logical_error_rate",
        "p99_response_time_us",
        "boundary_commit_success_rate",
    }
    if frame.empty or not needed.issubset(frame.columns):
        return
    mode_order = {mode: index for index, mode in enumerate(MAIN_PLOT_MODES)}
    frame = frame.copy()
    frame["_plot_order"] = frame["mode"].astype(str).map(lambda mode: mode_order.get(mode, len(mode_order)))
    frame = frame.sort_values("_plot_order").drop(columns=["_plot_order"])
    fig, axes = plt.subplots(1, 3, figsize=(7.05, 1.95))
    specs = [
        (
            "pauli_frame_lag_violation_ratio",
            "logical_error_rate",
            "Lag violation ratio",
            "Logical error rate",
            "y",
        ),
        (
            "p99_response_time_us",
            "logical_error_rate",
            "p99 response time (us)",
            "Logical error rate",
            "y",
        ),
        (
            "logical_error_rate",
            "boundary_commit_success_rate",
            "Logical error rate",
            "Boundary success rate",
            "x",
        ),
    ]
    legend_handles: list[plt.Artist] = []
    for index, (ax, spec) in enumerate(zip(axes, specs)):
        handles = _scatter_panel(ax, frame, *spec)
        if index == 0:
            legend_handles = handles
        ax.set_title(
            f"({chr(ord('a') + index)})",
            loc="left",
            fontsize=PAPER_FONT_SIZE,
            y=1.05,
            pad=0.0,
        )
    labels = [_mode_label(str(mode)) for mode in frame["mode"].astype(str).tolist()]
    fig.legend(
        legend_handles,
        labels,
        loc="center left",
        ncol=1,
        frameon=False,
        bbox_to_anchor=(0.83, 0.55),
        handletextpad=0.25,
        columnspacing=0.65,
        labelspacing=1.05,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(left=0.075, right=0.805, top=0.88, bottom=0.22, wspace=0.38)
    fig.savefig(ensure_parent(out_path), dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _plot_lag_over_time(run_path: Path, modes: list[str], out_path: Path) -> None:
    plt.figure(figsize=(8, 4))
    plotted = False
    for mode in modes:
        path = run_path / mode / "events.csv"
        if not path.exists():
            continue
        events = pd.read_csv(path)
        if "shot_id" not in events or "pauli_frame_lag" not in events:
            continue
        plt.plot(events["shot_id"], events["pauli_frame_lag"], linewidth=1.1, label=_mode_label(mode))
        plotted = True
    if not plotted:
        plt.close()
        return
    plt.xlabel("Shot ID")
    plt.ylabel("Pauli-frame lag")
    plt.legend(fontsize=7)
    _savefig(out_path)


def _plot_response_time_cdf(run_path: Path, modes: list[str], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.45, 2.2))
    plotted = False
    positive_values: list[float] = []
    for mode in modes:
        path = run_path / mode / "events.csv"
        if not path.exists():
            continue
        events = pd.read_csv(path)
        if "response_time_us" not in events:
            continue
        values = np.sort(events["response_time_us"].to_numpy(dtype=float))
        values = values[np.isfinite(values) & (values > 0)]
        if values.size == 0:
            continue
        positive_values.extend([float(values[0]), float(values[-1])])
        y = np.arange(1, values.size + 1, dtype=float) / float(values.size)
        ax.plot(
            values,
            y,
            label=_mode_label(mode),
            color=MODE_COLORS.get(mode),
            linestyle=MODE_LINESTYLES.get(mode, "-"),
            linewidth=1.15 if mode != "rt_qec_ai" else 1.45,
            alpha=0.95,
        )
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xscale("log")
    if positive_values:
        x_min = max(min(positive_values) * 0.85, 1e-3)
        x_max = max(positive_values) * 1.12
        ax.set_xlim(x_min, x_max)
        ticks = [5, 10, 20, 50, 100, 200, 500, 1000, 5000]
        ticks = [tick for tick in ticks if x_min <= tick <= x_max]
        if ticks:
            ax.set_xticks(ticks)
            ax.set_xticklabels([str(tick) for tick in ticks])
    ax.axhline(0.99, color="#333333", linestyle=":", linewidth=0.65, alpha=0.65)
    ax.text(0.02, 0.965, "p99", transform=ax.transAxes, ha="left", va="center", fontsize=6.5)
    ax.set_xlabel("Response time (us, log scale)", labelpad=1.5)
    ax.set_ylabel("CDF", labelpad=1.5)
    ax.set_ylim(0.0, 1.005)
    ax.grid(True, linewidth=0.35, alpha=0.28)
    ax.tick_params(width=0.6, length=2.5, pad=1.5)
    ax.legend(
        loc="lower right",
        frameon=True,
        framealpha=0.88,
        borderpad=0.22,
        handlelength=1.5,
        handletextpad=0.35,
        ncol=1,
        columnspacing=0.0,
        labelspacing=0.3,
    )
    _savefig(out_path)


def _plot_burst_lag_backlog_trace(run_path: Path, modes: list[str], out_path: Path) -> None:
    selected_modes = [mode for mode in ["accurate_only", "rt_qec_ai", "fast_only", "oracle_predecoder"] if mode in modes]
    if not selected_modes:
        selected_modes = modes[:4]
    fig, axes = plt.subplots(2, 1, figsize=(3.45, 2.55), sharex=True)
    plotted = False
    deadline_values: list[float] = []
    for mode in selected_modes:
        path = run_path / mode / "events.csv"
        if not path.exists():
            continue
        events = pd.read_csv(path)
        if not {"shot_id", "pauli_frame_lag", "response_time_us"}.issubset(events.columns):
            continue
        color = MODE_COLORS.get(mode)
        axes[0].plot(events["shot_id"], events["pauli_frame_lag"], linewidth=0.7, label=_mode_label(mode), color=color)
        axes[1].plot(events["shot_id"], events["response_time_us"], linewidth=0.7, label=_mode_label(mode), color=color)
        if "deadline_us" in events:
            deadline_values.extend(events["deadline_us"].dropna().astype(float).unique().tolist())
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    axes[0].set_ylabel("Pauli-frame lag")
    axes[1].set_ylabel("Response (us)")
    axes[1].set_xlabel("Shot ID")
    if deadline_values:
        deadline = float(pd.Series(deadline_values).mode().iloc[0])
        axes[1].axhline(deadline, color="#333333", linestyle="--", linewidth=0.7, alpha=0.75)
        axes[1].text(0.99, 0.86, "deadline", transform=axes[1].transAxes, ha="right", va="center", fontsize=6.5)
    for ax in axes:
        ax.grid(True, linewidth=0.35, alpha=0.28)
        ax.tick_params(width=0.6, length=2.5, pad=1.5)
    axes[0].legend(
        loc="upper right",
        frameon=True,
        framealpha=0.9,
        ncol=2,
        borderpad=0.22,
        handlelength=1.0,
        handletextpad=0.25,
        columnspacing=0.45,
        labelspacing=0.25,
    )
    _savefig(out_path)


def _plot_threshold_frontier(path: Path, out_path: Path) -> None:
    if not path.exists():
        return
    frame = pd.read_csv(path)
    needed = {"logical_error_rate"}
    if frame.empty or not needed.issubset(frame.columns):
        return
    x_column = "scheduler_risk_threshold" if "scheduler_risk_threshold" in frame else "risk_threshold"
    if x_column not in frame:
        return
    right_column = "p99_response_time_us" if "p99_response_time_us" in frame else "fast_selection_rate"
    if right_column not in frame:
        return
    if (
        x_column == "scheduler_risk_threshold"
        and "mode" in frame
        and "predecode_risk_threshold" in frame
        and "confidence_threshold" in frame
    ):
        sweep_dir = path.parent / "threshold_sweep"
        predecode_threshold = float(frame["predecode_risk_threshold"].dropna().iloc[0])
        confidence_threshold = float(frame["confidence_threshold"].dropna().iloc[0])
        existing = set(zip(frame["mode"].astype(str), frame[x_column].round(2)))
        extra_rows: list[pd.DataFrame] = []
        for mode in frame["mode"].astype(str).unique().tolist():
            for threshold in THRESHOLD_FRONTIER_DISPLAY_THRESHOLDS:
                if (mode, round(threshold, 2)) in existing:
                    continue
                summary_path = (
                    sweep_dir
                    / f"{mode}_pre_{predecode_threshold:.2f}_sched_{threshold:.2f}_conf_{confidence_threshold:.2f}"
                    / "summary_metrics.csv"
                )
                if not summary_path.exists():
                    continue
                summary = pd.read_csv(summary_path)
                summary["predecode_risk_threshold"] = predecode_threshold
                summary["scheduler_risk_threshold"] = threshold
                summary["risk_threshold"] = threshold
                summary["confidence_threshold"] = confidence_threshold
                extra_rows.append(summary)
        if extra_rows:
            frame = pd.concat([frame, *extra_rows], ignore_index=True, sort=False)
        display_thresholds = {round(threshold, 2) for threshold in THRESHOLD_FRONTIER_DISPLAY_THRESHOLDS}
        frame = frame[frame[x_column].round(2).isin(display_thresholds)].copy()
    if frame.empty:
        return

    group_cols = ["mode", x_column] if "mode" in frame else [x_column]
    plot_frame = (
        frame.groupby(group_cols, as_index=False)
        .agg(
            logical_error_rate=("logical_error_rate", "mean"),
            right_metric=(right_column, "mean"),
        )
        .sort_values(group_cols)
    )
    x_values = plot_frame[x_column].to_numpy(dtype=float)
    x_min = float(np.nanmin(x_values))
    x_max = float(np.nanmax(x_values))
    threshold_positions = {
        round(threshold, 2): index for index, threshold in enumerate(THRESHOLD_FRONTIER_DISPLAY_THRESHOLDS)
    }
    use_threshold_positions = (
        x_column == "scheduler_risk_threshold"
        and set(np.round(x_values, 2)).issubset(set(threshold_positions.keys()))
    )

    fig, axes = plt.subplots(1, 2, figsize=(3.65, 2.18), sharex=False)
    if "mode" in plot_frame:
        preferred = ["rt_qec", "rt_qec_ai"]
        available = plot_frame["mode"].astype(str).unique().tolist()
        modes = [mode for mode in preferred if mode in available] + [
            mode for mode in available if mode not in preferred
        ]
    else:
        modes = ["threshold"]
    legend_handles: list[plt.Artist] = []
    for mode in modes:
        group = plot_frame[plot_frame["mode"].astype(str) == mode] if "mode" in plot_frame else plot_frame
        x_raw = group[x_column].to_numpy(dtype=float)
        if use_threshold_positions:
            x = np.array([threshold_positions[round(value, 2)] for value in x_raw], dtype=float)
        else:
            x = x_raw
        ler = group["logical_error_rate"].to_numpy(dtype=float) * 100.0
        right = group["right_metric"].to_numpy(dtype=float)
        if right_column == "fast_selection_rate":
            right = right * 100.0
        label = _mode_label(mode)
        color = MODE_COLORS.get(mode)
        marker = MODE_MARKERS.get(mode, "o")
        for panel_index, (ax, values) in enumerate(zip(axes, [ler, right])):
            (line,) = ax.plot(
                x,
                values,
                marker=marker,
                markersize=3.0,
                linewidth=1.15,
                color=color,
                label=label,
                markeredgecolor="white",
                markeredgewidth=0.35,
                alpha=0.96,
            )
            if panel_index == 0:
                legend_handles.append(line)

    x_label = "Scheduler threshold" if x_column == "scheduler_risk_threshold" else "Risk threshold"
    axes[0].set_ylabel("LER (%)", labelpad=1.5)
    axes[1].set_ylabel("p99 response (us)" if right_column == "p99_response_time_us" else "Fast selection (%)", labelpad=1.5)
    for index, ax in enumerate(axes):
        selected_x = threshold_positions.get(0.30, 0.30) if use_threshold_positions else 0.30
        ax.axvline(selected_x, color="#333333", linestyle=":", linewidth=0.7, alpha=0.75, zorder=1)
        ax.set_xlabel(x_label, labelpad=1.5)
        ax.set_title(f"({chr(ord('a') + index)})", loc="left", fontsize=PAPER_FONT_SIZE, pad=0.0)
        ax.grid(True, linewidth=0.35, alpha=0.28)
        ax.tick_params(width=0.6, length=2.5, pad=1.2)
        if use_threshold_positions:
            xticks = list(range(len(THRESHOLD_FRONTIER_DISPLAY_THRESHOLDS)))
            xtick_labels = [f"{tick:.2f}" for tick in THRESHOLD_FRONTIER_DISPLAY_THRESHOLDS]
            ax.set_xlim(-0.35, len(THRESHOLD_FRONTIER_DISPLAY_THRESHOLDS) - 0.65)
        else:
            xticks = THRESHOLD_FRONTIER_DISPLAY_THRESHOLDS
            xtick_labels = [f"{tick:.2f}" for tick in xticks]
            ax.set_xlim(max(0.0, x_min - 0.02), x_max + 0.02)
        ax.set_xticks(xticks)
        ax.set_xticklabels(
            xtick_labels,
            rotation=30,
            ha="right",
        )
        ax.tick_params(axis="x", labelsize=6)
    axes[0].set_ylim(bottom=0.0)
    if right_column == "p99_response_time_us":
        right_min = float(plot_frame["right_metric"].min())
        right_max = float(plot_frame["right_metric"].max())
        y_lower = max(0.0, right_min - 2.0)
        y_upper = right_max + 1.5
        axes[1].set_ylim(y_lower, y_upper)
        axes[1].set_yticks(np.arange(np.ceil(y_lower / 5.0) * 5.0, y_upper + 0.1, 5.0))
    fig.legend(
        legend_handles,
        [_mode_label(mode) for mode in modes],
        loc="upper center",
        bbox_to_anchor=(0.52, 1.03),
        ncol=2,
        frameon=True,
        framealpha=0.94,
        edgecolor="#8a8a8a",
        facecolor="white",
        handlelength=1.2,
        handletextpad=0.35,
        columnspacing=1.35,
        borderpad=0.28,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(left=0.12, right=0.98, top=0.80, bottom=0.31, wspace=0.50)
    fig.savefig(ensure_parent(out_path), dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


@app.command()
def main(
    run_dir: str = typer.Option("results/runs/paper_suite_d7_rtqec_ai_selected/main", "--run-dir"),
    out: str = typer.Option("results/figures/paper_rtss_v2", "--out"),
    threshold_sweep: str | None = typer.Option(None, "--threshold-sweep"),
    burst_run_dir: str | None = typer.Option(None, "--burst-run-dir"),
) -> None:
    """Render Pauli-frame-lag, response-tail, and boundary-commit figures."""
    run_path = Path(run_dir)
    out_path = Path(out)
    summary_path = run_path / "summary_metrics.csv"
    if not summary_path.exists():
        raise typer.BadParameter(f"missing summary_metrics.csv under {run_path}")
    summary = pd.read_csv(summary_path)
    modes = summary["mode"].astype(str).tolist() if "mode" in summary else []
    main_modes = _ordered_available_modes(summary, MAIN_PLOT_MODES)
    cdf_modes = _ordered_available_modes(summary, CDF_PLOT_MODES)
    main_summary = summary[summary["mode"].astype(str).isin(main_modes)].copy() if main_modes else summary

    _plot_lag_model_diagram(out_path / "lag_model_task_flow.png")
    _plot_architecture_diagram(out_path / "rt_qec_architecture.png")

    _plot_main_pareto_frontier(main_summary, out_path / "main_pareto_frontier.png")
    _annotated_scatter(
        main_summary,
        "pauli_frame_lag_violation_ratio",
        "logical_error_rate",
        "Pauli-frame lag violation ratio",
        "Logical error rate",
        out_path / "logical_error_rate_vs_pauli_frame_lag_violation.png",
    )
    _annotated_scatter(
        main_summary,
        "p99_response_time_us",
        "logical_error_rate",
        "p99 response time (us)",
        "Logical error rate",
        out_path / "logical_error_rate_vs_p99_response_time.png",
    )
    _annotated_scatter(
        main_summary,
        "logical_error_rate",
        "boundary_commit_success_rate",
        "Logical error rate",
        "Boundary commit success rate",
        out_path / "boundary_commit_success_vs_logical_error.png",
    )
    _plot_lag_over_time(run_path, main_modes or modes, out_path / "pauli_frame_lag_over_time_by_mode.png")
    _plot_response_time_cdf(run_path, cdf_modes or main_modes or modes, out_path / "response_time_cdf_by_mode.png")

    sweep_path = Path(threshold_sweep) if threshold_sweep else run_path.parent / "threshold_sweep.csv"
    _plot_threshold_frontier(sweep_path, out_path / "threshold_sweep_lag_frontier.png")

    if burst_run_dir is not None:
        burst_run_path = Path(burst_run_dir)
        burst_summary_path = burst_run_path / "summary_metrics.csv"
        if burst_summary_path.exists():
            burst_summary = pd.read_csv(burst_summary_path)
            burst_modes = burst_summary["mode"].astype(str).tolist() if "mode" in burst_summary else []
            _plot_burst_lag_backlog_trace(
                burst_run_path,
                burst_modes,
                out_path / "burst_lag_backlog_trace.png",
            )


if __name__ == "__main__":
    app()
