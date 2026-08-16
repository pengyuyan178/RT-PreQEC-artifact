# `figure/` and `table/` — paper artifacts

The figures and the CSV data behind the RTSS 2026 submission. Everything here is a
copy: `figure/` mirrors the assets the paper includes, and `table/` holds the source
data the data-driven figures and the LaTeX tables were made from, so a reader can
check a published number without rerunning the experiments.

> These files mirror the **current** paper (Overleaf revision `58f61e6`). The figure
> set and numbering differ from the original submission — the paper now uses **ten
> figures (figure1–figure9 plus `t1.png`)** and **three tables** (`tab:regimes`,
> `tab:modes`, `tab:overhead`). The original `tab:main` and `tab:ablation-delta`
> tables were replaced by two figures (`t1.png` and `figure9.pdf`).

Regenerating a data figure needs the event-level CSVs, not just the summaries, which
is why `table/figure_data/` carries per-mode `events.csv` files (~14 MB total).

## `figure/` — what the paper includes

Four of the ten are hand-drawn schematics with no experimental data behind them
(`figure1`–`figure4`). The rest are data-driven.

| File | Paper label | Location | Content | Regenerable from |
|---|---|---|---|---|
| `figure1.pdf` | `fig:intro-loop` | §1 Introduction | Real-time QEC control loop (stabilizer → decoder → Pauli frame → commit) | schematic, hand-drawn |
| `figure2.pdf` | `fig:lag-model` | §3 System Model | Time-domain view of lag-bounded processing | schematic, hand-drawn |
| `figure3.pdf` | `fig:architecture` | §4 Design | RT-PreQEC pipeline / safety contract | schematic, hand-drawn |
| `figure4.pdf` | `fig:priority-composition` | §4 Design | Priority-signal composition (urgency / risk / cost / boundary) | schematic, hand-drawn |
| `figure5.pdf` | `fig:pareto` | §5 Evaluation Setup | Main operating-point comparison, 3 panels | `d7_main/summary_metrics.csv` |
| `figure6.pdf` | `fig:cdf` | §5 Evaluation Setup | Response-time CDF by mode | `d7_main/<mode>/events.csv` |
| `t1.png` | `fig:main` | §6 Results | Main-results 4-panel figure (LER / deadline-miss / lag / boundary-accept) | derived from `d7_main` data |
| `figure7.pdf` | `fig:burst` | §7 Ablations | Burst-regime lag and response over time | `burst/<mode>/events.csv` |
| `figure8.pdf` | `fig:sweep` | §7 Ablations | Decoupled threshold-sweep frontier | `threshold_sweep/threshold_sweep.csv` |
| `figure9.pdf` | `fig:ablation-delta` | §8 Related Work | Ablation impact vs RT-PreQEC (delta plot) | `rtss_ablation_delta_vs_rt_qec.csv` |

`t1.png` and `figure9.pdf` are rendered from the `d7_main` summary and the
ablation-delta CSV respectively, but the exact plotting script that produced the
published artwork is not part of this artifact — treat the CSVs as the source of
truth and the images as the published rendering of that data.

## `table/` — paper table data

The paper has three tables. Only one is computed from data; the other two are
hand-written configuration tables.

| File | Paper table |
|---|---|
| `rtss_overhead_table.csv` | `tab:overhead` (§7) — per-component microbenchmark overhead |

`tab:regimes` (§3) and `tab:modes` (§5) are hand-written with no computed CSV behind
them. `regime_summary_table.csv` covers the same three regimes with measured metrics
rather than the declared parameters, so it corroborates `tab:regimes` without being
its source.

The remaining CSVs are **supporting data** that back figures and in-text numbers —
they are not tables in the paper:

| File | What it backs |
|---|---|
| `rtss_main_table.csv` | the `t1.png` main-results figure (`fig:main`) |
| `rtss_main_ablation_table.csv` | superset of the above, all 8 ablation modes |
| `rtss_ablation_delta_vs_rt_qec.csv` | the `figure9.pdf` ablation-delta figure (`fig:ablation-delta`) |
| `rtss_safety_contract_table.csv` | safety-contract comparison across the 4 headline modes |
| `rtss_ai_risk_table.csv` | learned-risk vs heuristic-risk scheduler input |
| `rtss_burst_capacity_table.csv` | burst regime at 1–2 workers, for the headroom discussion |
| `regime_summary_table.csv` | all three regimes x 5 modes, measured |
| `frontend_contract_table.csv` | front-end accept / abstain / false-accept rates |

`rtss_overhead_measurement_protocol.json` records the conditions behind
`rtss_overhead_table.csv` — 1 900 shots after 100 warm-up, single thread, no CPU
pinning — which is what makes those microbenchmark numbers interpretable.

Two mode-name mappings are not self-evident from the paper:

- The paper's **RT-PreQEC** row is mode `rt_qec_ai` (learned risk).
- The paper's **"Heuristic runtime" / "Heuristic certified"** row is mode `rt_qec`.

### `table/figure_data/` — figure source data

```text
figure_data/
├── d7_main/                    d=7 selected operating point (figures 5, 6, and t1.png)
│   ├── summary_metrics.csv     13 modes x 40 metrics
│   └── <mode>/events.csv       8 modes, 4000 rows each
├── burst/                      burst regime (figure 7)
│   ├── summary_metrics.csv
│   └── <mode>/events.csv       13 modes
└── threshold_sweep/
    └── threshold_sweep.csv     decoupled sweep (figure 8), 7 thresholds x 2 modes
```

Only the modes figures 5 and 6 actually plot were copied for `d7_main` (8 of the
run's 13 — see `MAIN_PLOT_MODES` / `CDF_PLOT_MODES` in `scripts/make_rtss_plots.py`).
`burst/` carries all 13, because figure 7 plots whatever modes the summary lists.

## Regenerating the data figures

The four data figures that `scripts/make_rtss_plots.py` produces map onto the current
paper numbering as follows: `main_pareto_frontier` → **figure5**,
`response_time_cdf_by_mode` → **figure6**, `burst_lag_backlog_trace` → **figure7**,
`threshold_sweep_lag_frontier` → **figure8**. Regenerate them from the committed data:

```python
import sys, tempfile
from pathlib import Path
sys.path[:0] = ["src", "."]
import pandas as pd
from scripts import make_rtss_plots as plots

data = Path("table/figure_data")
out = Path(tempfile.mkdtemp(prefix="rtss_figs_"))

d7 = data / "d7_main"
summary = pd.read_csv(d7 / "summary_metrics.csv")
main_modes = plots._ordered_available_modes(summary, plots.MAIN_PLOT_MODES)
cdf_modes = plots._ordered_available_modes(summary, plots.CDF_PLOT_MODES)
main = summary[summary["mode"].astype(str).isin(main_modes)]

plots._plot_main_pareto_frontier(main, out / "figure5.pdf")
plots._plot_response_time_cdf(d7, cdf_modes, out / "figure6.pdf")
plots._plot_burst_lag_backlog_trace(
    data / "burst",
    pd.read_csv(data / "burst" / "summary_metrics.csv")["mode"].astype(str).tolist(),
    out / "figure7.pdf",
)
plots._plot_threshold_frontier(data / "threshold_sweep" / "threshold_sweep.csv", out / "figure8.pdf")
print(out)
```

It writes to a fresh temp directory rather than over `figure/`, so a redraw can be
compared against the published assets instead of silently replacing them. The exact
styling of the published PDFs may differ slightly (they were re-rendered for the
current draft); the underlying numbers are identical.

## Verification status

The data figures were regenerated from `table/figure_data/` and compared against the
published assets by decompressing the PDF content streams (raw bytes always differ,
since a PDF embeds its creation time):

- The three-panel main comparison and the threshold-sweep frontier regenerate with
  byte-identical content streams.
- The response-time CDF and the burst trace differ only in tight-bbox canvas width
  introduced by a newer matplotlib; rendering them directly from
  `../experiment_records/` produces byte-identical output, so the difference is the
  tool version, not the data. The event CSVs match the archive row-by-row
  (max deviation 0).
- `rtss_main_table.csv` and `rtss_ablation_delta_vs_rt_qec.csv` agree with the typed
  LaTeX values to last-digit rounding.
- `rtss_overhead_table.csv` matches all 15 values in `tab:overhead` exactly
  (scheduler 1.19/3.9/28.1, validation 16.3/38.5/189.9, front-end 53.6/176.9/294.2,
  and both backends).

## Provenance

`figure/*` ← `../paper/RTSS/6a131841608c92582d0fcb7c/figure/` (current Overleaf
revision). `table/figure_data/` ←
`../experiment_records/results/runs/paper_suite_{d7_rtqec_ai_selected,burst_rtqec_ai,d7_rtqec_ai_decoupled_smoke}/`.
`table/rtss_overhead_table.csv` and `rtss_overhead_measurement_protocol.json` ←
`../experiment_records/results/runs/wcet_overheads_d7/`. All other `table/*.csv` ←
`results/tables/` in this repository.

The archives remain authoritative; these are reading copies pinned alongside the code.
Because `.gitignore` blanket-ignores `*.csv` and `*.pdf`, both directories are
un-ignored explicitly near the end of that file — if you add a file here in another
format, check `git check-ignore -v` before assuming it is committed.
