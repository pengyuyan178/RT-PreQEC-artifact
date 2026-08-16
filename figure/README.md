# `figure/` and `table/` — paper artifacts

The six figures in the RTSS 2026 submission, and the CSV data behind every figure and
table in it. Everything here is a copy: `figure/` mirrors the built PDFs the paper
includes, and `table/` holds the source data those PDFs and the LaTeX tables were made
from, so a reader can check a published number without rerunning the experiments.

Regenerating a figure needs the event-level CSVs, not just the summaries, which is why
`table/figure_data/` carries per-mode `events.csv` files (~14 MB total).

## `figure/` — what the paper includes

`figure1.pdf` and `figure2.pdf` are schematic diagrams (system model, architecture) with
no experimental data behind them. `make_rtss_plots.py` does contain generators for both
(`_plot_lag_model_diagram`, `_plot_architecture_diagram`), but the published assets were
redrawn by hand afterwards and are substantially different — 153 KB and 313 KB against
the generators' 17 KB and 12 KB. That is why `export_rtss_paper_figures.py` defaults to
`--skip-diagrams`: rerunning it with `--include-diagrams` would replace the redrawn
figures with the rougher generated ones.

| File | Paper location | Content | Regenerable from |
|---|---|---|---|
| `figure1.pdf` | §3 System Model | Time-domain view of lag-bounded processing | schematic, redrawn by hand |
| `figure2.pdf` | §4 Design | RT-PreQEC architecture | schematic, redrawn by hand |
| `figure3.pdf` | §6 Results | Main operating-point comparison, 3 panels | `d7_main/summary_metrics.csv` |
| `figure4.pdf` | §6 Results | Response-time CDF by mode | `d7_main/<mode>/events.csv` |
| `figure5.pdf` | §6 Results | Burst-regime lag and response over time | `burst/<mode>/events.csv` |
| `figure6.pdf` | §7 Ablations | Decoupled threshold sweep frontier | `threshold_sweep/threshold_sweep.csv` |

## `table/` — paper table data

The paper has five numbered tables. The LaTeX has the values typed in, so these CSVs are
how you check them:

| File | Paper table |
|---|---|
| `rtss_main_table.csv` | `tab:main` (§6) — main results, d=7, single worker |
| `rtss_ablation_delta_vs_rt_qec.csv` | `tab:ablation-delta` (§7) — ablation impact vs RT-PreQEC |
| `rtss_overhead_table.csv` | `tab:overhead` (§7) — per-component microbenchmark overhead |

`tab:regimes` (§2) and `tab:modes` (§5) are hand-written configuration tables with no
computed CSV behind them. `regime_summary_table.csv` covers the same three regimes with
measured metrics rather than the declared parameters, so it corroborates `tab:regimes`
without being its source.

The remaining CSVs are **supporting data cited in the prose or used to select the
reported operating point** — not tables in the paper. They are kept because several
in-text numbers come from them:

| File | What it holds |
|---|---|
| `rtss_main_ablation_table.csv` | main results with all 8 ablation modes (superset of `tab:main`) |
| `rtss_safety_contract_table.csv` | safety-contract comparison across the 4 headline modes |
| `rtss_ai_risk_table.csv` | learned-risk vs heuristic-risk scheduler input |
| `rtss_burst_capacity_table.csv` | burst regime at 1–4 workers, for the headroom discussion |
| `regime_summary_table.csv` | all three regimes x 5 modes, measured |
| `frontend_contract_table.csv` | front-end accept / abstain / false-accept rates |

`rtss_overhead_measurement_protocol.json` records the conditions behind
`rtss_overhead_table.csv` — 1 900 shots after 100 warm-up, single thread, no CPU pinning
— which is what makes those microbenchmark numbers interpretable.

Two mode-name mappings are not self-evident from the paper:

- The paper's **RT-PreQEC** row is mode `rt_qec_ai` (learned risk).
- The paper's **"Heuristic runtime"** ablation row is mode `rt_qec` (heuristic risk).

### `table/figure_data/` — figure source data

```text
figure_data/
├── d7_main/                    d=7 selected operating point (figures 3, 4)
│   ├── summary_metrics.csv     13 modes x 40 metrics
│   └── <mode>/events.csv       8 modes, 4000 rows each
├── burst/                      burst regime (figure 5)
│   ├── summary_metrics.csv
│   └── <mode>/events.csv       13 modes
└── threshold_sweep/
    └── threshold_sweep.csv     decoupled sweep (figure 6), 7 thresholds x 2 modes
```

Only the modes figures 3 and 4 actually plot were copied for `d7_main` (8 of the run's
13 — see `MAIN_PLOT_MODES` / `CDF_PLOT_MODES` in `scripts/make_rtss_plots.py`).
`burst/` carries all 13, because figure 5 plots whatever modes the summary lists.

Figure 6 needs only `threshold_sweep.csv`: the plotting code can backfill display
points from a sibling `threshold_sweep/<config>/` directory, but that path is
unused here because the CSV already covers every threshold in
`THRESHOLD_FRONTIER_DISPLAY_THRESHOLDS`. That directory is 91 MB and was left in
`../experiment_records/`.

## Regenerating figures 3–6

Run from the repository root:

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

plots._plot_main_pareto_frontier(main, out / "figure3.pdf")
plots._plot_response_time_cdf(d7, cdf_modes, out / "figure4.pdf")
plots._plot_burst_lag_backlog_trace(
    data / "burst",
    pd.read_csv(data / "burst" / "summary_metrics.csv")["mode"].astype(str).tolist(),
    out / "figure5.pdf",
)
plots._plot_threshold_frontier(data / "threshold_sweep" / "threshold_sweep.csv", out / "figure6.pdf")
print(out)
```

It writes to a fresh temp directory rather than over `figure/`, so a redraw can be
compared against the published PDFs instead of silently replacing them.

## Verification status

Figures 3–6 were regenerated from `table/figure_data/` and compared against the
PDFs in `figure/` by decompressing the PDF content streams (raw bytes always differ,
since a PDF embeds its creation time):

- **figure3, figure6** — content streams **byte-identical** to the paper PDFs.
- **figure4, figure5** — content differs only in tight-bbox canvas width
  (234.373 vs 234.242 pt). Rendering the same figures directly from
  `../experiment_records/` produces **byte-identical output to the copies here**, so the
  difference comes from the matplotlib version (3.10.9 now vs whatever built the
  submission), not from the data. The event CSVs were also compared row-by-row against
  the archive: 6 modes x 4000 rows, **max deviation 0**.
- `rtss_main_table.csv` and `rtss_ablation_delta_vs_rt_qec.csv` were checked against the
  typed LaTeX values and agree to last-digit rounding (e.g. `+4.97` vs `+4.98` pp).
- `rtss_overhead_table.csv` matches all 15 values in `tab:overhead` exactly (scheduler
  1.19/3.9/28.1, validation 16.3/38.5/189.9, front-end 53.6/176.9/294.2, and both
  backends).
- The regeneration snippet above was run verbatim; figure3 and figure6 came out at
  exactly the published byte size (16 569 and 13 833).

## Provenance

Copied on 2026-08-16 from:

- `figure/*.pdf` ← `../paper/RTSS/6a131841608c92582d0fcb7c/figure/`
- `table/figure_data/` ← `../experiment_records/results/runs/paper_suite_{d7_rtqec_ai_selected,burst_rtqec_ai,d7_rtqec_ai_decoupled_smoke}/`
- `table/rtss_overhead_table.csv`, `table/rtss_overhead_measurement_protocol.json`
  ← `../experiment_records/results/runs/wcet_overheads_d7/`
- all other `table/*.csv` ← `results/tables/` in this repository

The archives remain authoritative; these are reading copies pinned alongside the code.
Because `.gitignore` blanket-ignores `*.csv` and `*.pdf`, both directories are un-ignored
explicitly near the end of that file — if you add a file here in another format, check
`git check-ignore -v` before assuming it is committed.
