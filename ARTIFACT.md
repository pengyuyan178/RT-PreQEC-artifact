# RTSS 2026 Artifact Evaluation — RT-PreQEC

Artifact instructions for **“RT-PreQEC: Lag-Bounded Real-Time Scheduling for Quantum
Error Correction”** (RTSS 2026). Everything below runs from a plain `git clone`; no VM
or Docker image is required.

---

## 1. Artifact Overview

This artifact is the **reproduction package** for the paper. It contains the
RT-PreQEC runtime (a lag-bounded real-time scheduler for QEC decoding), the full
evaluation harness, every committed run input (syndrome traces), the trained models,
the published figures, and the CSV data behind every figure and table.

RT-PreQEC dispatches a continuous syndrome stream across a fast lookup decoder and an
accurate PyMatching decoder under deadline, backlog, and Pauli-frame-lag constraints,
using a front-end certificate and a learned LSTM risk estimate to decide what may
commit through the fast path. The artifact lets a reviewer (a) verify every reported
number against the committed traces in minutes, and (b) re-run any or all experiments
end-to-end.

- **Paper:** *RT-PreQEC: Lag-Bounded Real-Time Scheduling for Quantum Error
  Correction*, RTSS 2026.
- **Scope:** all ten figures (Figure 1–9 and `t1.png`) and all three tables
  (`tab:regimes`, `tab:modes`, `tab:overhead`) in the current draft. Figures 1–4 are
  hand-drawn schematics; Figures 5–9 and `t1.png` are data artifacts reproduced here.

## 2. Artifact Download

Clone the repository:

```bash
git clone https://github.com/pengyuyan178/RT-PreQEC-artifact.git
cd RT-PreQEC-artifact
```

- **Evaluated revision:** commit `756bb3c` (branch `main`). To pin exactly:
  `git checkout 756bb3c`.
- **Integrity check (optional):** after checkout, confirm the tree matches the
  evaluated commit with `git rev-parse HEAD` → `756bb3c…`, and that `git status` is
  clean. A release tarball with a SHA-256 checksum can be attached on the AE page if
  required by the chairs.

The repository is self-contained: all traces, trained checkpoints, and the published
figures/tables are committed, so no large external dataset download is needed.

## 3. Hardware Requirements

| Resource | Requirement | Notes |
|---|---|---|
| **CPU** | Any modern multi-core x86-64 | Workload is single-threaded by design |
| **RAM** | ≥ 8 GB | Full pipeline peaks at a few GB |
| **GPU** | **Not required** | All results are CPU-only |
| **Disk** | ≈ 3 GB free | ~2 GB repo + ~1 GB for a full re-run |

## 4. Software Environment

- **Host OS:** Windows 10+, Linux, or macOS (64-bit). No guest OS / VM / container is
  used.
- **Python:** ≥ 3.10 (3.10 recommended, to match the published runs).
- **Dependencies:** `numpy, scipy, pandas, pyyaml, tqdm, typer, rich, matplotlib,
  scikit-learn, torch, stim, sinter, pymatching` (plus `pytest` for the tests).
- **Install** (either):
  - Conda (recommended): `conda env create -f environment.yml && conda activate rt-preqec && python -m pip install -e .[dev]`
  - pip, into any Python ≥ 3.10: `python -m pip install -e .[dev]`

The dependency set used for the published numbers is recorded in each run's
`suite_manifest.json`: Python 3.10.20, numpy 2.2.6, pandas 2.3.3, torch 2.10.0, stim
1.16.0, pymatching 2.4.0, matplotlib 3.10.9. Install these exact versions if you need
the closest match on timing-derived metrics.

## 5. Getting Started

```bash
git clone https://github.com/pengyuyan178/RT-PreQEC-artifact.git
cd RT-PreQEC-artifact
conda env create -f environment.yml && conda activate rt-preqec   # or: pip install -e .[dev]
python -m pip install -e .[dev]
```

No login, license, or external service is needed. Verify the install:

```bash
python -m pytest tests -q        # expected: 164 passed, 1 failed (a known stale fixture, see §11)
```

## 6. Quick Test

One command replays every committed trace through the queue simulator and diffs each
reported metric against the published summaries:

```bash
python scripts/verify_paper_numbers.py
```

- **Expected runtime:** ~1 minute.
- **Expected output:** `364 metric checks reproduce, 0 deviate (rtol=1e-06)`.

This is the fastest end-to-end sanity check: if it prints `0 deviate`, the environment
is correct and the committed results are intact.

## 7. Reproducing Paper Results

Each figure/table lists: the command, expected runtime, where the output lands, and
what it should match. Commands below read the **committed traces** in
`table/figure_data/`, so they reproduce the paper exactly and quickly. (Re-running the
experiments themselves is §8.)

### Figures 5, 6, 7, 8 — data figures (PNG + PDF)

One command produces all four:

```bash
python scripts/make_rtss_plots.py \
    --run-dir table/figure_data/d7_main \
    --threshold-sweep table/figure_data/threshold_sweep/threshold_sweep.csv \
    --burst-run-dir table/figure_data/burst \
    --format both --out results/figures/rerun
```

- **Expected runtime:** ~30 s.
- **Output:** `results/figures/rerun/*.png` and `*.pdf`.

| Output file | Paper figure | Matches committed |
|---|---|---|
| `main_pareto_frontier.pdf` | **Figure 5** (`fig:pareto`) | `figure/figure5.pdf` |
| `response_time_cdf_by_mode.pdf` | **Figure 6** (`fig:cdf`) | `figure/figure6.pdf` |
| `burst_lag_backlog_trace.pdf` | **Figure 7** (`fig:burst`) | `figure/figure7.pdf` |
| `threshold_sweep_lag_frontier.pdf` | **Figure 8** (`fig:sweep`) | `figure/figure8.pdf` |

- **Expected result:** the PDFs carry the same data as the committed `figure/*.pdf`
  (the CDF and burst PDFs may differ by a sub-point canvas width from a newer
  matplotlib — the underlying numbers are identical).

### Figure 9 and `t1.png` — main results & ablation (from CSVs)

These two are figures rendered from CSVs. Rebuild the CSVs:

```bash
python scripts/export_rtss_tables.py --run-dir table/figure_data/d7_main \
    --out results/tables/rerun
```

- **Expected runtime:** ~5 s.
- **Output:** `results/tables/rerun/rtss_main_table.csv` (→ `t1.png`, `fig:main`) and
  `rtss_ablation_delta_vs_rt_qec.csv` (→ **Figure 9**, `fig:ablation-delta`), plus the
  ablation / safety / AI-risk tables.
- **Expected result:** the CSVs are **byte-identical** to `table/rtss_main_table.csv`
  and `table/rtss_ablation_delta_vs_rt_qec.csv`.

### `tab:overhead` — per-component overhead

This is the paper's only computed LaTeX table (an isolated microbenchmark):

```bash
python scripts/measure_rtss_overheads.py \
    --config configs/real_stream_eval_main.yaml \
    --num-shots 2000 --warmup-shots 100 \
    --out results/runs/wcet_overheads_d7
```

- **Expected runtime:** ~1 min.
- **Output:** `results/runs/wcet_overheads_d7/overhead_summary.csv` and
  `measurement_protocol.json`.
- **Expected result:** means reproduce closely against `table/rtss_overhead_table.csv`
  (e.g. scheduler ≈ 1.2–1.5 µs, accurate backend ≈ 8.7–8.9 µs); the p99/max tails shift
  with the machine, as expected for empirical timing.

### `tab:regimes` and `tab:modes` — configuration tables

These two tables are hand-written configuration summaries with no computed data behind
them. They correspond to the config files in `configs/` (regime table) and to the
mode-key mapping in `scripts/export_rtss_tables.py:MODE_LABELS` (modes table); both are
reproduced in the README appendix for inspection.

### Multi-regime / supporting tables (§6 discussion)

```bash
python scripts/summarize_rtss_results.py \
    --d7    table/figure_data/d7_main/summary_metrics.csv \
    --d11   table/figure_data/d11_scaling/summary_metrics.csv \
    --burst table/figure_data/burst/summary_metrics.csv \
    --out-dir results/tables/rerun

python scripts/export_rtss_tables.py \
    --run-dir table/figure_data/d7_main \
    --burst-1w-run-dir table/figure_data/burst \
    --burst-2w-run-dir table/figure_data/burst_2w \
    --out results/tables/rerun
```

- **Expected result:** `results/tables/rerun/regime_summary_table.csv`,
  `frontend_contract_table.csv`, and `rtss_burst_capacity_table.csv` match the committed
  `table/` copies.

## 8. Full Evaluation

To re-run **every experiment end-to-end** (front-end, risk model, routing, queueing,
metrics) on the committed traces, then re-render all figures and rebuild all tables
from the fresh runs:

```bash
bash scripts/run_all_experiments.sh
```

- **Expected runtime:** ≈ **25 minutes** on a single core. Per-step logs land in
  `results/logs/`; fresh runs in `results/runs/`; re-rendered figures/tables in
  `results/figures/rerun/` and `results/tables/rerun/`.
- It runs, in order: main d=7 (~1.5 min), scaling d=11 (~10 min), burst 1-worker +
  2-worker (~3 min), decoupled threshold sweep (~9 min), and the overhead microbenchmark
  (~1 min), then re-renders everything.

**Quick vs. full.** The *quick* path (§6–§7) replays the committed traces through the
simulator/plotters — minutes, and the numbers match exactly. The *full* path re-runs
the whole pipeline against those same traces (paired-shot protocol), exercising every
stage of the runtime, and still lands on the published numbers. Use
`RUN=0 bash scripts/run_all_experiments.sh` to do only the re-render step.

Individual regimes can be run one at a time; see `README.md` § Experiments for the
per-regime `run_paper_experiment_suite.py` commands.

## 9. Artifact Structure

```text
.
├── ARTIFACT.md / README.md    # this document / the reproduction guide
├── environment.yml            # conda environment (full stack)
├── pyproject.toml             # pip package + dev extras
├── configs/                   # all experiment configs (regimes, models, policies)
├── src/rt_preqec/             # the runtime, scheduler, front-end, evaluators, models
├── scripts/                   # reproduction entry points
│   └── run_all_experiments.sh # one-command full evaluation (§8)
├── tests/                     # pytest suite (164 passed / 1 known failure)
├── figure/                    # the paper's ten figures as published (9 PDFs + t1.png)
├── table/                     # CSV data behind every figure and table
│   └── figure_data/           #   committed per-regime traces that regenerate Fig. 5-8
├── checkpoints/               # trained models (LSTM risk profiler, predecoder)
├── data/processed/            # committed 300k-sample predecoder dataset
├── docs/assets/               # PNG renderings of the paper figures
└── results/                   # runs/ figures/ tables/ logs/ — reproduction outputs
```

## 10. Expected Results

**How to judge success**

- `python scripts/verify_paper_numbers.py` prints `364 metric checks reproduce,
  0 deviate (rtol=1e-06)`.
- `export_rtss_tables.py` produces CSVs **byte-identical** to `table/rtss_*.csv`.
- The re-rendered figure PDFs match the committed `figure/*.pdf` in data (canvas width
  may differ by a sub-point under a newer matplotlib).
- A full re-run's `results/runs/<regime>/main/summary_metrics.csv` matches
  `table/figure_data/<regime>/summary_metrics.csv` on every metric.

**Tolerance.** Trace-derived metrics (logical error rate, deadline-miss ratio, lag
violation, boundary-commit success, fast-selection ratio) are **exactly reproducible**
(rtol = 1e-6) because the traces are committed and all modes decode the same shots.
Timing-derived quantities (the `tab:overhead` microbenchmark, and absolute p99/p999
response times) are empirical and shift with the CPU and load — expect the *means* to
match closely and the *tails* to vary; this is expected and is stated in the paper.

## 11. Troubleshooting

- **`ModuleNotFoundError: torch` (or stim/pymatching).** The full stack is required even
  for the quick path — the simulator imports `torch`. Complete the install in §4/§5;
  there is no reduced install.
- **`pytest` shows 1 failure.** Expected and unrelated to the runtime:
  `tests/test_rtss_artifacts.py::test_export_rtss_tables_writes_expected_csvs` uses a
  stale fixture whose mode list omits `rt_qec_ai`. `164 passed, 1 failed` is the correct
  baseline.
- **Do not pass `--calibration` to the `run_paper_experiment_suite.py` commands.** The
  published runs read routing thresholds from the config; the calibration sidecar
  overrides the scheduler threshold (to 0.25) and shifts the metrics.
- **Timing numbers differ from the paper.** Expected — they are empirical and depend on
  the CPU. Trace-derived safety metrics still match exactly (see §10).
- **`metrics.json` shows `real_qec=false`.** `stim`/`sinter`/`pymatching` is missing, so
  the harness ran its toy fallback. Reinstall the full stack; `real_qec=true` is required
  for any paper claim.
- **Windows line endings.** `scripts/run_all_experiments.sh` is run with `bash` (Git
  Bash / WSL). If your checkout converted it to CRLF, run `bash scripts/run_all_experiments.sh`
  explicitly rather than `./scripts/run_all_experiments.sh`.

## 12. Contact

For questions about the artifact, please contact the corresponding author via the RTSS
2026 AE channel (or open an issue on the repository:
<https://github.com/pengyuyan178/RT-PreQEC-artifact/issues>).
