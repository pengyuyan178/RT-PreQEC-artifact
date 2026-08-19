#!/usr/bin/env bash
# RT-PreQEC — one-command reproduction of every experiment in the paper.
#
# Runs the four evaluation regimes end-to-end on the committed traces, measures
# the component overheads, then re-renders every figure and rebuilds every table
# from the fresh runs. Total wall-clock on a single core: roughly 25 minutes
# (d7 ~1.5 min, d11 ~10 min, burst ~3 min, threshold sweep ~9 min, overhead ~1 min).
#
# Usage:
#   bash scripts/run_all_experiments.sh            # full pipeline (~25 min)
#   RUN=0 bash scripts/run_all_experiments.sh      # skip the runs; only re-render
#                                                    figures/tables from the committed
#                                                    traces in table/figure_data/
#
# All outputs land under results/runs/, results/figures/rerun/, results/tables/rerun/.
set -euo pipefail
cd "$(dirname "$0")/.."          # repository root (code/)
mkdir -p results/logs

CKPT=checkpoints/risk_lstm_v2_smoke_30.pt
FD=table/figure_data
RUN="${RUN:-1}"

log() { echo "[run_all] $*"; }

if [ "$RUN" = "1" ]; then
  log "[1/5] Main regime (d=7)  -> Fig. 5, Fig. 6, t1.png, Fig. 9   (~1.5 min)"
  python scripts/run_paper_experiment_suite.py \
      --config configs/real_stream_eval_main_ai_selected.yaml \
      --records $FD/d7_main/records.csv --split test \
      --risk-checkpoint "$CKPT" --include-ai --no-threshold-sweep \
      --out results/runs/paper_suite_d7_rtqec_ai_selected \
      2>&1 | tee results/logs/d7_main.log

  log "[2/5] Scaling regime (d=11)  -> regime summary rows   (~10 min)"
  python scripts/run_paper_experiment_suite.py \
      --config configs/real_stream_eval_scaling.yaml \
      --records $FD/d11_scaling/records.csv --split test \
      --risk-checkpoint "$CKPT" --include-ai --no-threshold-sweep \
      --out results/runs/paper_suite_d11_rtqec_ai \
      2>&1 | tee results/logs/d11_scaling.log

  log "[3/5] Burst regime (d=7, 1w + 2w)  -> Fig. 7 + burst-capacity table   (~3 min)"
  python scripts/run_paper_experiment_suite.py \
      --config configs/real_stream_eval_burst.yaml \
      --records $FD/burst/records.csv --split test \
      --risk-checkpoint "$CKPT" --include-ai --no-threshold-sweep \
      --out results/runs/paper_suite_burst_rtqec_ai \
      2>&1 | tee results/logs/burst.log
  python scripts/run_paper_experiment_suite.py \
      --config configs/real_stream_eval_burst_2w.yaml \
      --records $FD/burst_2w/records.csv --split test \
      --risk-checkpoint "$CKPT" --include-ai --no-threshold-sweep \
      --out results/runs/paper_suite_burst_2w_rtqec_ai \
      2>&1 | tee results/logs/burst_2w.log

  log "[4/5] Decoupled threshold sweep (d=7 val)  -> Fig. 8   (~9 min)"
  python scripts/run_paper_experiment_suite.py \
      --config configs/real_stream_eval_main_ai_selected.yaml \
      --split test --threshold-split val \
      --risk-checkpoint "$CKPT" --include-ai --threshold-sweep \
      --predecode-risk-thresholds 0.35 --confidence-thresholds 0.50 \
      --out results/runs/paper_suite_d7_rtqec_ai_decoupled \
      2>&1 | tee results/logs/threshold_sweep.log

  log "[5/5] Component-overhead microbenchmark  -> tab:overhead   (~1 min)"
  python scripts/measure_rtss_overheads.py \
      --config configs/real_stream_eval_main.yaml \
      --num-shots 2000 --warmup-shots 100 \
      --out results/runs/wcet_overheads_d7 \
      2>&1 | tee results/logs/overheads.log

  # Point the figure/table re-render at the fresh runs.
  MAIN=results/runs/paper_suite_d7_rtqec_ai_selected/main
  D11=results/runs/paper_suite_d11_rtqec_ai/main
  BURST=results/runs/paper_suite_burst_rtqec_ai/main
  BURST2W=results/runs/paper_suite_burst_2w_rtqec_ai/main
  SWEEP=results/runs/paper_suite_d7_rtqec_ai_decoupled/threshold_sweep.csv
else
  log "RUN=0: re-rendering figures/tables from the committed traces in $FD"
  MAIN=$FD/d7_main
  D11=$FD/d11_scaling
  BURST=$FD/burst
  BURST2W=$FD/burst_2w
  SWEEP=$FD/threshold_sweep/threshold_sweep.csv
fi

log "Rendering figures (PNG + PDF)  -> Fig. 5, 6, 7, 8"
python scripts/make_rtss_plots.py \
    --run-dir "$MAIN" --threshold-sweep "$SWEEP" --burst-run-dir "$BURST" \
    --format both --out results/figures/rerun

log "Rebuilding tables  -> t1.png & Fig. 9 data, safety/AI/burst-capacity tables"
python scripts/export_rtss_tables.py \
    --run-dir "$MAIN" --burst-1w-run-dir "$BURST" --burst-2w-run-dir "$BURST2W" \
    --out results/tables/rerun

log "Rebuilding regime summary + front-end contract tables"
python scripts/summarize_rtss_results.py \
    --d7 "$MAIN/summary_metrics.csv" --d11 "$D11/summary_metrics.csv" \
    --burst "$BURST/summary_metrics.csv" --out-dir results/tables/rerun

log "Done. Figures: results/figures/rerun/   Tables: results/tables/rerun/   Logs: results/logs/"
