# RT-PreQEC Models

RT-PreQEC is not a full neural decoder. The AI paths are fallback-safe workload shapers: they score scheduler risk or select one local DEM candidate, and every output can be rejected by confidence/risk thresholds, validation, or backend fallback.

## RiskRuntimeModel

`RiskRuntimeModel` implements the Risk/Runtime Profiler for the risk-aware lag-bounded scheduler.

```text
features_t + causal history
  -> FeatureProjectionEncoder
  -> CausalHistoryEncoder
  -> Fusion
  -> risk / hard-runtime / runtime-regression / confidence heads
  -> scheduler
```

The model predicts:

- `risk`: fast decoder may be wrong or unsafe for this job.
- `hard_runtime`: accurate decoder may fall into the runtime tail.
- `runtime_pred`: `log1p(accurate_runtime_us)` service-time estimate.
- `confidence`: whether the scheduler should trust the model output. This is not the same as low risk.

Temporal encoding is optional:

- `none` / MLP is the non-temporal baseline over current shot/job features.
- `GRU` / `LSTM` models recent syndrome/job history such as burst, drift, repeated high-density syndrome, and backlog growth.
- `TCN` uses fixed causal convolutions, which are attractive when predictable timing matters.

All temporal modes are causal and unidirectional for online use; bidirectional history would look into the future and is rejected.

Temporal training must use `stream_block` or `episode` split policy. `random`
split is allowed only for non-temporal ablations unless explicitly overridden.
`HistoryRiskDataset` pads at split starts and never reads history from another
split, so validation/test history cannot include train features. Normalization
is computed from train features only and stored in the checkpoint.

The hard-runtime head is trained only when the dataset metadata marks
`hard_runtime_label_valid=true`. Valid hard-runtime labels come from
loop-per-shot decoder timing. Batch decode may still be used for accuracy, but
batch-average latency is not used as a per-shot runtime tail target.

Risk and confidence thresholds are not model constants. They are selected on
the validation split with `scripts/calibrate_risk_thresholds.py` and then
loaded by real-stream test evaluation.

## CandidatePredecoderModel

`CandidatePredecoderModel` implements the selective neural predecoder.

```text
DetectorPatch + DEM candidates
  -> DetectorPatchEncoder
  -> DEMCandidateEncoder
  -> PatchCandidateCompatibility
  -> candidate logits + abstain + confidence/risk
  -> validation
  -> residual syndrome
```

It uses DEM local candidates because the model should not invent arbitrary corrections. Candidate logits rank bounded local candidates that can be validated, and an explicit abstain head lets the system fall back when the patch is ambiguous, high risk, observable-touching, or poorly covered by candidates.

## Heads

- `RiskHead`: fast-decoder error/safety risk for scheduling.
- `HardRuntimeHead`: accurate-decoder tail-runtime classification.
- `RuntimeRegressionHead`: accurate-decoder service-time regression.
- `ConfidenceHead`: trust in model outputs.
- `CandidateHead`: masked compatibility over DEM local candidates.
- `AbstainHead`: explicit refusal to predecode.
