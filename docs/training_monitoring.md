# Training Metrics And Visualization

PARC-SAM-SSL records three layers of training evidence:

```text
outputs/<run>/
  metrics.jsonl
  diagnostics.jsonl
  health.jsonl
  validation.jsonl
  visualizations/
    manifest.jsonl
    paper/
    diagnostic/
    failure/
```

## 1. Paper-Ready Visuals

Path:

```text
visualizations/paper/
```

These PNGs are contact sheets for qualitative figures. Each unlabeled sample contains:

- weak image
- strong image
- EMA teacher prediction
- SAM proposal
- candidate pseudo target
- student prediction
- candidate-set size map
- pseudo-weight map

Each labeled sample contains:

- image
- ground truth overlay
- student prediction overlay
- error map

Use these figures to show the paper claim: the method does not blindly copy SAM; it converts teacher/SAM evidence into calibrated candidate sets before supervising the student.

## 2. Framework-Effect Diagnostics

Path:

```text
diagnostics.jsonl
visualizations/diagnostic/
```

Key scalar fields:

- `sam_used`: must be `true` or `1.0` for real SAM-assisted training.
- `foreground_candidate_ratio`: should stay non-zero.
- `background_only_ratio`: should not approach `1.0`.
- `proposal_singleton_ratio`: very high values mean the set-valued target has collapsed into hard pseudo labels.
- `teacher_entropy_mean` / `student_entropy_mean`: near-zero entropy too early indicates overconfidence.
- `class_1_candidate_ratio`, `class_2_candidate_ratio`: foreground classes must participate.
- `risk_q_class_1`, `risk_q_class_2`: long saturation at the foreground cap means calibration is failing.

Diagnostic PNGs show:

- teacher entropy
- student entropy
- SAM confidence
- soft-target confidence
- candidate-set size
- negative-class count
- reliable-pixel map
- singleton-pixel map

## 3. Root-Cause Failure Localization

Path:

```text
health.jsonl
visualizations/failure/
```

`health.jsonl` records automatic flags:

- `sam_inactive`: SAM is disabled or not producing proposal targets.
- `foreground_candidate_collapse`: candidate sets almost never include foreground.
- `student_foreground_collapse`: student predictions are almost all background.
- `background_only_dominance`: candidate supervision is background-only.
- `set_supervision_degenerate`: candidate sets have become almost all singleton hard labels.
- `student_overconfident`: student entropy is too low.
- `risk_q_class_*_saturated`: foreground calibration threshold is stuck at its cap.

Severity:

- `ok`: no warning.
- `warn`: at least one failure signal; inspect diagnostics.
- `critical`: three or more signals; likely mechanism failure, not just hyperparameter noise.

## Recommended Server Checks

During V100 training:

```bash
tail -n 5 outputs/PARC_SAM_SSL_v2_V100_32G_echoData/metrics.jsonl
tail -n 5 outputs/PARC_SAM_SSL_v2_V100_32G_echoData/diagnostics.jsonl
tail -n 5 outputs/PARC_SAM_SSL_v2_V100_32G_echoData/health.jsonl
ls outputs/PARC_SAM_SSL_v2_V100_32G_echoData/visualizations/paper | tail
```

If the final test is below KnowSAM, first inspect:

1. Did `sam_used` stay active?
2. Did `foreground_candidate_ratio` collapse?
3. Did `risk_q_class_1` or `risk_q_class_2` saturate?
4. Did `proposal_singleton_ratio` become almost `1.0`?
5. Did the best validation checkpoint occur much earlier than the final checkpoint?

This separates a hyperparameter issue from a mechanism issue.
