# PARC-SAM-SSL v2

PARC-SAM-SSL v2 is an independent SAM-based semi-supervised segmentation framework, written outside the original KnowSAM code path. It keeps a single deployable student model and uses SAM only during training as a calibrated proposal generalist.

The v2 research thesis is **calibrated set-valued foundation supervision**: SAM and the EMA teacher are not treated as hard annotators. They form risk-calibrated candidate label sets, while hard pseudo-label CE is used only when the candidate set is a singleton.

## Three Core Ideas

1. Calibrated set-valued foundation supervision: the EMA teacher, SAM proposals, and prototype evidence produce a candidate label set instead of forcing one brittle pseudo label.
2. Foreground-safe class-balanced risk control: class-wise thresholds, foreground caps, prior-aware weights, and a conservative foreground guard prevent minority foreground collapse.
3. Relation-anchored proposal evidence: unlabeled features update semantic prototypes and are regularized with correlation consistency; real SAM embeddings can additionally align student regions to SAM token regions.

## Data Layout

```text
SampleData/<dataset_name>/
  labeled/image
  labeled/mask
  unlabeled/image
  val/image
  val/mask
  test/image
  test/mask
```

Masks use class indices: `0` background, `1..K-1` foreground classes.

## Smoke Test

```bash
cd PARC_SAM_SSL
python tools/make_synthetic_dataset.py --root synthetic_data/smoke
python train.py --config configs/smoke_cpu.yaml
python evaluate.py --config configs/smoke_cpu.yaml --checkpoint outputs/smoke_cpu/checkpoints/best.pt --split test --device cpu
```

The smoke config explicitly uses a surrogate SAM proposal path so the training loop can be tested without a SAM checkpoint.

## Real Training

Place a SAM checkpoint where `configs/parc_sam_ssl_3class.yaml` points, or override it:

```bash
cd PARC_SAM_SSL
bash scripts/train_v100.sh
```

Common overrides:

```bash
MAX_ITERATIONS=10000 OUTPUT_DIR=outputs/PARC_SAM_SSL_3Class CUDA_VISIBLE_DEVICES=0 bash scripts/train_v100.sh
CHECKPOINT=outputs/PARC_SAM_SSL_3Class/checkpoints/best.pt bash scripts/test_v100.sh
```

## Outputs

```text
outputs/<run>/
  train.log
  metrics.jsonl
  diagnostics.jsonl
  health.jsonl
  validation.jsonl
  resolved_config.json
  checkpoints/best.pt
  checkpoints/final.pt
  predictions/
  visualizations/
```

`diagnostics.jsonl` contains class-wise candidate, pseudo-label, singleton, negative, prior, balance-weight, and risk-threshold signals. These are the first checks to inspect if a run looks background-heavy or collapses after warmup.

`visualizations/` contains paper-ready qualitative panels, diagnostic heatmaps, and failure-triggered root-cause panels. See `docs/training_monitoring.md`.

## Ablation Switches

```bash
python train.py --config configs/parc_sam_ssl_3class.yaml --target-mode hard --disable-sam --disable-prototype --disable-correlation --disable-alignment --disable-foreground-guard
python train.py --config configs/parc_sam_ssl_3class.yaml --target-mode hard
python train.py --config configs/parc_sam_ssl_3class.yaml --target-mode conformal_single
python train.py --config configs/parc_sam_ssl_3class.yaml --disable-foreground-guard
```

See `docs/top_conference_revision.md` for the reviewer-facing claim, literature-backed innovation map, and minimum ablation matrix.

Windows ablation runner:

```powershell
cd PARC_SAM_SSL
./scripts/run_ablation_matrix.ps1 -Python python -MaxIterations 10000
```

## Relation To KnowSAM

KnowSAM used a UNet/VNet/discriminator fusion path plus SAM distillation. PARC-SAM-SSL v2 changes the framework: one deployable student, training-only SAM proposal sets, calibrated set-valued targets, foreground-safe risk control, and relation-anchored proposal evidence. The original KnowSAM folder is not edited.
