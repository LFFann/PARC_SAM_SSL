# PARC-SAM-SSL v4

PARC-SAM-SSL v4 is an independent SAM-based semi-supervised segmentation framework, written outside the original KnowSAM code path. It keeps a single deployable student model and uses SAM only during training as a prototype-calibrated proposal generalist.

The v4 research thesis is **prototype-calibrated, uncertainty-paced foundation supervision**: SAM is not prompted by the EMA teacher alone. Labeled-set semantic prototypes calibrate the prompt prior, SAM contributes proposal evidence, and the student learns from set-valued plus soft uncertainty-paced targets instead of brittle hard pseudo labels.

## Three Core Ideas

1. Prototype-calibrated foundation prompting: labeled semantic prototypes and the EMA teacher jointly create the prompt prior that drives SAM, breaking the teacher-SAM self-confirmation loop.
2. Uncertainty-paced set supervision: hard pseudo-label CE is restricted to reliable singleton pixels, while ambiguous regions keep soft consistency supervision whose focus shifts from uncertainty to confidence during training.
3. Class-balanced prototype-relation learning: class-prior risk control, prototype evidence, correlation consistency, and SAM-token alignment keep minority foreground classes from being erased by majority-class pseudo labels.

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

On the server, the default V100/3090 scripts expect:

- dataset: `/root/autodl-tmp/echoData/260703_data_labeled30pct`
- SAM ViT-B checkpoint: `/root/autodl-tmp/sam_vit_b_01ec64.pth`

```bash
cd /root/autodl-tmp/PARC_SAM_SSL
bash scripts/train_v100_32g_echo.sh
```

Common overrides:

```bash
MAX_ITERATIONS=18000 OUTPUT_DIR=outputs/PARC_SAM_SSL_v4_ProtoPrompt_UPSC_V100_32G_echoData CUDA_VISIBLE_DEVICES=0 bash scripts/train_v100_32g_echo.sh
RUN_DIR=outputs/PARC_SAM_SSL_v4_ProtoPrompt_UPSC_V100_32G_echoData SPLIT=test bash scripts/test_v100_32g_echo.sh
```

For an RTX 3090 24G server:

```bash
bash scripts/train_3090_24g_echo.sh
bash scripts/test_3090_24g_echo.sh
```

Run the test script after training has produced `checkpoints/best.pt` or `checkpoints/final.pt`.

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

`diagnostics.jsonl` contains class-wise candidate, pseudo-label, singleton, negative, prior, balance-weight, risk-threshold, prompt-prior, SAM-IoU, and area-guard signals. These are the first checks to inspect if a run looks background-heavy, all-foreground, or single-class collapsed after warmup.

`visualizations/` contains paper-ready qualitative panels, diagnostic heatmaps, and failure-triggered root-cause panels. See `docs/training_monitoring.md`.

## Ablation Switches

```bash
python train.py --config configs/parc_sam_ssl_3class.yaml --target-mode hard --disable-sam --disable-prototype --disable-correlation --disable-alignment --disable-foreground-guard
python train.py --config configs/parc_sam_ssl_3class.yaml --target-mode hard
python train.py --config configs/parc_sam_ssl_3class.yaml --target-mode conformal_single
python train.py --config configs/parc_sam_ssl_3class.yaml --disable-foreground-guard
```

See `docs/top_conference_revision.md` for the reviewer-facing claim, literature-backed innovation map, and minimum ablation matrix. See `docs/v2_failure_analysis_v3_anticollapse.md` for the v2 collapse diagnosis and v3 anti-collapse update. See `docs/v4_effect_first_optimization.md` for the v4 root-cause-driven mechanism revision.

Windows ablation runner:

```powershell
cd PARC_SAM_SSL
./scripts/run_ablation_matrix.ps1 -Python python -MaxIterations 10000
```

## Relation To KnowSAM

KnowSAM used a UNet/VNet/discriminator fusion path plus SAM distillation. PARC-SAM-SSL v4 changes the framework: one deployable student, prototype-calibrated SAM prompting, uncertainty-paced set-valued targets, foreground-safe class-balanced risk control, and relation-anchored proposal evidence. The original KnowSAM folder is not edited.
