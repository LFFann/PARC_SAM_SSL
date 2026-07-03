# V100 32G Server Runbook

Target server paths:

- Dataset root: `/root/autodl-tmp/echoData/260703_data_labeled30pct`
- SAM ViT-B checkpoint: `/root/autodl-tmp/sam_vit_b_01ec64.pth`

Expected dataset layout:

```text
/root/autodl-tmp/echoData/260703_data_labeled30pct/
  labeled/image
  labeled/mask
  unlabeled/image
  val/image
  val/mask
  test/image
  test/mask
```

## Recommended V100 32G Config

Use:

```text
configs/parc_sam_ssl_v100_32g_echo.yaml
```

Main changes from the generic config:

- `base_channels=48`, `feature_dim=160`: stronger student while staying V100-safe.
- `batch_size_labeled=6`, `batch_size_unlabeled=4`: stable with real SAM ViT-B 1024 encoding.
- `lr=8e-4`, `ema_decay=0.996`: smoother teacher for real-data training.
- `proposal_set=0.45`, `pseudo=0.9`: emphasize set-valued supervision and reduce hard pseudo-label dominance.
- `alpha=0.12`, `max_foreground_quantile=0.82`, `min_foreground_participation=0.025`: foreground-safe calibration against background collapse.

## Train

```bash
cd /root/autodl-tmp/PARC_SAM_SSL
bash scripts/train_v100_32g_echo.sh 2>&1 | tee train_v100_32g_echo.log
```

Equivalent explicit command:

```bash
cd /root/autodl-tmp/PARC_SAM_SSL
python train.py \
  --config configs/parc_sam_ssl_v100_32g_echo.yaml \
  --device cuda \
  --max-iterations 12000 \
  --data-root /root/autodl-tmp/echoData/260703_data_labeled30pct \
  --sam-checkpoint /root/autodl-tmp/sam_vit_b_01ec64.pth \
  --output-dir outputs/PARC_SAM_SSL_v2_V100_32G_echoData
```

## Test

```bash
cd /root/autodl-tmp/PARC_SAM_SSL
bash scripts/test_v100_32g_echo.sh 2>&1 | tee test_v100_32g_echo.log
```

Equivalent explicit command:

```bash
cd /root/autodl-tmp/PARC_SAM_SSL
python evaluate.py \
  --config configs/parc_sam_ssl_v100_32g_echo.yaml \
  --checkpoint outputs/PARC_SAM_SSL_v2_V100_32G_echoData/checkpoints/best.pt \
  --split test \
  --device cuda \
  --save-dir outputs/PARC_SAM_SSL_v2_V100_32G_echoData/prediction_test
```

## If V100 Runs Out Of Memory

Start with a smaller output directory and override the config manually:

```bash
python train.py \
  --config configs/parc_sam_ssl_v100_32g_echo.yaml \
  --device cuda \
  --max-iterations 12000 \
  --data-root /root/autodl-tmp/echoData/260703_data_labeled30pct \
  --sam-checkpoint /root/autodl-tmp/sam_vit_b_01ec64.pth \
  --output-dir outputs/PARC_SAM_SSL_v2_V100_32G_echoData_bs2
```

Then edit these two fields in the config:

```yaml
train:
  batch_size_labeled: 4
  batch_size_unlabeled: 2
```

Keep `sam.image_size=1024` for the first real comparison. Lower it to `768` only if the above batch reduction still OOMs, because lowering SAM resolution changes the proposal quality.

## First Diagnostics To Check

After several hundred iterations, inspect:

```bash
tail -n 5 outputs/PARC_SAM_SSL_v2_V100_32G_echoData/metrics.jsonl
tail -n 5 outputs/PARC_SAM_SSL_v2_V100_32G_echoData/diagnostics.jsonl
tail -n 5 outputs/PARC_SAM_SSL_v2_V100_32G_echoData/health.jsonl
ls outputs/PARC_SAM_SSL_v2_V100_32G_echoData/visualizations/paper | tail
```

Healthy signs:

- `sam_used` is `1.0`.
- `foreground_candidate_ratio` is non-zero and not collapsing.
- `background_only_ratio` is not monotonically approaching `1.0`.
- foreground `risk_q_class_1` / `risk_q_class_2` are not stuck at the max cap for most of training.
