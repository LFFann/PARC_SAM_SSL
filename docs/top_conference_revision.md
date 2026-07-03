# PARC-SAM-SSL v2: Top-Conference Revision Note

Search/update date: 2026-07-02.

## Post-Review Diagnosis

The original PARC-SAM-SSL direction was promising but still looked like a combination of known components: ConformalSAM-style calibration, SemiSAM-style SAM assistance, and CorrMatch/AllSpark-style relation learning. The v2 revision narrows the method into one central thesis:

> Semi-supervised SAM supervision should be treated as calibrated set-valued foundation supervision, not as hard pseudo-label annotation.

This reframing changes the training target itself. The method asks the student to put probability mass on a calibrated candidate set, uses hard pseudo-label CE only where the candidate set is a singleton, and records class-wise participation so foreground collapse cannot hide behind average Dice.

## Three Core Innovations

### 1. Calibrated Set-Valued Foundation Supervision

Problem: EMA teachers and SAM-like foundation models can both be overconfident on target-domain pixels. Turning their outputs into a single pseudo label creates irreversible early errors.

Mechanism:

- Maintain class-wise conformal-style risk thresholds from labeled batches.
- Fuse EMA teacher, SAM proposal probabilities, and prototype evidence into candidate label sets.
- Apply hard pseudo-label loss only to singleton candidate pixels.
- Apply proposal-set loss to ambiguous pixels and safe negative loss to classes with consistently low evidence.

References:

- ConformalSAM, ICCV 2025: calibrates foundation-model masks on target data before using them for SSSS.
- When Confidence Fails, ICCV 2025: shows confidence-only pseudo-label selection is unreliable under overconfidence.
- UniMatch, CVPR 2023: provides the weak-to-strong SSL skeleton that v2 generalizes.

Code:

- `parc_sam/ssl/conformal.py`
- `parc_sam/ssl/proposal_bank.py`
- `parc_sam/losses.py`

### 2. Foreground-Safe Class-Balanced Risk Control

Problem: Previous local KnowSAM-family attempts showed foreground starvation and late collapse. A method can improve background confidence while silently suppressing minority classes.

Mechanism:

- Clamp foreground conformal thresholds separately so foreground candidate sets cannot saturate into meaningless ambiguity.
- Track labeled pixel priors and compute class-balance weights for pseudo supervision.
- Add a conservative foreground participation guard: when foreground candidate participation falls below a floor, rescue only high-evidence foreground pixels with a small minimum weight.
- Log class-wise candidate ratio, singleton ratio, pseudo-label ratio, negative ratio, prior, balance weight, and risk threshold.

References:

- DiffMatch, ICLR 2025: frames class imbalance and Matthew-effect pseudo labels as a central SSSS failure mode.
- S4Former, CVPR 2024: supports negative-class regularization and transformer/global-context regularization for SSSS.
- ConformalSAM, ICCV 2025: motivates class-conditioned calibration for foundation masks.

Code:

- `parc_sam/ssl/conformal.py`
- `parc_sam/ssl/proposal_bank.py`
- `parc_sam/engine/trainer.py`

### 3. Relation-Anchored Proposal Evidence

Problem: SAM proposals are class-agnostic and teacher probabilities are local. A student can match pixel labels while failing to preserve same-class region structure.

Mechanism:

- Keep a semantic prototype memory updated from labeled pixels and reliable unlabeled pixels.
- Use prototype logits as a third evidence source before candidate-set construction.
- Align student feature correlation with proposal-probability correlation on unlabeled images.
- Optionally align student region features with real SAM token-region embeddings during training only.

References:

- CorrMatch, CVPR 2024: shows correlation maps capture same-class grouping and shape cues.
- RankMatch, CVPR 2024: supports relation-level consistency beyond pixel-wise CE.
- AllSpark, CVPR 2024: supports architecture-level labeled/unlabeled feature interaction with semantic memory.
- SSR-SAM, AAAI 2026: supports SAM-style prompt/proposal consistency for semi-supervised segmentation.

Code:

- `parc_sam/ssl/semantic_memory.py`
- `parc_sam/losses.py`
- `parc_sam/engine/trainer.py`

## Why This Has A Credible Path To Beat KnowSAM

Known local baseline: the stable KnowSAM multiclass 30 percent labeled run has `avg_dice=0.7601`, `avg_iou=0.6217`, `avg_hd95=14.1232`, `class_1_avg_dice=0.7218`, and `class_2_avg_dice=0.7985` in the existing baseline summary.

The v2 framework is designed against the observed failure modes of prior variants:

- Hard pseudo-label damage is reduced because ambiguous pixels receive set-valued loss rather than forced CE.
- Foreground starvation is monitored and countered by foreground candidate participation, class-balanced weights, and rescue floors.
- Late collapse becomes visible through `diagnostics.jsonl`, especially class-wise candidate/pseudo ratios and risk thresholds.
- SAM is training-only, so the final inference path stays a single compact student rather than a deployment-heavy SAM pipeline.
- The strongest local baseline has no explicit calibrated set-valued target or class-wise proposal diagnostics, so v2 has a real mechanism-level advantage rather than only a tuning advantage.

Confidence condition: v2 should be considered a strong baseline-beating candidate if early training diagnostics satisfy all four checks:

- `class_1_candidate_ratio` and `class_2_candidate_ratio` remain non-zero after warmup.
- `background_only_ratio` does not monotonically climb toward 1.0.
- foreground `risk_q_class_*` values do not stay at the maximum cap for most of training.
- validation best checkpoint improves before the final checkpoint, without a large best-to-final collapse.

If these checks fail, the idea is not dead; the next repair should adjust the foreground participation floor, foreground quantile cap, SAM prompt threshold, or prototype weight before changing the whole framework again.

## Required Ablation Matrix

Main comparisons:

- KnowSAM stable baseline.
- UniMatch-style same-student weak-to-strong baseline: `--target-mode hard --disable-sam --disable-prototype --disable-correlation --disable-alignment --disable-foreground-guard`.
- SAM hard pseudo-label baseline: `--target-mode hard --disable-prototype --disable-correlation --disable-alignment --disable-foreground-guard`.
- Conformal single-label baseline: `--target-mode conformal_single`.
- Full PARC-SAM-SSL v2: default config.

Mechanism ablations:

- no SAM: `--disable-sam`.
- no prototype memory: `--disable-prototype`.
- no correlation: `--disable-correlation`.
- no foreground guard: `--disable-foreground-guard`.
- hard pseudo-label replacement: `--target-mode hard`.

Metrics:

- Dice, IoU, HD95, per-class Dice/IoU.
- best checkpoint and final checkpoint.
- class-wise candidate, pseudo, singleton, negative, prior, balance, and risk-threshold diagnostics.

## Reviewer-Facing Claim Boundary

Acceptable claim:

> We introduce calibrated set-valued foundation supervision for SAM-based semi-supervised segmentation, with foreground-safe class-balanced risk control and relation-anchored proposal evidence. The method converts foundation-model outputs into abstaining candidate sets rather than brittle hard labels, improving robustness under label scarcity and class imbalance.

Claims to avoid before results:

- Guaranteed SOTA across all SSSS benchmarks.
- SAM is always correct or always improves pseudo labels.
- Conformal thresholds alone solve domain shift.
- The method is training-free or deployment-free; only SAM is removed at inference.
