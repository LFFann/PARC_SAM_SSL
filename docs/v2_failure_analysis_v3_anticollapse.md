# V2 Failure Analysis and V3 Anti-Collapse Update

This note summarizes the run at `outputs/PARC_SAM_SSL_v2_V100_32G_echoData` and the code changes made for the next V100 run.

## Observed Failure

The v2 run collapsed into an almost full-image class-1 prediction.

- Best validation foreground Dice was only `0.014071` at iteration 400; final logged validation Dice was `0.005660` at iteration 2200.
- By iteration 2350, `student_foreground_ratio=0.982628`, `teacher_foreground_ratio=0.988483`, and `pseudo_foreground_ratio=0.982540`.
- Class distribution at iteration 2350 was degenerate: `student_class_1_ratio=0.982628`, `student_class_2_ratio=0.0`, `class_1_pseudo_ratio=0.982540`, `class_2_pseudo_ratio=0.0`.
- Health logs repeatedly flagged `risk_q_class_1_saturated`, `risk_q_class_2_saturated`, and `student_overconfident`.
- Visualizations showed nearly all pixels marked reliable, high-confidence, and singleton, so the low training loss was mainly fitting wrong high-confidence pseudo labels.

## Root Cause

1. Early class-wise conformal thresholds were too permissive. With a weak early teacher, high foreground `q` values allowed broad foreground candidate sets instead of rejecting uncertainty.
2. SAM prompts were generated from those broad teacher regions. Large prompt boxes and dense mask prompts made SAM return fan-shaped foreground proposals, confirming the teacher error.
3. Confidence was treated as reliability. The framework did not check whether a foreground pseudo label occupied an implausible area under the labeled-set class prior.
4. Unsupervised losses were active from iteration 1 with large weights, so the erroneous pseudo labels competed with supervised learning before the student stabilized.
5. Zero-weight pseudo-label CE was skipped, but the Dice part still returned a nonzero loss. This made unreliable pseudo-label suppression incomplete.

## Literature-Motivated Changes

- ConformalSAM motivates target-domain calibration of foundation-model masks and staged reliance on foundation supervision instead of trusting raw SAM masks.
- SemiSAM+ motivates specialist-generalist collaborative learning, but also uses confidence-aware regularization to avoid generalist misguidance in scarce-label medical settings.
- CSL shows why confidence alone can fail under overconfidence; the update therefore adds area-prior and agreement checks before accepting high-confidence pseudo labels.
- DiffMatch highlights class imbalance and Matthew-effect failure in limited-label semi-supervised segmentation; the update weakens head-class self-reinforcement and adds per-class prior limits.
- CorrMatch and UniMatch support preserving weak-to-strong consistency and relational consistency, but only after pseudo targets are reliable enough.

## Implemented V3 Updates

- Added class-area anti-collapse guarding in `parc_sam/ssl/proposal_bank.py`.
  Large foreground pseudo regions are converted to background-foreground ambiguous candidate sets and excluded from reliable pseudo-label CE/prototype updates.
- Added SAM reliability calibration.
  SAM foreground evidence is weighted by predicted IoU, foreground/background weights, and stricter confidence thresholds.
- Added SAM prompt area control in `parc_sam/sam/proposal_engine.py`.
  Oversized teacher foreground regions are reduced to top-scoring prompt regions, and low-confidence prompts are skipped instead of passed into SAM.
- Added unsupervised ramp-up in `parc_sam/engine/trainer.py`.
  The V100 config now starts unsupervised learning after 300 iterations and ramps it over 1800 iterations.
- Fixed zero-weight pseudo loss in `parc_sam/losses.py`.
- Added diagnostics: `unsup_ramp`, `area_guard_ratio`, per-class pseudo ratios, per-class student ratios, and `sam_iou_mean`.
- Added unit tests for the all-foreground collapse case and zero-weight pseudo loss.

## Next Run

Use the v3 V100 config and output directory:

```bash
cd /root/autodl-tmp/PARC_SAM_SSL
bash scripts/train_v100_32g_echo.sh
```

Evaluation:

```bash
cd /root/autodl-tmp/PARC_SAM_SSL
bash scripts/test_v100_32g_echo.sh
```

During the first 2000 iterations, monitor:

- `pseudo_foreground_ratio`: should not climb toward `0.9+`.
- `student_class_1_ratio` and `student_class_2_ratio`: neither foreground class should become the whole image.
- `area_guard_ratio`: may be high early; should fall as predictions become plausible.
- `validation.jsonl`: use `best.pt`, not `final.pt`, for the final comparison.
