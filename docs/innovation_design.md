# PARC-SAM-SSL v2 Innovation Design

## Target Claim

Semi-supervised segmentation with SAM should not ask whether SAM is correct on every pixel. A stronger formulation is: under target-domain uncertainty, SAM provides a structured proposal distribution, and the trainable specialist learns from calibrated candidate label sets instead of brittle hard annotations.

## Innovation 1: Calibrated Set-Valued Foundation Supervision

Problem: Hard pseudo labels and fixed confidence thresholds amplify early teacher mistakes, especially in small medical datasets and foreground-minority classes.

Mechanism:

- Fit class-wise conformal-style risk thresholds from labeled predictions.
- Convert EMA teacher, SAM proposals, and prototype evidence into candidate label sets.
- Use hard pseudo-label CE only for singleton candidate pixels.
- Use multi-label proposal-set loss for ambiguous pixels.
- Use safe negative evidence to suppress impossible classes.

Reference support:

- ConformalSAM uses conformal prediction to calibrate foundation-model masks for SSSS.
- When Confidence Fails shows confidence-only pseudo-label selection is unreliable.
- UniMatch remains the weak-to-strong consistency baseline that PARC generalizes.

Code mapping:

- `parc_sam/ssl/conformal.py`
- `parc_sam/ssl/proposal_bank.py`
- `parc_sam/losses.py`

## Innovation 2: Specialist-Generalist Prompt Loop With Deploy-Only Student

Problem: A SAM-assisted SSL method can look strong in training but become hard to deploy if SAM is part of the inference path, or can collapse if SAM is treated as an unquestioned teacher.

Mechanism:

- The EMA teacher creates class-wise prompts from unlabeled weak views.
- SAM returns foreground proposals per class.
- The student learns from proposal sets during training; inference uses only the compact student.
- Smoke configs can use surrogate proposals, but real configs require an actual SAM checkpoint unless explicitly overridden.

Reference support:

- Segment Anything establishes promptable segmentation foundation models.
- SemiSAM+ supports the specialist-generalist collaboration framing for medical SSL.
- SSR-SAM supports diverse prompt-driven consistency for SAM-based SSL.

Code mapping:

- `parc_sam/sam/proposal_engine.py`
- `parc_sam/engine/trainer.py`
- `evaluate.py`

## Innovation 3: Prototype-Correlation Alignment Across Unlabeled Regions

Problem: Prior KnowSAM-style runs showed mechanism collapse and foreground starvation. A proposal system must explicitly preserve minority foreground evidence and relation structure.

Mechanism:

- Clamp foreground risk thresholds separately from background.
- Track class-wise candidate/pseudo/singleton/negative ratios in `diagnostics.jsonl`.
- Rescue only high-evidence foreground pixels when foreground candidate participation falls below a configured floor.
- Maintain a semantic prototype memory updated from labeled pixels and reliable unlabeled proposal pixels.
- Use prototype logits as a third evidence source beside teacher and SAM.
- Regularize unlabeled features so feature correlation follows proposal probability correlation.
- When real SAM embeddings are available, align student region features with SAM token-region embeddings.

Reference support:

- CorrMatch and RankMatch support correlation/relation consistency beyond pixel-wise losses.
- AllSpark supports architecture-level interaction between labeled and unlabeled features.
- DiffMatch motivates class-imbalance-aware alternatives to brittle discriminative pseudo-labeling.
- S4Former supports negative-class regularization for SSSS.

Code mapping:

- `parc_sam/ssl/semantic_memory.py`
- `parc_sam/ssl/conformal.py`
- `parc_sam/ssl/proposal_bank.py`
- `parc_sam/losses.py`
- `parc_sam/engine/trainer.py`

## Minimum Evidence Package

Main comparisons:

- KnowSAM multiclass baseline.
- UniMatch-style weak-to-strong baseline with the same student.
- Student + SAM hard pseudo labels.
- PARC-SAM-SSL full model.

Ablations:

- Remove conformal candidate sets.
- Replace proposal-set loss with hard pseudo CE.
- Remove SAM proposals.
- Remove prototype memory.
- Remove correlation consistency.
- Remove SAM token-region alignment.

Diagnostics:

- Foreground participation ratio per class.
- Candidate-set size distribution.
- SAM/teacher conflict ratio.
- Risk thresholds per class.
- Best-vs-final checkpoint curves to detect late collapse.
