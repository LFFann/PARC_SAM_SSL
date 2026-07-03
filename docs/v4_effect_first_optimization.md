# V4 Effect-First Optimization

This note records the root-cause-driven v4 revision after the failed `PARC_SAM_SSL_v2_V100_32G_echoData` run. The goal is not to hide bad predictions with engineering filters, but to change the semi-supervised learning mechanism so the model has a real path to better segmentation.

## Failure Root Cause

The v2 framework formed a circular confirmation loop:

```text
weak EMA teacher prediction -> SAM prompt -> SAM proposal -> pseudo target -> student update -> EMA teacher
```

The latest failed run shows this loop clearly:

- Best validation foreground Dice was only `0.014071` at iteration 400, and the final logged Dice dropped to `0.005660` at iteration 2200.
- At iteration 2350, `student_foreground_ratio=0.982628`, `teacher_foreground_ratio=0.988483`, and `pseudo_foreground_ratio=0.982540`.
- Class 2 disappeared: `student_class_2_ratio=0.0` and `class_2_pseudo_ratio=0.0`.
- Visualizations showed the ultrasound fan region almost entirely marked as reliable singleton foreground.

The important conclusion is that SAM was not an independent semantic teacher. It was prompted by the weak teacher, then amplified the teacher's wrong foreground region. The framework optimized confidently wrong pseudo labels.

## Literature Basis

| Direction | Reference | Usable idea for v4 |
| --- | --- | --- |
| SAM-based medical SSL | SemiSAM+ / MedIA 2025, arXiv: https://arxiv.org/html/2502.20749v1 | Use a specialist-generalist relation and confidence-aware regularization to avoid generalist misguidance. |
| Prototype-driven foundation SSL | SemiSAM-O1 / arXiv 2026: https://arxiv.org/abs/2604.24109 | Use labeled semantic prototypes and uncertainty correction instead of relying only on online foundation outputs. |
| SAM-2 semi-supervised medical segmentation | SSS / arXiv 2025: https://arxiv.org/abs/2506.08949 | Add constraints to prompt generation so SAM priors do not drift freely. |
| Uncertainty-aware consistency | DyCON / CVPR 2025: https://openaccess.thecvf.com/content/CVPR2025/papers/Assefa_DyCON_Dynamic_Uncertainty-aware_Consistency_and_Contrastive_Learning_for_Semi-supervised_Medical_CVPR_2025_paper.pdf | Preserve informative uncertain regions instead of filtering them away. |
| Confidence failure in pseudo labels | CSL / ICCV 2025: https://openaccess.thecvf.com/content/ICCV2025/papers/Liu_When_Confidence_Fails_Revisiting_Pseudo-Label_Selection_in_Semi-supervised_Semantic_Segmentation_ICCV_2025_paper.pdf | High confidence is not reliability; overconfident pixels need context and sample-adaptive treatment. |
| Class imbalance in SSL | DiffMatch / ICLR 2025: https://proceedings.iclr.cc/paper_files/paper/2025/file/9e6293bfc454a286697e6487c141769b-Paper-Conference.pdf | Suppress majority-class Matthew-effect reinforcement. |
| Relation propagation | CorrMatch / CVPR 2024: https://openaccess.thecvf.com/content/CVPR2024/papers/Sun_CorrMatch_Label_Propagation_via_Correlation_Matching_for_Semi-Supervised_Semantic_Segmentation_CVPR_2024_paper.pdf | Use pixel/region relations to spread reliable structure rather than isolated confidence. |
| Semantic memory | AllSpark / CVPR 2024: https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_AllSpark_Reborn_Labeled_Features_from_Unlabeled_in_Transformer_for_Semi-Supervised_CVPR_2024_paper.pdf | Use memory-like semantic anchors to reduce low-quality pseudo-label domination. |
| Medical class imbalance | Gradient-Aware / ECCV 2024: https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/07246.pdf | Treat class imbalance as a training-gradient problem, not only a data-count problem. |

## Three Core Innovations

### 1. Prototype-Calibrated Foundation Prompting

SAM is no longer prompted by `teacher_prob` alone. The training step first updates labeled semantic prototypes from the supervised batch, computes prototype logits on the EMA teacher features, blends prototype probabilities with the EMA teacher probability, and constrains the foreground prompt area before SAM sees it:

```text
prompt_prob = w_teacher * teacher_prob + w_proto * prototype_prob
prompt_prob = area_constrained(prompt_prob)
SAM proposal = SAM(image, prompt_prob)
```

This directly attacks the previous teacher-SAM self-confirmation loop. If the teacher expands the foreground fan but labeled prototypes still encode object semantics, the prompt prior can pull SAM back toward plausible semantic regions.

Implemented in:

- `parc_sam/engine/trainer.py`: `_build_prompt_probability`
- `configs/parc_sam_ssl_v100_32g_echo.yaml`: `use_prototype_prompt`, `prompt_teacher_weight`, `prompt_prototype_weight`, `prompt_temperature`, `prompt_max_foreground_ratio`, `prompt_min_foreground_confidence`

### 2. Uncertainty-Paced Set Supervision

The v2 failure showed that high confidence can be catastrophically wrong. v4 therefore reduces hard pseudo-label dominance:

- hard CE/Dice pseudo supervision is still limited to reliable singleton pixels;
- ambiguous candidate-set pixels receive soft KL consistency through UPSC;
- the UPSC pixel weight starts by emphasizing uncertain/ambiguous regions and gradually shifts toward confident regions as the unsupervised ramp increases.

This follows the DyCON/CSL insight: uncertainty is not simply noise. It is useful supervision when pseudo labels are still unstable.

Implemented in:

- `parc_sam/losses.py`: `uncertainty_paced_consistency_loss`
- `parc_sam/engine/trainer.py`: `uncertainty_loss`
- `configs/parc_sam_ssl_v100_32g_echo.yaml`: `uncertainty_consistency`, `min_uncertainty_weight`, `ambiguity_bonus`

### 3. Class-Balanced Prototype-Relation Learning

The failed run erased class 2 completely. v4 strengthens class-balanced semantic anchors instead of only adding a foreground area guard:

- prototype evidence weight is increased;
- class-balance power is increased;
- foreground quantile and area ceiling are tightened;
- correlation consistency and prototype loss remain active to preserve local relation structure;
- SAM-token alignment is retained but with lower weight so bad SAM masks cannot dominate.

This is aligned with DiffMatch, Gradient-Aware SSL, CorrMatch, and AllSpark: minority classes need semantic and relational anchors, otherwise pseudo-label training amplifies the majority-class prediction.

## Expected Baseline-Beating Logic

Compared with KnowSAM/Baseline, v4 should have an advantage if the following conditions hold:

1. The labeled prototypes are stable enough to prevent class-2 disappearance.
2. `prompt_foreground_ratio` is lower and more plausible than `teacher_foreground_ratio` during the first 2000 iterations.
3. `sam_foreground_ratio` follows the calibrated prompt rather than becoming a full fan proposal.
4. `pseudo_foreground_ratio` does not climb above the healthy range before validation Dice improves.
5. Best checkpoint validation Dice improves over the supervised-only and original KnowSAM baselines, not just over the collapsed v2 run.

The relevant logs are:

- `metrics.jsonl`: `prompt_foreground_ratio`, `prompt_area_constraint_ratio`, `teacher_foreground_ratio`, `student_class_1_ratio`, `student_class_2_ratio`, `uncertainty`
- `diagnostics.jsonl`: prompt entropy, prompt weights, class-wise pseudo ratios, class-wise prompt ratios
- `visualizations/paper`: qualitative panels for paper
- `visualizations/diagnostic`: prompt/SAM/pseudo failure localization

## V100 Commands

```bash
cd /root/autodl-tmp/PARC_SAM_SSL
bash scripts/train_v100_32g_echo.sh
```

```bash
cd /root/autodl-tmp/PARC_SAM_SSL
bash scripts/test_v100_32g_echo.sh
```

Default paths:

- dataset: `/root/autodl-tmp/echoData/260703_data_labeled30pct`
- SAM checkpoint: `/root/autodl-tmp/sam_vit_b_01ec64.pth`
- output: `outputs/PARC_SAM_SSL_v4_ProtoPrompt_UPSC_V100_32G_echoData`

## Required Ablations

To support a top-conference claim, run at least:

1. Full v4.
2. Without prototype-calibrated prompting: set `target.use_prototype_prompt=false`.
3. Without UPSC: set `loss.uncertainty_consistency=0`.
4. Without class-balanced prototype evidence: reduce `target.prototype_evidence_weight=0` and `risk.class_balance_power=0`.
5. Without SAM: `--disable-sam`.
6. Supervised-only or KnowSAM baseline under the same labeled split.
