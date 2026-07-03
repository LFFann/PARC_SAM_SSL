# 2023-2026 Literature Map For PARC-SAM-SSL

Search date: 2026-07-02. Sources prioritize CVF/OpenReview/AAAI/arXiv/IEEE/ScienceDirect and exclude MDPI-style low-signal sources.

## Closest High-Signal Papers

| Year | Paper | Venue / source | Why it matters for PARC-SAM-SSL |
| --- | --- | --- | --- |
| 2023 | Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation | CVPR 2023, CVF: https://openaccess.thecvf.com/content/CVPR2023/papers/Yang_Revisiting_Weak-to-Strong_Consistency_in_Semi-Supervised_Semantic_Segmentation_CVPR_2023_paper.pdf | Establishes strong weak-to-strong consistency as a reliable SSSS baseline. PARC keeps this as the training skeleton, but replaces single hard pseudo labels with proposal sets. |
| 2023 | Segment Anything | arXiv / ICCV 2023 foundation model: https://arxiv.org/abs/2304.02643 | Supplies the promptable segmentation generalist. PARC treats SAM as a target-domain-calibrated proposal model, not as a final oracle. |
| 2024 | CorrMatch: Label Propagation via Correlation Matching for Semi-Supervised Semantic Segmentation | CVPR 2024, CVF: https://openaccess.thecvf.com/content/CVPR2024/papers/Sun_CorrMatch_Label_Propagation_via_Correlation_Matching_for_Semi-Supervised_Semantic_Segmentation_CVPR_2024_paper.pdf | Shows correlation maps carry same-class grouping and shape information. PARC adds correlation consistency on unlabeled student features and proposal probabilities. |
| 2024 | RankMatch: Exploring the Better Consistency Regularization for Semi-supervised Semantic Segmentation | CVPR 2024, CVF: https://openaccess.thecvf.com/content/CVPR2024/html/Mai_RankMatch_Exploring_the_Better_Consistency_Regularization_for_Semi-supervised_Semantic_Segmentation_CVPR_2024_paper.html | Supports moving beyond pixel-wise consistency to inter-pixel and inter-agent relation constraints. |
| 2024 | AllSpark: Reborn Labeled Features from Unlabeled in Transformer for Semi-Supervised Semantic Segmentation | CVPR 2024, CVF: https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_AllSpark_Reborn_Labeled_Features_from_Unlabeled_in_Transformer_for_Semi-Supervised_CVPR_2024_paper.pdf | Argues for architecture-level interaction between labeled and unlabeled feature flows. PARC implements a semantic prototype memory updated by both labeled and reliable unlabeled regions. |
| 2024 | Training Vision Transformers for Semi-Supervised Semantic Segmentation | CVPR 2024, CVF: https://openaccess.thecvf.com/content/CVPR2024/html/Hu_Training_Vision_Transformers_for_Semi-Supervised_Semantic_Segmentation_CVPR_2024_paper.html | Supports transformer/global context specific regularization; relevant if replacing the UNet student with a ViT student later. |
| 2025 | DiffMatch: Towards Unbiased Learning in Semi-Supervised Semantic Segmentation | ICLR 2025 proceedings: https://proceedings.iclr.cc/paper_files/paper/2025/file/9e6293bfc454a286697e6487c141769b-Paper-Conference.pdf | Frames class imbalance / Matthew effect as a central weakness of discriminative pseudo-labeling. PARC counters this with class-conditional risk and prototype memory. |
| 2025 | ConformalSAM: Unlocking the Potential of Foundational Segmentation Models in Semi-Supervised Semantic Segmentation with Conformal Prediction | ICCV 2025, CVF: https://openaccess.thecvf.com/content/ICCV2025/papers/Chen_ConformalSAM_Unlocking_the_Potential_of_Foundational_Segmentation_Models_in_Semi-Supervised_ICCV_2025_paper.pdf | Directly supports target-domain calibration of foundation-model masks before using them as supervision. PARC uses class-conditional conformal-style candidate sets for both teacher and SAM evidence. |
| 2025 | When Confidence Fails: Revisiting Pseudo-Label Selection in Semi-supervised Semantic Segmentation | ICCV 2025, CVF: https://openaccess.thecvf.com/content/ICCV2025/papers/Liu_When_Confidence_Fails_Revisiting_Pseudo-Label_Selection_in_Semi-supervised_Semantic_Segmentation_ICCV_2025_paper.pdf | Shows max-confidence selection can fail under overconfidence. PARC avoids confidence-only filtering by combining separability, candidate set size, SAM agreement, and prototype evidence. |
| 2025 | SemiSAM+: Rethinking Semi-Supervised Medical Image Segmentation in the Era of Foundation Models | arXiv / MedIA 2025: https://arxiv.org/abs/2502.20749 | Supports the specialist-generalist collaborative framing for medical SSL. PARC narrows this into training-only SAM proposal supervision plus deployable specialist student. |
| 2026 | SSR-SAM: Retrieval-Style Segment Anything Model for Semi-Supervised Ultra-High-Resolution Image Segmentation | AAAI 2026: https://ojs.aaai.org/index.php/AAAI/article/view/37566 | Supports diverse visual-semantic prompt consistency for SAM-based SSL. PARC uses dynamic class-wise prompts and proposal consistency rather than a fixed prompt path. |

## Opportunity Gap

Recent SSSS methods are strong, but three gaps remain useful for a CVPR/ICCV/TPAMI-style claim:

1. Most SSL frameworks still collapse pseudo-label quality into a single hard label or fixed confidence threshold, even though 2025 work shows confidence can be poorly calibrated.
2. SAM-based SSL often uses the foundation model as an annotator/teacher; less work treats SAM output as a calibrated proposal set with abstention, conflict, and negative evidence.
3. Correlation/prototype methods improve unlabeled feature usage, but they are rarely tied to SAM proposal masks and target-domain risk control in one framework.

## PARC-SAM-SSL Positioning

PARC-SAM-SSL v2 is not a KnowSAM module addition. It replaces the dual-branch KnowSAM fusion design with a single deployable student and a training-only generalist proposal loop:

```text
labeled data -> supervised student + class-wise risk calibration
unlabeled weak view -> EMA teacher -> prompts -> SAM proposal set
unlabeled strong view -> student -> singleton hard CE + ambiguous set-valued supervision
student features -> semantic memory + correlation consistency + optional SAM token-region alignment
```

The closest prior-art risk is ConformalSAM, because it already connects foundation segmentation models and conformal prediction in SSSS. The v2 differentiation is to make the training target set-valued rather than only filtering to hard reliable pixels, add foreground-safe class-balanced calibration against minority-class collapse, and expose class-wise diagnostics as part of the framework rather than an after-the-fact plot.
