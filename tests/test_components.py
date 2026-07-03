from __future__ import annotations

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader

from parc_sam.models import PARCStudent
from parc_sam.engine.evaluator import evaluate
from parc_sam.engine.visualization import TrainingVisualizer
from parc_sam.losses import uncertainty_paced_consistency_loss, weighted_pseudo_loss
from parc_sam.sam import SAMProposalEngine
from parc_sam.ssl import ClassConditionalRiskController, ProposalSetBuilder, SemanticPrototypeMemory


def test_student_forward_shape():
    model = PARCStudent(in_channels=3, num_classes=3, base_channels=8, feature_dim=32)
    out = model(torch.rand(2, 3, 64, 64), return_features=True)
    assert out["logits"].shape == (2, 3, 64, 64)
    assert out["features"].shape == (2, 32, 64, 64)


def test_risk_controller_candidate_sets_are_nonempty():
    risk = ClassConditionalRiskController(num_classes=3, alpha=0.2, min_pixels_per_class=2)
    probs = torch.softmax(torch.randn(2, 3, 8, 8), dim=1)
    masks = torch.randint(0, 3, (2, 8, 8))
    risk.update(probs, masks)
    candidate, low = risk.prediction_sets(probs)
    assert candidate.shape == probs.shape
    assert torch.all(candidate.sum(dim=1) >= 1)
    assert low.shape == (2, 8, 8)


def test_proposal_builder_with_surrogate_sam():
    probs = torch.softmax(torch.randn(2, 3, 16, 16), dim=1)
    risk = ClassConditionalRiskController(num_classes=3)
    risk.update(probs, torch.randint(0, 3, (2, 16, 16)))
    sam = {"valid": True, "prob": probs.flip(1)}
    builder = ProposalSetBuilder(3, {"max_candidate_set_size": 2, "teacher_confidence": 0.4, "min_sam_confidence": 0.4})
    targets = builder.build(probs, risk, sam)
    assert targets["pseudo"].shape == (2, 16, 16)
    assert targets["candidate_set"].shape == probs.shape
    assert targets["weight"].min() >= 0


def test_area_guard_prevents_all_foreground_singletons():
    risk = ClassConditionalRiskController(num_classes=3)
    risk.pixel_prior = torch.tensor([0.96, 0.02, 0.02])
    risk.q_per_class = torch.tensor([0.20, 0.66, 0.66])
    teacher = torch.zeros(1, 3, 32, 32)
    teacher[:, 0] = 0.04
    teacher[:, 1] = 0.93
    teacher[:, 2] = 0.03
    sam = {"valid": True, "prob": teacher.clone(), "iou": torch.ones(1, 2) * 0.9}
    builder = ProposalSetBuilder(
        3,
        {
            "use_risk": True,
            "use_sam": True,
            "use_prototype": False,
            "max_candidate_set_size": 2,
            "teacher_confidence": 0.6,
            "min_sam_confidence": 0.5,
            "class_area_guard": True,
            "max_foreground_area_multiplier": 2.0,
            "max_foreground_area_floor": 0.02,
            "max_foreground_area_ceiling": 0.08,
            "area_guard_min_pixels": 4,
            "area_guard_weight": 0.0,
        },
    )
    targets = builder.build(teacher, risk, sam)
    stats = targets["stats"]
    assert stats["area_guard_ratio"] > 0.80
    assert stats["class_1_pseudo_ratio"] < 0.10
    assert stats["class_1_singleton_ratio"] < 0.10
    assert targets["reliable"].float().mean() < 0.20


def test_weighted_pseudo_loss_zero_weight_is_zero():
    logits = torch.randn(2, 3, 8, 8)
    pseudo = torch.randint(0, 3, (2, 8, 8))
    weight = torch.zeros(2, 8, 8)
    loss = weighted_pseudo_loss(logits, pseudo, weight, num_classes=3)
    assert loss.item() == 0.0


def test_uncertainty_paced_consistency_loss_runs_with_candidate_sets():
    logits = torch.randn(2, 3, 8, 8, requires_grad=True)
    target_prob = torch.softmax(torch.randn(2, 3, 8, 8), dim=1)
    candidate_set = target_prob > 0.25
    loss = uncertainty_paced_consistency_loss(
        logits,
        target_prob,
        candidate_set,
        ramp=0.25,
        min_weight=0.05,
        ambiguity_bonus=0.7,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None


def test_prototype_memory_logits():
    memory = SemanticPrototypeMemory(num_classes=3, feature_dim=8, min_pixels=1)
    features = torch.randn(2, 8, 8, 8)
    masks = torch.randint(0, 3, (2, 8, 8))
    memory.update(features, masks)
    logits = memory.logits(features)
    assert logits is not None
    assert logits.shape == (2, 3, 8, 8)


def test_sam_surrogate_runs_without_checkpoint():
    cfg = {"enabled": True, "allow_surrogate_without_checkpoint": True, "checkpoint": "missing.pth"}
    engine = SAMProposalEngine(cfg, num_classes=3)
    probs = torch.softmax(torch.randn(2, 3, 16, 16), dim=1)
    out = engine(torch.rand(2, 3, 16, 16), probs)
    assert out["valid"] is True
    assert out["prob"].shape == probs.shape
    assert out["source"] == "surrogate"


def test_local_sam_prompt_format_uses_embeddings():
    cfg = {"enabled": False, "image_size": 1024}
    engine = SAMProposalEngine(cfg, num_classes=3)
    engine.sam_source = "local:/tmp/PARC_SAM_SSL"
    engine.sam = nn.Module()
    engine.sam.prompt_encoder = nn.Module()
    engine.sam.prompt_encoder.pe_layer = type(
        "PE",
        (),
        {"forward_with_coords": staticmethod(lambda coords, image_size: torch.zeros(coords.shape[0], coords.shape[1], 256, device=coords.device))},
    )()
    engine.sam.prompt_encoder.point_embeddings = nn.ModuleList([nn.Embedding(1, 256) for _ in range(4)])
    prompts = {
        "point_coords": torch.rand(5, 2, 2) * 1024,
        "point_labels": torch.tensor([[1, 0]] * 5),
        "boxes": torch.rand(5, 4) * 1024,
        "mask_inputs": torch.rand(5, 1, 256, 256),
    }
    points, boxes, masks = engine._format_prompts_for_sam(prompts)
    assert points[0].shape == (5, 2, 256)
    assert points[1].shape == (5, 2)
    assert boxes.shape == (5, 2, 256)
    assert masks.shape == (5, 1, 256, 256)


def test_training_visualizer_writes_outputs(tmp_path):
    visualizer = TrainingVisualizer(tmp_path, num_classes=3, config={"max_images": 1, "panel_size": 64})
    payload = {
        "images_l": torch.rand(1, 3, 32, 32),
        "masks_l": torch.randint(0, 3, (1, 32, 32)),
        "pred_l": torch.randint(0, 3, (1, 32, 32)),
        "labeled_ids": ["lab"],
        "weak_u": torch.rand(1, 3, 32, 32),
        "strong_u": torch.rand(1, 3, 32, 32),
        "unlabeled_ids": ["unlab"],
        "teacher_prob": torch.softmax(torch.randn(1, 3, 32, 32), dim=1),
        "prompt_prob": torch.softmax(torch.randn(1, 3, 32, 32), dim=1),
        "student_prob": torch.softmax(torch.randn(1, 3, 32, 32), dim=1),
        "sam_prob": torch.softmax(torch.randn(1, 3, 32, 32), dim=1),
        "pseudo": torch.randint(0, 3, (1, 32, 32)),
        "weight": torch.rand(1, 32, 32),
        "candidate_set": torch.rand(1, 3, 32, 32) > 0.5,
        "negative_set": torch.rand(1, 3, 32, 32) > 0.8,
        "reliable": torch.rand(1, 32, 32) > 0.2,
        "singleton": torch.rand(1, 32, 32) > 0.5,
        "soft_target": torch.softmax(torch.randn(1, 3, 32, 32), dim=1),
    }
    visualizer.write(1, payload, {"severity": "warn", "flags": ["unit_test"]})
    assert (tmp_path / "visualizations" / "paper").exists()
    assert (tmp_path / "visualizations" / "diagnostic").exists()
    assert (tmp_path / "visualizations" / "failure").exists()
    assert (tmp_path / "visualizations" / "manifest.jsonl").exists()


def test_evaluator_writes_color_and_overlay_predictions(tmp_path):
    class FixedModel(nn.Module):
        def forward(self, image):
            b, _, h, w = image.shape
            logits = image.new_zeros((b, 3, h, w))
            logits[:, 0] = 0.2
            logits[:, 1, : h // 2, : w // 2] = 4.0
            logits[:, 2, h // 2 :, w // 2 :] = 4.0
            return logits

    image = torch.zeros(3, 16, 16)
    image[0] = 0.4
    image[1] = 0.5
    image[2] = 0.6
    mask = torch.zeros(16, 16, dtype=torch.long)
    mask[:8, :8] = 1
    mask[8:, 8:] = 2
    loader = DataLoader([{"image": image, "mask": mask, "id": "case_a"}], batch_size=1)
    metrics = evaluate(FixedModel(), loader, num_classes=3, device="cpu", save_dir=tmp_path)
    assert metrics["avg_dice"] > 0.99
    for folder in ["pred_mask", "gt_mask", "image", "pred_color", "gt_color", "pred_overlay", "gt_overlay"]:
        assert (tmp_path / folder / "case_a.png").exists()
    pred_color = Image.open(tmp_path / "pred_color" / "case_a.png")
    overlay = Image.open(tmp_path / "pred_overlay" / "case_a.png")
    assert pred_color.mode == "RGB"
    assert overlay.mode == "RGB"
    assert len(pred_color.getcolors(maxcolors=256 * 256)) >= 3
