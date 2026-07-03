from __future__ import annotations

import torch
import torch.nn as nn

from parc_sam.models import PARCStudent
from parc_sam.engine.visualization import TrainingVisualizer
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
