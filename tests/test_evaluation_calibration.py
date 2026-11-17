import pytest
import torch

from industrialguard.evaluation.runner import evaluate_model
from industrialguard.models.base import AnomalyOutput


def batch(scores, labels, sample_ids):
    images = torch.tensor(scores, dtype=torch.float32).view(-1, 1, 1, 1)
    masks = torch.tensor(labels, dtype=torch.float32).view(-1, 1, 1, 1)
    return {
        "image": images,
        "mask": masks,
        "label": torch.tensor(labels, dtype=torch.long),
        "category": ["bracket_black"] * len(scores),
        "sample_id": sample_ids,
    }


class SingleBatchLoader:
    def __init__(self, value):
        self.value = value
        self.iterated = False

    def __iter__(self):
        self.iterated = True
        yield self.value


class ThresholdModel:
    def eval(self):
        return self

    def predict(self, images):
        scores = images.flatten(start_dim=1).mean(dim=1)
        return AnomalyOutput(scores, images)

    def metadata(self):
        return {"model": "synthetic"}


class SyntheticDataModule:
    dataset_name = "mpdd"

    def __init__(self, calibration_loader, test_batch):
        self.calibration_loader = calibration_loader
        self.test_loader = SingleBatchLoader(test_batch)

    def val_dataloader(self):
        raise AssertionError("Independent calibration loader should be used")

    def test_dataloader(self):
        assert self.calibration_loader.iterated, "test was opened before calibration"
        return self.test_loader


def test_independent_calibration_freezes_threshold_before_test_access():
    calibration = SingleBatchLoader(
        batch(
            [0.1, 0.2, 0.8, 0.9],
            [0, 0, 1, 1],
            ["val-normal-1", "val-normal-2", "val-defect-1", "val-defect-2"],
        )
    )
    test = batch([0.05, 0.95], [0, 1], ["test-normal", "test-defect"])
    report = evaluate_model(
        ThresholdModel(),
        SyntheticDataModule(calibration, test),
        calibration_loader=calibration,
        threshold_method="f1",
        device="cpu",
    )

    calibrated = report["thresholds_by_category"]["bracket_black"]
    assert calibrated["image"] == pytest.approx(0.8)
    assert calibrated["pixel"] == pytest.approx(0.8)
    assert calibrated["validation_normal_images"] == 2
    assert calibrated["validation_anomalous_images"] == 2
    assert report["threshold_source"] == "independent labeled validation"
    assert [item["sample_id"] for item in report["per_category"][0]["per_image"]] == [
        "test-normal",
        "test-defect",
    ]
    assert report["per_category"][0]["per_image"][0]["image_outcome"] == "tn"
    assert report["per_category"][0]["per_image"][1]["image_outcome"] == "tp"
