import pytest
import torch
from torch import nn

from industrialguard.evaluation.metrics import (
    aggregate_by_category,
    au_pro,
    pixel_ranking_metrics,
    ranking_metrics,
    segmentation_metrics,
    select_threshold,
)
from industrialguard.models.base import AnomalyOutput
from industrialguard.models.padim import PaDiM
from industrialguard.models.patchcore import PatchCore, greedy_coreset
from industrialguard.models.registry import create_model, list_models


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1)
        self.layer2 = nn.Conv2d(8, 12, kernel_size=3, stride=2, padding=1)
        self.layer3 = nn.Conv2d(12, 16, kernel_size=3, stride=2, padding=1)

    def forward(self, images):
        first = self.layer1(images).relu()
        second = self.layer2(first).relu()
        third = self.layer3(second).relu()
        return {"layer1": first, "layer2": second, "layer3": third}


def normal_batches(count=4, size=64):
    generator = torch.Generator().manual_seed(4)
    images = torch.randn(count, 3, size, size, generator=generator)
    return [
        {
            "image": images[index : index + 2],
            "label": torch.zeros(min(2, count - index), dtype=torch.long),
        }
        for index in range(0, count, 2)
    ]


@pytest.mark.parametrize(
    "model",
    [
        PaDiM(
            backbone=TinyBackbone(),
            feature_dim=6,
            patch_batch_size=32,
            smoothing_sigma=1,
            pretrained=False,
        ),
        PatchCore(
            backbone=TinyBackbone(),
            coreset_ratio=0.2,
            max_memory_bank=32,
            max_candidates=256,
            patch_stride=4,
            smoothing_sigma=1,
            pretrained=False,
        ),
    ],
)
def test_models_fit_predict_and_checkpoint_cpu(tmp_path, model):
    model.fit(normal_batches())
    output = model.predict(torch.randn(2, 3, 64, 64))
    assert isinstance(output, AnomalyOutput)
    assert output.anomaly_map.shape == (2, 1, 64, 64)
    assert output.image_score.shape == (2,)
    assert torch.isfinite(output.anomaly_map).all()
    assert torch.isfinite(output.image_score).all()
    assert output.anomaly_map.device.type == "cpu"
    checkpoint = tmp_path / f"{model.model_name}.pt"
    model.save_checkpoint(checkpoint)
    restored_config = {
        key: value for key, value in model.model_config.items() if key != "backbone"
    }
    restored = type(model)(backbone=TinyBackbone(), **restored_config)
    restored.load_state(checkpoint)
    restored_output = restored.predict(torch.randn(1, 3, 64, 64))
    assert restored_output.anomaly_map.shape == (1, 1, 64, 64)


def test_registry_and_deterministic_coreset():
    assert list_models() == ("padim", "patchcore", "efficientad")
    with pytest.raises(ValueError, match="Unknown model"):
        create_model("not-a-model")
    features = torch.randn(100, 16, generator=torch.Generator().manual_seed(1))
    first = greedy_coreset(features, 12, seed=7)
    second = greedy_coreset(features, 12, seed=7)
    assert torch.equal(first, second)
    assert first.shape == (12, 16)


def test_metric_values_thresholds_and_one_class_handling():
    labels = [0, 0, 1, 1]
    scores = [0.05, 0.1, 0.8, 0.9]
    assert ranking_metrics(labels, scores) == {"auroc": 1.0, "average_precision": 1.0}
    assert ranking_metrics([0, 0], [0.2, 0.3]) == {
        "auroc": None,
        "average_precision": None,
    }
    assert select_threshold([0, 0, 0], [1, 2, 3], quantile=0.9) >= 2.0
    with pytest.raises(ValueError, match="both classes"):
        select_threshold([0, 0], [0.1, 0.2], method="f1")
    masks = torch.tensor([[[0, 0], [0, 1]], [[0, 0], [1, 1]]], dtype=torch.bool)
    maps = masks.float() * 0.8 + 0.1
    pixel = pixel_ranking_metrics(masks, maps)
    assert pixel["auroc"] == 1.0
    assert pixel["average_precision"] == 1.0
    assert pixel["aupro"] == pytest.approx(1.0)
    assert segmentation_metrics(masks, maps, 0.5)["dice"] == 1.0
    assert au_pro(torch.zeros((1, 2, 2)), torch.zeros((1, 2, 2))) is None
    summaries = aggregate_by_category(
        [
            {"dataset": "mpdd", "category": "a", "auroc": 0.8},
            {"dataset": "mpdd", "category": "b", "auroc": 1.0},
            {"dataset": "mvtec", "category": "c", "auroc": 0.5},
        ]
    )
    assert summaries["mpdd"]["macro_mean"]["auroc"] == pytest.approx(0.9)
    assert set(summaries) == {"mpdd", "mvtec"}


def test_efficientad_upstream_architecture_maps_on_cpu():
    pytest.importorskip("anomalib.models.image.efficient_ad.torch_model")
    from industrialguard.models.efficientad import EfficientAD

    model = EfficientAD(model_size="s", teacher_out_channels=64)
    model.network.eval()
    with torch.inference_mode():
        teacher_student_map, student_autoencoder_map = model.network.get_maps(
            torch.rand(1, 3, 256, 256), normalize=False
        )
    assert teacher_student_map.shape == (1, 1, 256, 256)
    assert student_autoencoder_map.shape == (1, 1, 256, 256)
    assert torch.isfinite(teacher_student_map).all()
    assert torch.isfinite(student_autoencoder_map).all()
    with pytest.raises(ValueError, match="penalty_loader"):
        model.fit([], [])
