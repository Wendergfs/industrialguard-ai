import numpy as np
import pytest

from industrialguard.evaluation.metrics import au_pro, select_threshold, threshold_sweep


def test_mad_threshold_uses_only_normal_validation_scores():
    threshold = select_threshold(
        [0, 0, 0, 1], [1.0, 2.0, 3.0, 100.0], method="mad", mad_multiplier=3.5
    )
    assert threshold == pytest.approx(2.0 + 3.5 * 1.4826)


def test_mad_threshold_handles_tied_normal_scores_conservatively():
    threshold = select_threshold([0, 0, 1], [2.0, 2.0, 9.0], method="mad")
    assert threshold > 2.0


def test_threshold_sweep_reports_confusion_and_overlap_for_fixed_candidates():
    rows = threshold_sweep([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], [0.35, 0.5])
    assert rows[0] == {
        "threshold": 0.35,
        "tp": 2,
        "fp": 1,
        "fn": 0,
        "tn": 1,
        "precision": pytest.approx(2 / 3),
        "recall": 1.0,
        "f1": pytest.approx(0.8),
        "dice": pytest.approx(0.8),
        "iou": pytest.approx(2 / 3),
    }
    assert rows[1]["tp"] == 1
    assert rows[1]["fp"] == 0
    assert rows[1]["fn"] == 1
    assert rows[1]["tn"] == 2


def test_f1_selection_matches_sweep_and_requires_both_validation_classes():
    labels = [0, 0, 1, 1]
    scores = [0.1, 0.4, 0.35, 0.8]
    threshold = select_threshold(labels, scores, method="f1")
    selected = threshold_sweep(labels, scores, [threshold])[0]
    assert selected["f1"] == pytest.approx(0.8)
    with pytest.raises(ValueError, match="both classes"):
        select_threshold([0, 0], [0.1, 0.2], method="f1")


def test_threshold_inputs_reject_invalid_values():
    with pytest.raises(ValueError, match="binary"):
        select_threshold([0, 2], [0.1, 0.2])
    with pytest.raises(ValueError, match="thresholds"):
        threshold_sweep([0, 1], [0.1, 0.2], [])


def test_aupro_matches_anomalib_reference_on_ties_and_diagonal_components():
    pytest.importorskip("anomalib.metrics")
    from anomalib.metrics.aupro import _AUPRO
    import torch

    masks = np.zeros((2, 5, 5), dtype=bool)
    masks[0, 1, 1] = True
    masks[0, 2, 2] = True  # diagonal pixels are one 8-connected region
    masks[1, 3, 2:4] = True
    maps = np.asarray(
        [
            [[0.1, 0.2, 0.3, 0.2, 0.1]] * 5,
            [[0.2, 0.2, 0.8, 0.8, 0.2]] * 5,
        ],
        dtype=np.float32,
    )
    max_fpr = 0.3
    reference = _AUPRO(fpr_limit=max_fpr)
    reference.update(
        torch.from_numpy(maps),
        torch.from_numpy(masks.astype("float32")),
    )
    expected = reference.compute()

    assert au_pro(masks, maps, max_fpr=max_fpr) == pytest.approx(
        float(expected), abs=1e-6
    )
