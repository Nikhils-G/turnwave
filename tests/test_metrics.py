import pytest

from turnwave.metrics import average_precision, binary_metrics


def test_binary_metrics_perfect():
    m = binary_metrics([0.9, 0.1, 0.8, 0.2], [1, 0, 1, 0])
    assert m == {"accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0}


def test_binary_metrics_mixed():
    # preds: 1, 1, 0, 0 vs labels 1, 0, 1, 0 -> tp=1 fp=1 fn=1 tn=1
    m = binary_metrics([0.9, 0.7, 0.3, 0.1], [1, 0, 1, 0])
    assert m["accuracy"] == 0.5
    assert m["precision"] == 0.5
    assert m["recall"] == 0.5
    assert m["f1"] == 0.5


def test_average_precision_hand_computed():
    # ranked: (0.9, pos) P=1/1; (0.8, neg); (0.7, pos) P=2/3 -> AP = (1 + 2/3)/2
    ap = average_precision([0.9, 0.8, 0.7], [1, 0, 1])
    assert ap == pytest.approx((1 + 2 / 3) / 2)


def test_average_precision_no_positives():
    assert average_precision([0.9, 0.1], [0, 0]) == 0.0
