"""Binary classification metrics, implemented directly (no sklearn dependency)."""


def binary_metrics(probs: list[float], labels: list[float], threshold: float = 0.5) -> dict:
    tp = fp = fn = tn = 0
    for p, y in zip(probs, labels, strict=True):
        pred = p >= threshold
        pos = y >= 0.5
        if pred and pos:
            tp += 1
        elif pred and not pos:
            fp += 1
        elif not pred and pos:
            fn += 1
        else:
            tn += 1
    total = tp + fp + fn + tn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def average_precision(probs: list[float], labels: list[float]) -> float:
    """Area under the precision-recall curve (AP), computed over the ranked list."""
    n_pos = sum(1 for y in labels if y >= 0.5)
    if n_pos == 0:
        return 0.0
    ranked = sorted(zip(probs, labels, strict=True), key=lambda t: t[0], reverse=True)
    hits = 0
    ap = 0.0
    for rank, (_, y) in enumerate(ranked, start=1):
        if y >= 0.5:
            hits += 1
            ap += hits / rank
    return ap / n_pos
