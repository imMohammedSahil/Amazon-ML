import time
import polars as pl
import numpy as np
from pathlib import Path

def compute_f05_score(pred_dict: dict, gt_dict: dict) -> dict:
    """
    Computes exact competition Macro F0.5 evaluation metric:
    - Precision weighted 4x higher than Recall (beta = 0.5)
    - Perfect 1.0 reward for true singleton prediction (empty pred on empty GT)
    """
    total = len(gt_dict)
    f05_scores = []
    precisions = []
    recalls = []

    singleton_tp = 0
    singleton_total = 0

    for s1_id, gt_set in gt_dict.items():
        pred_set = pred_dict.get(s1_id, set())

        # Singleton evaluation
        if len(gt_set) == 0:
            singleton_total += 1
            if len(pred_set) == 0:
                singleton_tp += 1
                f05_scores.append(1.0)
                precisions.append(1.0)
                recalls.append(1.0)
            else:
                f05_scores.append(0.0)
                precisions.append(0.0)
                recalls.append(0.0)
            continue

        # Matched entity evaluation
        if len(pred_set) == 0:
            f05_scores.append(0.0)
            precisions.append(0.0)
            recalls.append(0.0)
            continue

        tp = len(gt_set.intersection(pred_set))
        p = tp / len(pred_set)
        r = tp / len(gt_set)

        precisions.append(p)
        recalls.append(r)

        if p + r == 0:
            f05_scores.append(0.0)
        else:
            # F_0.5 formula: (1 + 0.25) * P * R / (0.25 * P + R)
            f05 = (1.25 * p * r) / (0.25 * p + r)
            f05_scores.append(f05)

    macro_f05 = float(np.mean(f05_scores))
    macro_p = float(np.mean(precisions))
    macro_r = float(np.mean(recalls))
    singleton_acc = (singleton_tp / singleton_total) if singleton_total > 0 else 1.0

    return {
        "macro_f05": macro_f05,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "singleton_accuracy": singleton_acc,
        "total_evaluated": total
    }
