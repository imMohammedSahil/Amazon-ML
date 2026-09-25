from typing import Dict, List, Set, Union
import numpy as np


def compute_entity_f_beta(
    true_matches: Union[Set[str], List[str]], 
    pred_matches: Union[Set[str], List[str]], 
    beta: float = 0.5
) -> float:
    """
    Computes singleton-aware F_beta score for a single S1 entity.
    Beta = 0.5 heavily weights Precision over Recall (Competition official metric).
    
    Corner Cases:
      - Singleton True ([]) & Singleton Pred ([]) => 1.0
      - Singleton True ([]) & Match Pred ([...])   => 0.0
      - Match True ([...]) & Singleton Pred ([])   => 0.0
      - Match True ([...]) & Match Pred ([...])    => Standard F_beta
    """
    true_set = set(true_matches)
    pred_set = set(pred_matches)

    # Both empty: correctly identified singleton
    if len(true_set) == 0 and len(pred_set) == 0:
        return 1.0
    
    # One empty, one non-empty: failed
    if len(true_set) == 0 or len(pred_set) == 0:
        return 0.0

    tp = len(true_set.intersection(pred_set))
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)

    if tp == 0:
        return 0.0

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    beta_sq = beta ** 2
    f_beta = (1.0 + beta_sq) * (precision * recall) / (beta_sq * precision + recall)
    return float(f_beta)


def evaluate_macro_f05(
    ground_truth: Dict[str, List[str]], 
    predictions: Dict[str, List[str]],
    beta: float = 0.5
) -> Dict[str, float]:
    """
    Computes Macro-averaged F_0.5 score across all S1 entities.
    Returns:
      dict with 'macro_f05', 'singleton_accuracy', 'match_macro_f05', 'total_entities'
    """
    scores = []
    singleton_true_count = 0
    singleton_correct_count = 0
    match_scores = []

    for s1_id, true_list in ground_truth.items():
        pred_list = predictions.get(s1_id, [])
        score = compute_entity_f_beta(true_list, pred_list, beta=beta)
        scores.append(score)

        if len(true_list) == 0:
            singleton_true_count += 1
            if len(pred_list) == 0:
                singleton_correct_count += 1
        else:
            match_scores.append(score)

    macro_f05 = float(np.mean(scores)) if scores else 0.0
    singleton_acc = (singleton_correct_count / singleton_true_count) if singleton_true_count > 0 else 1.0
    match_f05 = float(np.mean(match_scores)) if match_scores else 0.0

    return {
        "macro_f05": macro_f05,
        "singleton_accuracy": singleton_acc,
        "match_macro_f05": match_f05,
        "total_entities": len(ground_truth),
        "total_singletons": singleton_true_count
    }
