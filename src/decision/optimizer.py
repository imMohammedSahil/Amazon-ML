from typing import Dict, List, Tuple
import polars as pl
import numpy as np
from src.evaluation.metrics import evaluate_macro_f05


class DynamicThresholdOptimizer:
    """
    Optimizes singleton gating and candidate selection thresholds to maximize Macro F_0.5.
    """
    def __init__(
        self,
        singleton_threshold: float = 0.35,
        match_threshold: float = 0.50
    ):
        self.singleton_threshold = singleton_threshold
        self.match_threshold = match_threshold

    def predict_matches(
        self,
        all_s1_ids: List[str],
        pairs_with_probs: pl.DataFrame,
        singleton_th: float = None,
        match_th: float = None
    ) -> Dict[str, List[str]]:
        """
        Generates final match dictionary mapping s1_id -> list of matched candidate IDs.
        Handles singleton prediction when no candidates are strong matches.
        """
        s_th = singleton_th if singleton_th is not None else self.singleton_threshold
        m_th = match_th if match_th is not None else self.match_threshold

        # Initialize all entities with empty list (singletons by default)
        predictions: Dict[str, List[str]] = {s1_id: [] for s1_id in all_s1_ids}

        # Filter pairs where probability exceeds match threshold
        # and group by s1_id
        if len(pairs_with_probs) == 0:
            return predictions

        # Group candidates by s1_id and collect with probabilities
        grouped = pairs_with_probs.group_by("s1_id").agg([
            pl.col("candidate_id"),
            pl.col("prob")
        ])

        for row in grouped.iter_rows():
            s1_id, candidates, probs = row
            probs_arr = np.array(probs)
            max_p = np.max(probs_arr) if len(probs_arr) > 0 else 0.0

            # Singleton Gate: If maximum confidence is below singleton threshold, return []
            if max_p < s_th:
                predictions[s1_id] = []
                continue

            # Select all candidates exceeding match threshold
            selected = [
                cand for cand, p in zip(candidates, probs)
                if p >= m_th
            ]

            # If max_p >= s_th but no candidate exceeds m_th, select the single best candidate
            if not selected and max_p >= s_th:
                best_idx = np.argmax(probs_arr)
                selected = [candidates[best_idx]]

            predictions[s1_id] = selected

        return predictions

    def optimize_thresholds(
        self,
        all_s1_ids: List[str],
        oof_pairs_with_probs: pl.DataFrame,
        ground_truth: Dict[str, List[str]],
        singleton_range: Tuple[float, float, int] = (0.20, 0.55, 8),
        match_range: Tuple[float, float, int] = (0.35, 0.70, 8)
    ) -> Dict[str, float]:
        """
        Grid-searches singleton and match thresholds to find the global optimum for Macro F_0.5.
        """
        s_vals = np.linspace(singleton_range[0], singleton_range[1], singleton_range[2])
        m_vals = np.linspace(match_range[0], match_range[1], match_range[2])

        best_score = -1.0
        best_s_th = self.singleton_threshold
        best_m_th = self.match_threshold

        for s_th in s_vals:
            for m_th in m_vals:
                if m_th < s_th:
                    continue  # Match threshold should generally be >= singleton threshold

                preds = self.predict_matches(
                    all_s1_ids,
                    oof_pairs_with_probs,
                    singleton_th=s_th,
                    match_th=m_th
                )

                eval_res = evaluate_macro_f05(ground_truth, preds)
                score = eval_res["macro_f05"]

                if score > best_score:
                    best_score = score
                    best_s_th = float(s_th)
                    best_m_th = float(m_th)

        self.singleton_threshold = best_s_th
        self.match_threshold = best_m_th

        return {
            "best_macro_f05": best_score,
            "best_singleton_threshold": best_s_th,
            "best_match_threshold": best_m_th
        }
