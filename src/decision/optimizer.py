from typing import Dict, List, Tuple, Optional
import polars as pl
import numpy as np
from src.evaluation.metrics import evaluate_macro_f05


class DynamicThresholdOptimizer:
    """
    Optimizes singleton gating and candidate selection thresholds to maximize Macro F_0.5.
    Applies relative probability margins to eliminate false-positive satellite candidates.
    """
    def __init__(
        self,
        singleton_threshold: float = 0.45,
        match_threshold: float = 0.55,
        relative_margin: float = 0.55
    ):
        self.singleton_threshold = singleton_threshold
        self.match_threshold = match_threshold
        self.relative_margin = relative_margin

    def predict_matches(
        self,
        all_s1_ids: List[str],
        pairs_with_probs: pl.DataFrame,
        singleton_th: Optional[float] = None,
        match_th: Optional[float] = None,
        rel_margin: Optional[float] = None
    ) -> Dict[str, List[str]]:
        """
        Generates final match dictionary mapping s1_id -> list of matched candidate IDs.
        Applies:
          1. Singleton Gate: If max_p < singleton_th -> returns []
          2. Absolute Threshold: Candidate prob >= match_th
          3. Relative Margin: Candidate prob >= max_p * rel_margin (protects Precision)
        """
        s_th = singleton_th if singleton_th is not None else self.singleton_threshold
        m_th = match_th if match_th is not None else self.match_threshold
        r_margin = rel_margin if rel_margin is not None else self.relative_margin

        predictions: Dict[str, List[str]] = {s1_id: [] for s1_id in all_s1_ids}

        if len(pairs_with_probs) == 0:
            return predictions

        grouped = pairs_with_probs.group_by("s1_id").agg([
            pl.col("candidate_id"),
            pl.col("prob")
        ])

        for row in grouped.iter_rows():
            s1_id, candidates, probs = row
            probs_arr = np.array(probs, dtype=np.float32)
            max_p = float(np.max(probs_arr)) if len(probs_arr) > 0 else 0.0

            # Singleton Gate
            if max_p < s_th:
                predictions[s1_id] = []
                continue

            # Candidate Selection with absolute & relative margin
            min_prob_cutoff = max(m_th, max_p * r_margin)
            selected = [
                cand for cand, p in zip(candidates, probs)
                if p >= min_prob_cutoff
            ]

            # If max_p >= s_th but none passed cutoff, keep the single best match
            if not selected and max_p >= s_th:
                best_idx = int(np.argmax(probs_arr))
                selected = [candidates[best_idx]]

            predictions[s1_id] = selected

        return predictions

    def optimize_thresholds(
        self,
        all_s1_ids: List[str],
        oof_pairs_with_probs: pl.DataFrame,
        ground_truth: Dict[str, List[str]],
        singleton_range: Tuple[float, float, int] = (0.35, 0.65, 7),
        match_range: Tuple[float, float, int] = (0.45, 0.75, 7),
        relative_margin_range: Tuple[float, float, int] = (0.40, 0.70, 4)
    ) -> Dict[str, float]:
        """
        Grid-searches singleton, match, and relative margin thresholds to maximize Macro F_0.5.
        """
        s_vals = np.linspace(singleton_range[0], singleton_range[1], singleton_range[2])
        m_vals = np.linspace(match_range[0], match_range[1], match_range[2])
        r_vals = np.linspace(relative_margin_range[0], relative_margin_range[1], relative_margin_range[2])

        best_score = -1.0
        best_s_th = self.singleton_threshold
        best_m_th = self.match_threshold
        best_r_margin = self.relative_margin

        for s_th in s_vals:
            for m_th in m_vals:
                if m_th < s_th:
                    continue
                for r_margin in r_vals:
                    preds = self.predict_matches(
                        all_s1_ids,
                        oof_pairs_with_probs,
                        singleton_th=s_th,
                        match_th=m_th,
                        rel_margin=r_margin
                    )

                    eval_res = evaluate_macro_f05(ground_truth, preds)
                    score = eval_res["macro_f05"]

                    if score > best_score:
                        best_score = score
                        best_s_th = float(s_th)
                        best_m_th = float(m_th)
                        best_r_margin = float(r_margin)

        self.singleton_threshold = best_s_th
        self.match_threshold = best_m_th
        self.relative_margin = best_r_margin

        return {
            "best_macro_f05": best_score,
            "best_singleton_threshold": best_s_th,
            "best_match_threshold": best_m_th,
            "best_relative_margin": best_r_margin
        }
