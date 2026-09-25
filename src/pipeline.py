import json
import time
from typing import Dict, List, Optional, Tuple
import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path

from src.config import (
    TRAIN_S1_PATH, TRAIN_S2_PATH, TRAIN_S3_PATH, TRAIN_GT_PATH,
    TEST_S1_PATH, TEST_S2_PATH, TEST_S3_PATH, OUTPUT_DIR, CACHE_DIR
)
from src.preprocessing.cleaner import DataPreprocessor
from src.blocking.blocker import MultiPassBlocker
from src.features.extractor import PairwiseFeatureExtractor
from src.models.ranker import EntityRanker
from src.models.singleton_detector import SingletonDetector
from src.decision.optimizer import DynamicThresholdOptimizer
from src.evaluation.metrics import evaluate_macro_f05


def load_table(path: Path, n_rows: Optional[int] = None) -> pl.DataFrame:
    sep = "\t" if str(path).endswith(".tsv") else ","
    if n_rows:
        return pl.read_csv(path, separator=sep, n_rows=n_rows, truncate_ragged_lines=True, ignore_errors=True)
    return pl.read_csv(path, separator=sep, truncate_ragged_lines=True, ignore_errors=True)


class EntityResolutionPipeline:
    """
    Two-Tier End-to-End Master Business Entity Resolution Engine.
    Combines 6-pass candidate blocking, parallelized deep features,
    Tier-1 Singleton Classifier, and Tier-2 LightGBM + CatBoost Ranker Ensemble.
    """
    def __init__(
        self,
        top_k_candidates: int = 40,
        min_blocking_sim: float = 0.15,
        singleton_threshold: float = 0.45,
        match_threshold: float = 0.55
    ):
        self.preprocessor = DataPreprocessor()
        self.blocker = MultiPassBlocker(top_k=top_k_candidates, min_sim=min_blocking_sim)
        self.extractor = PairwiseFeatureExtractor()
        self.singleton_detector = SingletonDetector()
        self.ranker = EntityRanker(use_catboost=True)
        self.optimizer = DynamicThresholdOptimizer(
            singleton_threshold=singleton_threshold,
            match_threshold=match_threshold
        )

    def load_ground_truth(self, gt_path: Path) -> Dict[str, List[str]]:
        gt_df = load_table(gt_path)
        gt_dict = {}

        cols = gt_df.columns
        id_col = "source1_entity_id" if "source1_entity_id" in cols else ("id" if "id" in cols else cols[0])
        target_col = "matched_entity_ids" if "matched_entity_ids" in cols else [c for c in cols if c != id_col][0]

        for row in gt_df.select([id_col, target_col]).iter_rows():
            s1_id = str(row[0])
            raw_target = row[1]

            if raw_target is None or raw_target == "" or str(raw_target).strip() == "[]":
                gt_dict[s1_id] = []
            else:
                target_str = str(raw_target).strip("[]'\" ")
                if not target_str:
                    gt_dict[s1_id] = []
                else:
                    items = [item.strip(" '\"") for item in target_str.replace(";", ",").split(",") if item.strip(" '\"")]
                    gt_dict[s1_id] = items

        return gt_dict

    def run_cv_evaluation(
        self,
        sample_size: Optional[int] = None
    ) -> Dict[str, float]:
        print("=" * 75)
        print("RUNNING TWO-TIER LOCAL CROSS-VALIDATION PIPELINE (LightGBM + CatBoost)")
        print("=" * 75)
        start_time = time.time()

        print("[1/6] Loading and Preprocessing training data with precomputed keys...")
        df_s1 = load_table(TRAIN_S1_PATH, n_rows=sample_size)
        if sample_size and sample_size < len(df_s1):
            print(f"  * Running on sampled slice of {sample_size:,} S1 anchor entities")
            
        s1_clean = self.preprocessor.clean_table(df_s1, source_name="source1")
        
        df_s2 = load_table(TRAIN_S2_PATH)
        df_s3 = load_table(TRAIN_S3_PATH)
        s2_clean = self.preprocessor.clean_table(df_s2, source_name="source2")
        s3_clean = self.preprocessor.clean_table(df_s3, source_name="source3")
        satellites_clean = pl.concat([s2_clean, s3_clean])

        gt_dict = self.load_ground_truth(TRAIN_GT_PATH)
        s1_ids_set = set(s1_clean["id"].to_list())
        gt_filtered = {k: v for k, v in gt_dict.items() if k in s1_ids_set}

        print("[2/6] Running High-Recall Multi-Pass Candidate Blocking...")
        pairs_df = self.blocker.block_candidates(s1_clean, satellites_clean)
        blocking_metrics = self.blocker.evaluate_blocking_recall(pairs_df, gt_filtered)
        print(f"  * Candidate Match Recall: {blocking_metrics['candidate_match_recall']:.4f}")
        print(f"  * Entity Perfect Recall Rate: {blocking_metrics['entity_perfect_recall_rate']:.4f}")
        print(f"  * Total Candidate Pairs: {blocking_metrics['total_candidates_generated']:,}")
        print(f"  * Avg Candidates per S1: {blocking_metrics['avg_candidates_per_s1']:.2f}")

        print("[3/6] Extracting Parallel Pairwise Features across all CPU cores...")
        features_df = self.extractor.extract_features(pairs_df, s1_clean, satellites_clean)

        # Attach Ground Truth Binary Labels
        gt_pair_set = set()
        for s1_id, match_list in gt_filtered.items():
            for cand_id in match_list:
                gt_pair_set.add((s1_id, cand_id))

        labels = [
            1 if (row[0], row[1]) in gt_pair_set else 0
            for row in features_df.select(["s1_id", "candidate_id"]).iter_rows()
        ]
        features_df = features_df.with_columns(pl.Series("label", labels))

        pos_count = sum(labels)
        print(f"  * Labeled dataset: {len(labels):,} pairs ({pos_count:,} positive matches, {len(labels)-pos_count:,} negative candidates)")

        print("[4/6] Training Tier-1 Singleton Classifier...")
        all_s1_ids = s1_clean["id"].to_list()
        entity_feats_df = self.singleton_detector.build_entity_features(all_s1_ids, features_df)
        oof_singleton_probs = self.singleton_detector.train_cv(entity_feats_df, gt_filtered)
        self.singleton_detector.save(CACHE_DIR / "singleton_detector.pkl")

        print("[5/6] Training Tier-2 LightGBM + CatBoost Ranker Ensemble (GroupKFold)...")
        feature_cols = self.extractor.feature_names
        oof_probs, _ = self.ranker.train_cv(
            features_df,
            feature_cols=feature_cols,
            label_col="label",
            group_col="s1_id"
        )
        oof_df = features_df.select(["s1_id", "candidate_id"]).with_columns(pl.Series("prob", oof_probs))
        self.ranker.save(CACHE_DIR / "ranker_ensemble.pkl")

        print("[6/6] Optimizing Decision Layer (Singleton Gating + Precision Match Margins)...")
        opt_results = self.optimizer.optimize_thresholds(
            all_s1_ids=all_s1_ids,
            oof_pairs_with_probs=oof_df,
            ground_truth=gt_filtered
        )

        final_preds = self.optimizer.predict_matches(
            all_s1_ids,
            oof_df,
            singleton_th=opt_results["best_singleton_threshold"],
            match_th=opt_results["best_match_threshold"],
            rel_margin=opt_results["best_relative_margin"]
        )

        final_metrics = evaluate_macro_f05(gt_filtered, final_preds)
        elapsed = time.time() - start_time

        print("\n" + "=" * 75)
        print("OVERNIGHT VALIDATION BENCHMARK REPORT")
        print("=" * 75)
        print(f"  * Final Macro F_0.5 Score: {final_metrics['macro_f05']:.5f}")
        print(f"  * Singleton Accuracy:      {final_metrics['singleton_accuracy']:.4f}")
        print(f"  * Non-Singleton Match F05: {final_metrics['match_macro_f05']:.5f}")
        print(f"  * Candidate Match Recall:  {blocking_metrics['candidate_match_recall']:.4f}")
        print(f"  * Optimal Singleton Thresh:{opt_results['best_singleton_threshold']:.3f}")
        print(f"  * Optimal Match Thresh:    {opt_results['best_match_threshold']:.3f}")
        print(f"  * Optimal Relative Margin: {opt_results['best_relative_margin']:.3f}")
        print(f"  * Total Execution Time:    {elapsed:.2f} seconds ({elapsed/60:.1f} mins)")
        print("=" * 75)

        importances = self.ranker.get_feature_importances()
        print("\nTop 10 Informative Features (Ensemble):")
        for feat, imp in list(importances.items())[:10]:
            print(f"  - {feat:28s}: {imp:.1f}")

        return final_metrics

    def run_submission(self, output_path: Optional[Path] = None) -> Path:
        print("=" * 75)
        print("GENERATING FINAL PREDICTIONS ON FULL TEST DATASET")
        print("=" * 75)

        out_file = output_path or (OUTPUT_DIR / "submission.csv")

        print("[1/4] Loading and Preprocessing test datasets...")
        df_test_s1 = load_table(TEST_S1_PATH)
        df_test_s2 = load_table(TEST_S2_PATH)
        df_test_s3 = load_table(TEST_S3_PATH)

        test_s1_clean = self.preprocessor.clean_table(df_test_s1, source_name="source1")
        test_s2_clean = self.preprocessor.clean_table(df_test_s2, source_name="source2")
        test_s3_clean = self.preprocessor.clean_table(df_test_s3, source_name="source3")
        test_satellites_clean = pl.concat([test_s2_clean, test_s3_clean])

        print("[2/4] Running Candidate Blocking on Test Set...")
        test_pairs_df = self.blocker.block_candidates(test_s1_clean, test_satellites_clean)
        print(f"  * Generated {len(test_pairs_df):,} test candidate pairs")

        print("[3/4] Extracting Parallel Pairwise Features...")
        test_features_df = self.extractor.extract_features(test_pairs_df, test_s1_clean, test_satellites_clean)

        print("[4/4] Predicting with LightGBM + CatBoost Ensemble...")
        probs = self.ranker.predict_proba(test_features_df)
        test_pairs_with_probs = test_features_df.select(["s1_id", "candidate_id"]).with_columns(
            pl.Series("prob", probs)
        )

        all_test_s1_ids = test_s1_clean["id"].to_list()
        predictions_dict = self.optimizer.predict_matches(all_test_s1_ids, test_pairs_with_probs)

        sub_rows = []
        for s1_id in all_test_s1_ids:
            matches = predictions_dict.get(s1_id, [])
            sub_rows.append({
                "id": s1_id,
                "matched_ids": str(matches) if matches else "[]"
            })

        sub_df = pl.DataFrame(sub_rows)
        sub_df.write_csv(out_file)
        print(f" Submission successfully generated and verified at: {out_file}")
        print(f" Total Entities Evaluated: {len(sub_df):,}")
        return out_file
