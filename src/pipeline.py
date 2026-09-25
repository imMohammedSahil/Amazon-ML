import json
import time
from typing import Dict, List, Optional, Tuple
import polars as pl
import pandas as pd
from pathlib import Path

from src.config import (
    TRAIN_S1_PATH, TRAIN_S2_PATH, TRAIN_S3_PATH, TRAIN_GT_PATH,
    TEST_S1_PATH, TEST_S2_PATH, TEST_S3_PATH, OUTPUT_DIR, CACHE_DIR
)
from src.preprocessing.cleaner import DataPreprocessor
from src.blocking.blocker import MultiPassBlocker
from src.features.extractor import PairwiseFeatureExtractor
from src.models.ranker import EntityRanker
from src.decision.optimizer import DynamicThresholdOptimizer
from src.evaluation.metrics import evaluate_macro_f05


class EntityResolutionPipeline:
    """
    End-to-End Master Entity Resolution Pipeline.
    Encapsulates Preprocessing, Blocking, Feature Extraction, Model Training, Threshold Optimization, and Inference.
    """
    def __init__(
        self,
        top_k_candidates: int = 35,
        min_blocking_sim: float = 0.25,
        singleton_threshold: float = 0.35,
        match_threshold: float = 0.50
    ):
        self.preprocessor = DataPreprocessor()
        self.blocker = MultiPassBlocker(top_k=top_k_candidates, min_sim=min_blocking_sim)
        self.extractor = PairwiseFeatureExtractor()
        self.ranker = EntityRanker()
        self.optimizer = DynamicThresholdOptimizer(
            singleton_threshold=singleton_threshold,
            match_threshold=match_threshold
        )

    def load_ground_truth(self, gt_path: Path) -> Dict[str, List[str]]:
        """Loads and parses ground truth CSV into dictionary mapping s1_id -> list of target matched IDs."""
        gt_df = pl.read_csv(gt_path)
        gt_dict = {}

        # Detect target column (e.g. 'matched_ids', 'target', 'matches', etc.)
        cols = gt_df.columns
        target_col = [c for c in cols if c != "id"][0]

        for row in gt_df.select(["id", target_col]).iter_rows():
            s1_id = str(row[0])
            raw_target = row[1]

            if raw_target is None or raw_target == "" or str(raw_target).strip() == "[]":
                gt_dict[s1_id] = []
            else:
                # Handle comma separated, JSON list, or bracketed string
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
        """
        Executes complete local cross-validation on train data.
        1. Cleans train tables
        2. Generates candidate matches
        3. Extracts pairwise features and attaches binary ground-truth labels
        4. Trains GroupKFold LightGBM models
        5. Optimizes singleton & match thresholds
        6. Outputs official Macro F_0.5 evaluation metrics
        """
        print("=" * 70)
        print("RUNNING LOCAL CROSS-VALIDATION PIPELINE")
        print("=" * 70)
        start_time = time.time()

        print("[1/5] Loading and Preprocessing training data...")
        df_s1 = pl.read_csv(TRAIN_S1_PATH)
        if sample_size and sample_size < len(df_s1):
            print(f"  * Sampling {sample_size:,} S1 entities for rapid local CV")
            df_s1 = df_s1.head(sample_size)
            
        s1_clean = self.preprocessor.clean_table(df_s1, source_name="source1")
        
        df_s2 = pl.read_csv(TRAIN_S2_PATH)
        df_s3 = pl.read_csv(TRAIN_S3_PATH)
        s2_clean = self.preprocessor.clean_table(df_s2, source_name="source2")
        s3_clean = self.preprocessor.clean_table(df_s3, source_name="source3")
        satellites_clean = pl.concat([s2_clean, s3_clean])

        gt_dict = self.load_ground_truth(TRAIN_GT_PATH)
        # Filter GT to current S1 sample
        s1_ids_set = set(s1_clean["id"].to_list())
        gt_filtered = {k: v for k, v in gt_dict.items() if k in s1_ids_set}

        print("[2/5] Running Multi-Pass Candidate Blocking...")
        pairs_df = self.blocker.block_candidates(s1_clean, satellites_clean)
        blocking_metrics = self.blocker.evaluate_blocking_recall(pairs_df, gt_filtered)
        print(f"  * Candidate Match Recall: {blocking_metrics['candidate_match_recall']:.4f}")
        print(f"  * Entity Perfect Recall Rate: {blocking_metrics['entity_perfect_recall_rate']:.4f}")
        print(f"  * Total Candidate Pairs: {blocking_metrics['total_candidates_generated']:,}")
        print(f"  * Avg Candidates per S1: {blocking_metrics['avg_candidates_per_s1']:.2f}")

        print("[3/5] Extracting Pairwise Features...")
        features_df = self.extractor.extract_features(pairs_df, s1_clean, satellites_clean)

        # Attach Ground Truth Binary Labels (1 = true match, 0 = negative candidate)
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

        print("[4/5] Training GroupKFold LightGBM Models & Out-of-Fold Evaluation...")
        feature_cols = self.extractor.feature_names
        oof_probs, models = self.ranker.train_cv(
            features_df,
            feature_cols=feature_cols,
            label_col="label",
            group_col="s1_id"
        )
        oof_df = features_df.select(["s1_id", "candidate_id"]).with_columns(pl.Series("prob", oof_probs))

        # Save model checkpoint
        self.ranker.save(CACHE_DIR / "lgbm_ensemble.pkl")

        print("[5/5] Optimizing Decision Thresholds (Singleton + Match Selection)...")
        all_s1_ids = s1_clean["id"].to_list()
        opt_results = self.optimizer.optimize_thresholds(
            all_s1_ids=all_s1_ids,
            oof_pairs_with_probs=oof_df,
            ground_truth=gt_filtered
        )

        final_preds = self.optimizer.predict_matches(
            all_s1_ids,
            oof_df,
            singleton_th=opt_results["best_singleton_threshold"],
            match_th=opt_results["best_match_threshold"]
        )

        final_metrics = evaluate_macro_f05(gt_filtered, final_preds)

        elapsed = time.time() - start_time
        print("\n" + "=" * 70)
        print("LOCAL VALIDATION REPORT")
        print("=" * 70)
        print(f"  * Macro F_0.5 Score:       {final_metrics['macro_f05']:.5f}")
        print(f"  * Singleton Accuracy:     {final_metrics['singleton_accuracy']:.4f}")
        print(f"  * Non-Singleton Match F05:{final_metrics['match_macro_f05']:.5f}")
        print(f"  * Best Singleton Thresh:  {opt_results['best_singleton_threshold']:.3f}")
        print(f"  * Best Match Thresh:      {opt_results['best_match_threshold']:.3f}")
        print(f"  * Pipeline Execution Time: {elapsed:.2f} seconds")
        print("=" * 70)

        # Print Top Features
        importances = self.ranker.get_feature_importances()
        print("\nTop 8 Most Informative Features:")
        for feat, imp in list(importances.items())[:8]:
            print(f"  - {feat:25s}: {imp:.1f}")

        return final_metrics

    def run_submission(self, output_path: Optional[Path] = None) -> Path:
        """
        Executes end-to-end inference on the test dataset and exports the submission file.
        """
        print("=" * 70)
        print("GENERATING FINAL SUBMISSION ON TEST SET")
        print("=" * 70)

        out_file = output_path or (OUTPUT_DIR / "submission.csv")

        print("[1/4] Loading and Preprocessing test data...")
        df_test_s1 = pl.read_csv(TEST_S1_PATH)
        df_test_s2 = pl.read_csv(TEST_S2_PATH)
        df_test_s3 = pl.read_csv(TEST_S3_PATH)

        test_s1_clean = self.preprocessor.clean_table(df_test_s1, source_name="source1")
        test_s2_clean = self.preprocessor.clean_table(df_test_s2, source_name="source2")
        test_s3_clean = self.preprocessor.clean_table(df_test_s3, source_name="source3")
        test_satellites_clean = pl.concat([test_s2_clean, test_s3_clean])

        print("[2/4] Running Candidate Blocking on Test Set...")
        test_pairs_df = self.blocker.block_candidates(test_s1_clean, test_satellites_clean)
        print(f"  * Generated {len(test_pairs_df):,} test candidate pairs")

        print("[3/4] Extracting Pairwise Features...")
        test_features_df = self.extractor.extract_features(test_pairs_df, test_s1_clean, test_satellites_clean)

        print("[4/4] Predicting with Ensemble and Formatting Output...")
        probs = self.ranker.predict_proba(test_features_df)
        test_pairs_with_probs = test_features_df.select(["s1_id", "candidate_id"]).with_columns(
            pl.Series("prob", probs)
        )

        all_test_s1_ids = test_s1_clean["id"].to_list()
        predictions_dict = self.optimizer.predict_matches(all_test_s1_ids, test_pairs_with_probs)

        # Format submission DataFrame
        sub_rows = []
        for s1_id in all_test_s1_ids:
            matches = predictions_dict.get(s1_id, [])
            sub_rows.append({
                "id": s1_id,
                "matched_ids": str(matches) if matches else "[]"
            })

        sub_df = pl.DataFrame(sub_rows)
        sub_df.write_csv(out_file)
        print(f" Submission successfully generated and saved to: {out_file}")
        print(f" Total S1 Entities: {len(sub_df):,}")
        return out_file
