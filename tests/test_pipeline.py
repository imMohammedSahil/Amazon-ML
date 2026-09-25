import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import pytest
import polars as pl
import numpy as np
from src.preprocessing.cleaner import DataPreprocessor, clean_name_field, clean_phone_field
from src.blocking.blocker import MultiPassBlocker
from src.features.extractor import PairwiseFeatureExtractor
from src.evaluation.metrics import compute_entity_f_beta, evaluate_macro_f05
from src.models.ranker import EntityRanker
from src.decision.optimizer import DynamicThresholdOptimizer


def test_metric_f05_singletons():
    # True singleton vs Pred singleton => 1.0
    assert compute_entity_f_beta([], []) == 1.0
    # True singleton vs Pred non-empty => 0.0
    assert compute_entity_f_beta([], ["s2_1"]) == 0.0
    # True non-empty vs Pred singleton => 0.0
    assert compute_entity_f_beta(["s2_1"], []) == 0.0
    # Exact match
    assert compute_entity_f_beta(["s2_1", "s3_1"], ["s2_1", "s3_1"]) == 1.0
    # Partial match
    score = compute_entity_f_beta(["s2_1", "s3_1"], ["s2_1"])
    assert 0.0 < score < 1.0


def test_preprocessing():
    preprocessor = DataPreprocessor()
    df = pl.DataFrame({
        "id": ["1", "2"],
        "name": ["Starbucks Coffee Inc.", "Café de Paris SARL"],
        "address": ["123 Main Street, Suite 400", "10 Rue de la Paix"],
        "city": ["Seattle", "Paris"],
        "state": ["WA", ""],
        "zip": ["98101", "75002"],
        "country": ["United States", "France"],
        "phone": ["+1 (206) 555-0199", "01.42.68.55.00"]
    })

    cleaned = preprocessor.clean_table(df, source_name="source1")
    assert "clean_name" in cleaned.columns
    assert "clean_full_address" in cleaned.columns
    assert cleaned["clean_name"][0] == "starbucks coffee"
    assert cleaned["clean_name"][1] == "cafe de paris"
    assert "inc" not in cleaned["clean_name"][0]
    assert "sarl" not in cleaned["clean_name"][1]


def test_e2e_synthetic_pipeline():
    # 1. Synthetic Data
    df_s1 = pl.DataFrame({
        "id": ["s1_1", "s1_2", "s1_3"],
        "name": ["Amazon Web Services Inc", "Blue Bottle Coffee LLC", "Acme Corporation"],
        "address": ["410 Terry Ave N", "315 Linden St", "100 Industrial Pkwy"],
        "city": ["Seattle", "San Francisco", "Austin"],
        "state": ["WA", "CA", "TX"],
        "zip": ["98109", "94102", "78701"],
        "country": ["US", "US", "US"],
        "phone": ["2065551234", "4155554321", "5125559876"]
    })

    df_sat = pl.DataFrame({
        "id": ["s2_1", "s2_2", "s3_1", "s3_x"],
        "source": ["source2", "source2", "source3", "source3"],
        "name": ["AWS Cloud Services", "Blue Bottle Coffee", "Amazon Web Services", "Completely Unrelated Inc"],
        "address": ["410 Terry Ave North", "315 Linden Street", "Terry Avenue Seattle", "999 Far Away St"],
        "city": ["Seattle", "San Francisco", "Seattle", "New York"],
        "state": ["WA", "CA", "WA", "NY"],
        "zip": ["98109", "94102", "98109", "10001"],
        "country": ["US", "US", "US", "US"],
        "phone": ["2065551234", "4155554321", "2065551234", "2125550000"]
    })

    preprocessor = DataPreprocessor()
    s1_clean = preprocessor.clean_table(df_s1, source_name="source1")
    sat_clean = preprocessor.clean_table(df_sat)

    # 2. Blocking
    blocker = MultiPassBlocker(top_k=5, min_sim=0.1)
    pairs_df = blocker.block_candidates(s1_clean, sat_clean)
    assert len(pairs_df) > 0

    # 3. Features
    extractor = PairwiseFeatureExtractor()
    features_df = extractor.extract_features(pairs_df, s1_clean, sat_clean)
    assert len(extractor.feature_names) > 5

    # 4. Ground Truth Labels
    gt = {
        "s1_1": ["s2_1", "s3_1"],
        "s1_2": ["s2_2"],
        "s1_3": []  # Singleton
    }

    gt_pair_set = set()
    for s1_id, match_list in gt.items():
        for cand_id in match_list:
            gt_pair_set.add((s1_id, cand_id))

    labels = [
        1 if (row[0], row[1]) in gt_pair_set else 0
        for row in features_df.select(["s1_id", "candidate_id"]).iter_rows()
    ]
    features_df = features_df.with_columns(pl.Series("label", labels))

    # 5. Model
    ranker = EntityRanker(params={"n_estimators": 10, "min_child_samples": 1, "verbose": -1, "n_jobs": 1})
    oof_probs, models = ranker.train_cv(
        features_df,
        feature_cols=extractor.feature_names,
        label_col="label",
        group_col="s1_id",
        n_folds=2
    )
    assert len(oof_probs) == len(features_df)

    # 6. Decision & Dynamic Thresholds
    optimizer = DynamicThresholdOptimizer()
    oof_df = features_df.select(["s1_id", "candidate_id"]).with_columns(pl.Series("prob", oof_probs))
    preds = optimizer.predict_matches(s1_clean["id"].to_list(), oof_df)
    
    metrics = evaluate_macro_f05(gt, preds)
    assert "macro_f05" in metrics
    assert metrics["total_singletons"] == 1
