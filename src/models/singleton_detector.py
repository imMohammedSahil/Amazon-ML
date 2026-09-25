from typing import Dict, List, Tuple
import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.model_selection import KFold
import joblib
from pathlib import Path


class SingletonDetector:
    """
    Tier-1 Dedicated Singleton Gating Classifier.
    Predicts whether an anchor entity is an isolated singleton with zero satellite matches.
    """
    def __init__(self):
        self.models: List[lgb.LGBMClassifier] = []
        self.feature_names: List[str] = []

    def build_entity_features(
        self,
        all_s1_ids: List[str],
        features_df: pl.DataFrame
    ) -> pl.DataFrame:
        """
        Aggregates candidate-level features to entity-level summary statistics.
        """
        if len(features_df) == 0:
            return pl.DataFrame({
                "s1_id": all_s1_ids,
                "agg_max_tfidf": np.zeros(len(all_s1_ids), dtype=np.float32),
                "agg_max_name_jw": np.zeros(len(all_s1_ids), dtype=np.float32),
                "agg_max_addr_jw": np.zeros(len(all_s1_ids), dtype=np.float32),
                "agg_cand_count": np.zeros(len(all_s1_ids), dtype=np.float32)
            })

        agg_df = features_df.group_by("s1_id").agg([
            pl.col("feat_blocking_tfidf_sim").max().alias("agg_max_tfidf"),
            pl.col("feat_name_jw").max().alias("agg_max_name_jw"),
            pl.col("feat_addr_jw").max().alias("agg_max_addr_jw"),
            pl.col("feat_name_ratio").max().alias("agg_max_name_ratio"),
            pl.col("feat_street_num_match").max().alias("agg_max_street_match"),
            pl.len().alias("agg_cand_count")
        ])

        # Left join onto all S1 IDs so singletons with 0 candidates get 0 values
        all_s1_df = pl.DataFrame({"s1_id": all_s1_ids})
        entity_df = all_s1_df.join(agg_df, on="s1_id", how="left").fill_null(0.0)

        self.feature_names = [c for c in entity_df.columns if c.startswith("agg_")]
        return entity_df

    def train_cv(
        self,
        entity_features_df: pl.DataFrame,
        ground_truth: Dict[str, List[str]],
        n_folds: int = 5
    ) -> np.ndarray:
        s1_ids = entity_features_df["s1_id"].to_list()
        y = np.array([1 if len(ground_truth.get(sid, [])) == 0 else 0 for sid in s1_ids], dtype=np.int32)
        X = entity_features_df.select(self.feature_names).to_numpy()

        oof_probs = np.zeros(len(s1_ids), dtype=np.float32)
        self.models = []

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            model = lgb.LGBMClassifier(
                n_estimators=150,
                learning_rate=0.05,
                num_leaves=31,
                random_state=42 + fold,
                verbose=-1,
                n_jobs=-1
            )
            model.fit(X_train, y_train)

            oof_probs[val_idx] = model.predict_proba(X_val)[:, 1]
            self.models.append(model)

        return oof_probs

    def predict_singleton_prob(self, entity_features_df: pl.DataFrame) -> np.ndarray:
        if not self.models:
            raise ValueError("SingletonDetector has not been trained yet.")
        X = entity_features_df.select(self.feature_names).to_numpy()
        probs = np.zeros(len(entity_features_df), dtype=np.float32)
        for model in self.models:
            probs += model.predict_proba(X)[:, 1] / len(self.models)
        return probs

    def save(self, filepath: Path):
        joblib.dump({"models": self.models, "feature_names": self.feature_names}, filepath)

    def load(self, filepath: Path):
        data = joblib.load(filepath)
        self.models = data["models"]
        self.feature_names = data["feature_names"]
