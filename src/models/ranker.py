from tqdm.auto import tqdm
from typing import Dict, List, Tuple
import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
import joblib
from pathlib import Path

from src.config import LGBM_PARAMS, N_FOLDS, RANDOM_SEED


class EntityRanker:
    """
    Pairwise gradient-boosted decision tree ranker for business entity matching.
    Trained on candidate pairs to predict match probability P(Candidate matches S1).
    """
    def __init__(self, params: dict = None):
        self.params = params or LGBM_PARAMS.copy()
        self.models: List[lgb.LGBMClassifier] = []
        self.feature_names: List[str] = []

    def train_cv(
        self,
        features_df: pl.DataFrame,
        feature_cols: List[str],
        label_col: str = "label",
        group_col: str = "s1_id",
        n_folds: int = N_FOLDS
    ) -> Tuple[np.ndarray, List[lgb.LGBMClassifier]]:
        """
        Trains an ensemble of LightGBM models across leak-free GroupKFold splits on S1 IDs.
        Returns out-of-fold probability predictions and list of trained models.
        """
        self.feature_names = feature_cols
        
        # Extract numpy arrays
        X = features_df.select(feature_cols).to_numpy()
        y = features_df[label_col].to_numpy()
        groups = features_df[group_col].to_numpy()

        oof_preds = np.zeros(len(features_df), dtype=np.float32)
        self.models = []

        gkf = GroupKFold(n_splits=n_folds)
        folds = list(gkf.split(X, y, groups))

        for fold, (train_idx, val_idx) in enumerate(tqdm(folds, desc="Training folds", unit="fold")):
                X_train, y_train = X[train_idx], y[train_idx]
                X_val, y_val = X[val_idx], y[val_idx]

                model = lgb.LGBMClassifier(**self.params)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=50, verbose=False),
                        lgb.log_evaluation(period=50)
            ]
        )

                val_preds = model.predict_proba(X_val)[:, 1]
                oof_preds[val_idx] = val_preds
                self.models.append(model)

        return oof_preds, self.models

    def predict_proba(self, features_df: pl.DataFrame) -> np.ndarray:
        """
        Predicts match probabilities using ensemble average across all trained fold models.
        """
        if not self.models:
            raise ValueError("Model has not been trained yet. Call train_cv() or load() first.")
        
        X = features_df.select(self.feature_names).to_numpy()
        preds = np.zeros(len(features_df), dtype=np.float32)

        for model in self.models:
            preds += model.predict_proba(X)[:, 1] / len(self.models)

        return preds

    def get_feature_importances(self) -> Dict[str, float]:
        """Returns mean feature importances across all trained folds."""
        if not self.models:
            return {}
        importances = np.zeros(len(self.feature_names), dtype=np.float32)
        for model in self.models:
            importances += model.feature_importances_ / len(self.models)
        
        return dict(sorted(zip(self.feature_names, importances), key=lambda x: x[1], reverse=True))

    def save(self, filepath: Path):
        """Saves trained models and feature names to disk."""
        joblib.dump({"models": self.models, "feature_names": self.feature_names}, filepath)

    def load(self, filepath: Path):
        """Loads trained models and feature names from disk."""
        data = joblib.load(filepath)
        self.models = data["models"]
        self.feature_names = data["feature_names"]
