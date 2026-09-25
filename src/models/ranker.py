from typing import Dict, List, Tuple
import numpy as np
import polars as pl
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import GroupKFold
import joblib
from pathlib import Path

from src.config import LGBM_PARAMS, N_FOLDS, RANDOM_SEED


class EntityRanker:
    """
    Ensemble Pairwise Ranker combining LightGBM and CatBoost.
    Trained on candidate pairs using leak-free GroupKFold splits on S1 entity IDs.
    """
    def __init__(self, lgbm_params: dict = None, params: dict = None, use_catboost: bool = True):
        self.lgbm_params = lgbm_params or params or LGBM_PARAMS.copy()
        self.use_catboost = use_catboost
        self.lgbm_models: List[lgb.LGBMClassifier] = []
        self.cat_models: List[CatBoostClassifier] = []
        self.feature_names: List[str] = []

    def train_cv(
        self,
        features_df: pl.DataFrame,
        feature_cols: List[str],
        label_col: str = "label",
        group_col: str = "s1_id",
        n_folds: int = N_FOLDS
    ) -> Tuple[np.ndarray, List[lgb.LGBMClassifier]]:
        self.feature_names = feature_cols
        
        X = features_df.select(feature_cols).to_numpy()
        y = features_df[label_col].to_numpy()
        groups = features_df[group_col].to_numpy()

        oof_lgb_preds = np.zeros(len(features_df), dtype=np.float32)
        oof_cat_preds = np.zeros(len(features_df), dtype=np.float32)
        self.lgbm_models = []
        self.cat_models = []

        gkf = GroupKFold(n_splits=n_folds)

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            unique_train_classes = np.unique(y_train)
            if len(unique_train_classes) < 2:
                default_p = float(unique_train_classes[0]) if len(unique_train_classes) == 1 else 0.5
                oof_lgb_preds[val_idx] = default_p
                oof_cat_preds[val_idx] = default_p
                continue

            # Model 1: LightGBM
            lgb_model = lgb.LGBMClassifier(**self.lgbm_params)
            if set(np.unique(y_val)).issubset(set(unique_train_classes)):
                lgb_model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
                )
            else:
                lgb_model.fit(X_train, y_train)

            oof_lgb_preds[val_idx] = lgb_model.predict_proba(X_val)[:, 1]
            self.lgbm_models.append(lgb_model)

            # Model 2: CatBoost
            if self.use_catboost:
                cat_model = CatBoostClassifier(
                    iterations=400,
                    learning_rate=0.06,
                    depth=6,
                    loss_function="Logloss",
                    eval_metric="Logloss",
                    random_seed=RANDOM_SEED + fold,
                    verbose=False,
                    thread_count=-1
                )
                cat_model.fit(
                    X_train, y_train,
                    eval_set=(X_val, y_val),
                    early_stopping_rounds=40,
                    verbose=False
                )
                oof_cat_preds[val_idx] = cat_model.predict_proba(X_val)[:, 1]
                self.cat_models.append(cat_model)

        # Blended Out-Of-Fold probabilities
        if self.use_catboost and self.cat_models:
            oof_ensemble = 0.55 * oof_lgb_preds + 0.45 * oof_cat_preds
        else:
            oof_ensemble = oof_lgb_preds

        return oof_ensemble, self.lgbm_models

    def predict_proba(self, features_df: pl.DataFrame) -> np.ndarray:
        if not self.lgbm_models:
            raise ValueError("Model has not been trained yet. Call train_cv() or load() first.")
        
        X = features_df.select(self.feature_names).to_numpy()
        lgb_preds = np.zeros(len(features_df), dtype=np.float32)
        for model in self.lgbm_models:
            lgb_preds += model.predict_proba(X)[:, 1] / len(self.lgbm_models)

        if self.use_catboost and self.cat_models:
            cat_preds = np.zeros(len(features_df), dtype=np.float32)
            for model in self.cat_models:
                cat_preds += model.predict_proba(X)[:, 1] / len(self.cat_models)
            return 0.55 * lgb_preds + 0.45 * cat_preds

        return lgb_preds

    def get_feature_importances(self) -> Dict[str, float]:
        if not self.lgbm_models:
            return {}
        importances = np.zeros(len(self.feature_names), dtype=np.float32)
        for model in self.lgbm_models:
            importances += model.feature_importances_ / len(self.lgbm_models)
        
        return dict(sorted(zip(self.feature_names, importances), key=lambda x: x[1], reverse=True))

    def save(self, filepath: Path):
        joblib.dump({
            "lgbm_models": self.lgbm_models,
            "cat_models": self.cat_models,
            "feature_names": self.feature_names,
            "use_catboost": self.use_catboost
        }, filepath)

    def load(self, filepath: Path):
        data = joblib.load(filepath)
        self.lgbm_models = data["lgbm_models"]
        self.cat_models = data.get("cat_models", [])
        self.feature_names = data["feature_names"]
        self.use_catboost = data.get("use_catboost", True)
