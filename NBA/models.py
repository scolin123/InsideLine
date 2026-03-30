"""
models.py — MODULE 3: ModelTrainer
Trains two models:
  • XGBoost  → regresses on Points scored (used for totals & spread)
  • LightGBM → classifies Win / Loss (used for Moneyline)

Uses TimeSeriesSplit to respect temporal ordering.
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, accuracy_score, log_loss

log = logging.getLogger(__name__)


class ModelTrainer:
    """
    Trains two models:
      • XGBoost  → regresses on Points scored (used for totals & spread)
      • LightGBM → classifies Win / Loss (used for Moneyline)

    Uses TimeSeriesSplit to respect temporal ordering.
    """

    def __init__(self, n_splits: int = 5):
        self.n_splits = n_splits
        self.xgb_model: Optional[xgb.XGBRegressor]   = None
        self.lgb_model: Optional[lgb.LGBMClassifier] = None
        self.tscv      = TimeSeriesSplit(n_splits=n_splits)
        self.cv_results: dict = {}

    # ── Public ────────────────────────────────────────────────────────────────
    def train(
        self,
        X: pd.DataFrame,
        y_pts: np.ndarray,
        y_win: np.ndarray,
    ) -> None:
        """Train both models with cross-validated evaluation."""
        log.info("Training XGBoost (points regression) …")
        self.xgb_model = self._train_xgb(X, y_pts)

        log.info("Training LightGBM (win classification) …")
        self.lgb_model = self._train_lgb(X, y_win)

    def predict_pts(self, X: pd.DataFrame) -> np.ndarray:
        """Predict expected points for each team-row."""
        self._check_trained()
        return self.xgb_model.predict(X)

    def predict_win_prob(self, X: pd.DataFrame) -> np.ndarray:
        """Predict win probability for each team-row."""
        self._check_trained()
        return self.lgb_model.predict_proba(X)[:, 1]

    # ── Private ───────────────────────────────────────────────────────────────
    def _train_xgb(self, X: pd.DataFrame, y: np.ndarray) -> xgb.XGBRegressor:
        params = dict(
            n_estimators=600,
            max_depth=5,
            learning_rate=0.04,
            subsample=0.80,
            colsample_bytree=0.75,
            reg_alpha=0.1,
            reg_lambda=1.5,
            objective="reg:squarederror",
            eval_metric="mae",
            n_jobs=-1,
            random_state=42,
            verbosity=0,
        )
        model     = xgb.XGBRegressor(**params)
        mae_folds = []

        for fold, (tr, val) in enumerate(self.tscv.split(X), 1):
            model.fit(
                X.iloc[tr], y[tr],
                eval_set=[(X.iloc[val], y[val])],
                verbose=False,
            )
            preds = model.predict(X.iloc[val])
            mae   = mean_absolute_error(y[val], preds)
            mae_folds.append(mae)
            log.info(f"  XGB fold {fold}/{self.n_splits}  MAE={mae:.3f}")

        self.cv_results["xgb_mae"] = np.mean(mae_folds)
        log.info(f"  XGB CV-MAE (avg): {self.cv_results['xgb_mae']:.3f} pts")

        # Final fit on all data
        model.fit(X, y, verbose=False)
        return model

    def _train_lgb(self, X: pd.DataFrame, y: np.ndarray) -> lgb.LGBMClassifier:
        params = dict(
            n_estimators=600,
            max_depth=5,
            learning_rate=0.04,
            num_leaves=31,
            subsample=0.80,
            colsample_bytree=0.75,
            reg_alpha=0.1,
            reg_lambda=1.5,
            objective="binary",
            metric="binary_logloss",
            n_jobs=-1,
            random_state=42,
            verbose=-1,
        )
        model      = lgb.LGBMClassifier(**params)
        acc_folds, ll_folds = [], []

        for fold, (tr, val) in enumerate(self.tscv.split(X), 1):
            model.fit(
                X.iloc[tr], y[tr],
                eval_set=[(X.iloc[val], y[val])],
            )
            proba = model.predict_proba(X.iloc[val])[:, 1]
            pred  = (proba >= 0.5).astype(int)
            acc   = accuracy_score(y[val], pred)
            ll    = log_loss(y[val], proba)
            acc_folds.append(acc)
            ll_folds.append(ll)
            log.info(f"  LGB fold {fold}/{self.n_splits}  Acc={acc:.3f}  LogLoss={ll:.4f}")

        self.cv_results["lgb_acc"] = np.mean(acc_folds)
        self.cv_results["lgb_ll"]  = np.mean(ll_folds)
        log.info(f"  LGB CV-Acc: {self.cv_results['lgb_acc']:.3f}   CV-LogLoss: {self.cv_results['lgb_ll']:.4f}")

        model.fit(X, y)
        return model

    def _check_trained(self) -> None:
        if self.xgb_model is None or self.lgb_model is None:
            raise RuntimeError("Models not trained. Call .train() first.")
