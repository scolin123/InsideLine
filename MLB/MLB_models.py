"""
MLB_models.py — MODULE 3: ModelTrainer (MLB)
Trains two models:
  • XGBoost  → regresses on Runs scored (used for totals & run-line)
  • LightGBM → classifies Win / Loss (used for Moneyline)

Uses TimeSeriesSplit to respect temporal ordering.

MLB-specific design notes vs. NBA version
──────────────────────────────────────────
• Pitcher Handedness (SP_HAND: 0=LHP, 1=RHP) is declared as a categorical
  feature in the LightGBM call.  XGBoost handles it naturally as a numeric
  binary split, but categorical_feature is passed to LGB so that the
  leaf-splitting logic treats it as an unordered category rather than an
  ordinal number.

• The run-scoring MAE target for XGBoost (~1.5–2.5 runs) is much smaller than
  the NBA equivalent (~12–18 pts), so the learning-rate, subsample, and
  regularisation parameters are tuned to avoid over-fitting on a tighter
  target distribution.

• CV metric logging surfaces Runs-MAE (XGBoost) and both Accuracy and
  Log-Loss (LightGBM) at each fold for transparency.
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

# Categorical features that LightGBM should treat as unordered categories.
_LGB_CATEGORICAL_FEATURES: list[str] = ["SP_HAND"]


class ModelTrainer:
    """
    Trains two models:
      • XGBoost  → regresses on Runs scored (used for totals & run-line)
      • LightGBM → classifies Win / Loss (used for Moneyline)

    Uses TimeSeriesSplit to respect temporal ordering and avoid look-ahead bias
    in the rolling feature matrix produced by MLB_features.FeatureEngine.

    Pitcher handedness
    ──────────────────
    SP_HAND (0=LHP, 1=RHP) is passed to LightGBM via the
    categorical_feature parameter.  When SP_HAND is absent from the feature
    matrix (e.g. older season data) the parameter is silently dropped — the
    model degrades gracefully rather than raising an error.
    """

    def __init__(self, n_splits: int = 5) -> None:
        self.n_splits:   int                         = n_splits
        self.xgb_model:  Optional[xgb.XGBRegressor]  = None
        self.lgb_model:  Optional[lgb.LGBMClassifier] = None
        self.tscv:       TimeSeriesSplit              = TimeSeriesSplit(n_splits=n_splits)
        self.cv_results: dict                         = {}

    # ── Public ────────────────────────────────────────────────────────────────
    def train(
        self,
        X:      pd.DataFrame,
        y_runs: np.ndarray,
        y_win:  np.ndarray,
    ) -> None:
        """
        Train both models with cross-validated evaluation.

        Parameters
        ──────────
        X      : feature matrix from FeatureEngine.get_feature_matrix()
        y_runs : actual runs scored per team-game row (XGBoost regression target)
        y_win  : binary win / loss per team-game row  (LightGBM classification target)
        """
        log.info("Training XGBoost (runs regression) …")
        self.xgb_model = self._train_xgb(X, y_runs)

        log.info("Training LightGBM (win classification) …")
        self.lgb_model = self._train_lgb(X, y_win)

    def predict_runs(self, X: pd.DataFrame) -> np.ndarray:
        """Predict expected runs for each team-row."""
        self._check_trained()
        return self.xgb_model.predict(X)                    # type: ignore[union-attr]

    def predict_win_prob(self, X: pd.DataFrame) -> np.ndarray:
        """Predict win probability for each team-row."""
        self._check_trained()
        return self.lgb_model.predict_proba(X)[:, 1]        # type: ignore[union-attr]

    # ── Private — XGBoost ────────────────────────────────────────────────────
    def _train_xgb(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
    ) -> xgb.XGBRegressor:
        """
        XGBoost regressor tuned for MLB run-scoring targets.

        Hyperparameter notes
        ─────────────────────
        • lower learning_rate (0.03) and higher n_estimators (700) vs NBA:
          run margins are much tighter so finer gradient steps help.
        • max_depth=4 keeps trees shallow — with ~40 features a depth-5 tree
          risks over-fitting matchup-specific quirks.
        • reg_lambda=2.0 / reg_alpha=0.15: slightly stronger L2/L1 vs NBA
          given the smaller target variance.
        """
        params = dict(
            n_estimators      = 700,
            max_depth         = 4,
            learning_rate     = 0.03,
            subsample         = 0.80,
            colsample_bytree  = 0.75,
            reg_alpha         = 0.15,
            reg_lambda        = 2.0,
            objective         = "reg:squarederror",
            eval_metric       = "mae",
            n_jobs            = -1,
            random_state      = 42,
            verbosity         = 0,
        )
        model      = xgb.XGBRegressor(**params)
        mae_folds: list[float] = []

        for fold, (tr, val) in enumerate(self.tscv.split(X), 1):
            model.fit(
                X.iloc[tr], y[tr],
                eval_set=[(X.iloc[val], y[val])],
                verbose=False,
            )
            preds = model.predict(X.iloc[val])
            mae   = mean_absolute_error(y[val], preds)
            mae_folds.append(mae)
            log.info(f"  XGB fold {fold}/{self.n_splits}  MAE={mae:.3f} runs")

        self.cv_results["xgb_mae"] = float(np.mean(mae_folds))
        log.info(
            f"  XGB CV-MAE (avg): {self.cv_results['xgb_mae']:.3f} runs"
        )

        # Final fit on full training set
        model.fit(X, y, verbose=False)
        return model

    # ── Private — LightGBM ───────────────────────────────────────────────────
    def _train_lgb(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
    ) -> lgb.LGBMClassifier:
        """
        LightGBM binary classifier for win probability.

        Pitcher handedness (SP_HAND) is declared as a categorical feature so
        that LightGBM handles the LHP/RHP split as an unordered category rather
        than inferring ordinal relationships from the 0/1 encoding.

        If SP_HAND is absent from the feature matrix the parameter is dropped
        gracefully without raising an error.
        """
        params = dict(
            n_estimators      = 700,
            max_depth         = 4,
            learning_rate     = 0.03,
            num_leaves        = 31,
            subsample         = 0.80,
            colsample_bytree  = 0.75,
            reg_alpha         = 0.15,
            reg_lambda        = 2.0,
            objective         = "binary",
            metric            = "binary_logloss",
            n_jobs            = -1,
            random_state      = 42,
            verbose           = -1,
        )
        model                     = lgb.LGBMClassifier(**params)
        acc_folds:  list[float]   = []
        ll_folds:   list[float]   = []

        # Resolve categorical feature names that are actually present in X.
        cat_features: list[str] = [
            c for c in _LGB_CATEGORICAL_FEATURES if c in X.columns
        ]
        fit_kwargs: dict = {}
        if cat_features:
            fit_kwargs["categorical_feature"] = cat_features
            log.info(
                f"  LGB: declaring categorical features → {cat_features}"
            )

        for fold, (tr, val) in enumerate(self.tscv.split(X), 1):
            model.fit(
                X.iloc[tr], y[tr],
                eval_set=[(X.iloc[val], y[val])],
                **fit_kwargs,
            )
            proba = model.predict_proba(X.iloc[val])[:, 1]
            pred  = (proba >= 0.5).astype(int)
            acc   = accuracy_score(y[val], pred)
            ll    = log_loss(y[val], proba)
            acc_folds.append(acc)
            ll_folds.append(ll)
            log.info(
                f"  LGB fold {fold}/{self.n_splits}  "
                f"Acc={acc:.3f}  LogLoss={ll:.4f}"
            )

        self.cv_results["lgb_acc"] = float(np.mean(acc_folds))
        self.cv_results["lgb_ll"]  = float(np.mean(ll_folds))
        log.info(
            f"  LGB CV-Acc: {self.cv_results['lgb_acc']:.3f}   "
            f"CV-LogLoss: {self.cv_results['lgb_ll']:.4f}"
        )

        # Final fit on full training set
        model.fit(X, y, **fit_kwargs)
        return model

    # ── Guard ─────────────────────────────────────────────────────────────────
    def _check_trained(self) -> None:
        if self.xgb_model is None or self.lgb_model is None:
            raise RuntimeError(
                "Models not trained. Call .train() first."
            )