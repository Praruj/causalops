from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance


def train_model(X: pd.DataFrame, y: np.ndarray, seed: int = 0) -> HistGradientBoostingClassifier:
    """Gradient-boosted trees. They accept missing values, so a null spike reaches
    the model as it would in production instead of crashing the scoring job."""
    model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, random_state=seed)
    return model.fit(X, y)


def predict(model, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def importances(model, X: pd.DataFrame, y: np.ndarray, seed: int = 0) -> dict[str, float]:
    """Permutation importance on log loss, scaled so the top feature is 1."""
    r = permutation_importance(
        model, X, y, scoring="neg_log_loss", n_repeats=3, random_state=seed
    )
    imp = np.clip(r.importances_mean, 0, None)
    imp = imp / imp.max() if imp.max() > 0 else imp
    return dict(zip(X.columns, imp.tolist(), strict=True))
