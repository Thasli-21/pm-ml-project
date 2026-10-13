"""Inference helpers used by the Streamlit dashboard (and usable from notebooks)."""
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MODELS = Path(__file__).resolve().parents[1] / "models"


@lru_cache(maxsize=None)
def _load(name: str):
    return joblib.load(MODELS / f"{name}.joblib")


def predict_effort(text: str) -> float:
    """Estimated story points for a task description."""
    return float(np.clip(_load("effort_model").predict([text])[0], 1, 20))


def predict_complexity(text: str) -> dict:
    """Small / Medium / Large with class scores when the model supports them."""
    model = _load("complexity_model")
    label = str(model.predict([text])[0])
    out = {"label": label, "scores": {}}
    clf = model[-1]
    if hasattr(clf, "predict_proba"):
        probs = model.predict_proba([text])[0]
        out["scores"] = {c: float(p) for c, p in zip(clf.classes_, probs)}
    return out


def similar_tasks(text: str, k: int = 5) -> dict:
    """Most similar past tasks, plus the project (team proxy) that fits best."""
    rec = _load("recommender")
    X = rec["vec"].transform([text])
    dist, idx = rec["nn"].kneighbors(X, n_neighbors=10)
    meta = rec["meta"].iloc[idx[0]].copy()
    meta["similarity"] = 1 - dist[0]
    weights = meta.groupby("project")["similarity"].sum().sort_values(ascending=False)
    return {"tasks": meta.head(k).reset_index(drop=True),
            "team_scores": (weights / weights.sum()).round(3),
            "best_team": weights.index[0],
            "neighbour_avg_points": float(meta["storypoint"].mean())}


def delay_options() -> dict:
    art = _load("delay_risk")
    return {"categories": art["categories"], "num_cols": art["num_cols"],
            "defaults": art["num_defaults"], "num_max": art["num_max"]}


def predict_delay_and_risk(row: dict) -> dict:
    """row: dict with the delay model's input columns."""
    art = _load("delay_risk")
    X = pd.DataFrame([{c: row.get(c, art["num_defaults"].get(c, 0)) for c in art["feat_cols"]}])
    p = float(art["delay"].predict_proba(X)[0, 1])
    exp_days = float(np.expm1(max(art["severity"].predict(X)[0], 0)))
    raw = p * exp_days
    ref = art["risk_reference"]
    score = float(100 * np.searchsorted(ref, raw) / len(ref))
    q1, q2 = art["thresholds"]
    level = "High" if score >= q2 else "Medium" if score >= q1 else "Low"
    return {"delay_probability": p, "expected_delay_days_if_late": exp_days,
            "risk_score": score, "risk_level": level}
