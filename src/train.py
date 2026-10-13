"""Train and evaluate modules 1-5. Run:  python src/train.py

Outputs: models/*.joblib, reports/metrics.json, reports/figures/*.png
"""
import json
import time
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (ConfusionMatrixDisplay, f1_score,
                             mean_absolute_error, mean_squared_error,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, cross_val_score,
                                     train_test_split)
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC

import data as D

warnings.filterwarnings("ignore")
SEED = 42
ROOT = D.ROOT
MODELS = ROOT / "models"
FIGS = ROOT / "reports" / "figures"
MODELS.mkdir(exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)
metrics = {}


def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------
# Shared text data
# ----------------------------------------------------------------------------
sp = D.load_storypoints()
log(f"story point data: {sp.shape}, projects={sp.project.nunique()}")
# Some projects use very large point scales (max 100). Cap at 20 so a few extreme
# labels do not dominate the error (affects ~3.5% of tasks; documented in the report).
N_CAPPED = int((sp["storypoint"] > 20).sum())
sp["storypoint"] = sp["storypoint"].clip(upper=20)
sp["complexity"] = sp["storypoint"].map(D.complexity_label)
sp_train, sp_test = train_test_split(sp, test_size=0.2, random_state=SEED,
                                     stratify=sp["complexity"])


def tfidf(max_features=5000):
    return TfidfVectorizer(max_features=max_features, ngram_range=(1, 2),
                           min_df=3, sublinear_tf=True, stop_words="english")


# ----------------------------------------------------------------------------
# MODULE 1 - Effort estimation (regression on story points)
# ----------------------------------------------------------------------------
log("MODULE 1: effort estimation")
y_tr, y_te = sp_train["storypoint"].values, sp_test["storypoint"].values

# Baseline used in the literature: predict the median of the training set.
base = DummyRegressor(strategy="median").fit(sp_train[["text"]], y_tr)
p = base.predict(sp_test[["text"]])
m1 = {"Median baseline": {"MAE": mean_absolute_error(y_te, p), "RMSE": rmse(y_te, p)}}

models1 = {
    "Ridge (TF-IDF)": make_pipeline(tfidf(), Ridge(alpha=3.0)),
    "Random Forest (TF-IDF+SVD)": make_pipeline(
        tfidf(3000), TruncatedSVD(120, random_state=SEED),
        RandomForestRegressor(n_estimators=120, min_samples_leaf=5, max_features="sqrt",
                              n_jobs=-1, random_state=SEED)),
    "Gradient Boosting (TF-IDF+SVD)": make_pipeline(
        tfidf(3000), TruncatedSVD(120, random_state=SEED),
        HistGradientBoostingRegressor(max_iter=200, learning_rate=0.06,
                                      random_state=SEED)),
}
fitted1 = {}
for name, pipe in models1.items():
    pipe.fit(sp_train["text"], y_tr)
    pr = np.clip(pipe.predict(sp_test["text"]), 1, 20)
    m1[name] = {"MAE": mean_absolute_error(y_te, pr), "RMSE": rmse(y_te, pr)}
    fitted1[name] = (pipe, pr)
    log(f"  {name}: MAE={m1[name]['MAE']:.3f} RMSE={m1[name]['RMSE']:.3f}")

best1 = min(fitted1, key=lambda k: m1[k]["MAE"])
cv = -cross_val_score(models1[best1], sp["text"], sp["storypoint"], cv=3,
                      scoring="neg_mean_absolute_error", n_jobs=1)
m1["best_model"] = best1
m1["best_model_3fold_cv_MAE"] = {"mean": float(cv.mean()), "std": float(cv.std())}
# Ratio vs. baseline is the headline number for the report
m1["improvement_over_baseline_pct"] = float(
    100 * (1 - m1[best1]["MAE"] / m1["Median baseline"]["MAE"]))
metrics["effort_estimation"] = m1
log(f"  best={best1} CV MAE {cv.mean():.3f}±{cv.std():.3f}")

# Re-fit best on ALL data for deployment
best_pipe1 = models1[best1].fit(sp["text"], sp["storypoint"])
joblib.dump(best_pipe1, MODELS / "effort_model.joblib")

best_pred = fitted1[best1][1]
fig, ax = plt.subplots(figsize=(6, 4.5))
ax.scatter(y_te + np.random.RandomState(0).uniform(-.25, .25, len(y_te)), best_pred,
           s=6, alpha=.25)
ax.plot([0, 20], [0, 20], "r--", lw=1)
ax.set_xlim(0, 21); ax.set_ylim(0, 21)
ax.set_xlabel("Actual story points"); ax.set_ylabel("Predicted story points")
ax.set_title(f"Effort estimation - {best1}")
fig.tight_layout(); fig.savefig(FIGS / "effort_actual_vs_pred.png", dpi=130); plt.close(fig)

names = list(m1.keys())[:4]
fig, ax = plt.subplots(figsize=(6.5, 3.8))
ax.bar(names, [m1[n]["MAE"] for n in names], color=["#999", "#4c78a8", "#59a14f", "#f28e2b"])
ax.set_ylabel("MAE (story points, lower is better)")
ax.set_title("Effort estimation: model comparison")
plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
fig.tight_layout(); fig.savefig(FIGS / "effort_model_comparison.png", dpi=130); plt.close(fig)

# ----------------------------------------------------------------------------
# MODULE 5 - NLP: classify task complexity from text
# ----------------------------------------------------------------------------
log("MODULE 5: NLP complexity classification")
ytr5, yte5 = sp_train["complexity"], sp_test["complexity"]
m5 = {}
models5 = {
    "Majority baseline": make_pipeline(tfidf(), DummyClassifier(strategy="most_frequent")),
    "Naive Bayes (Complement)": make_pipeline(tfidf(), ComplementNB(alpha=0.5)),
    "Logistic Regression": make_pipeline(tfidf(), LogisticRegression(
        C=3, max_iter=1000, class_weight="balanced")),
    "Linear SVM": make_pipeline(tfidf(), LinearSVC(C=0.4, class_weight="balanced")),
}
fitted5 = {}
for name, pipe in models5.items():
    pipe.fit(sp_train["text"], ytr5)
    pr = pipe.predict(sp_test["text"])
    m5[name] = {"accuracy": float((pr == yte5).mean()),
                "macro_F1": float(f1_score(yte5, pr, average="macro")),
                "macro_precision": float(precision_score(yte5, pr, average="macro")),
                "macro_recall": float(recall_score(yte5, pr, average="macro"))}
    fitted5[name] = pr
    log(f"  {name}: acc={m5[name]['accuracy']:.3f} F1={m5[name]['macro_F1']:.3f}")
best5 = max([k for k in m5], key=lambda k: m5[k]["macro_F1"])
m5["best_model"] = best5
metrics["nlp_complexity"] = m5
joblib.dump(models5[best5].fit(sp["text"], sp["complexity"]), MODELS / "complexity_model.joblib")
fig, ax = plt.subplots(figsize=(4.8, 4.2))
ConfusionMatrixDisplay.from_predictions(yte5, fitted5[best5], ax=ax, cmap="Blues",
                                        labels=["Small", "Medium", "Large"],
                                        colorbar=False)
ax.set_title(f"Complexity classifier: {best5}")
fig.tight_layout(); fig.savefig(FIGS / "complexity_confusion.png", dpi=130); plt.close(fig)

# ----------------------------------------------------------------------------
# MODULE 4 - Task recommender: similar past tasks + team (project) routing
# ----------------------------------------------------------------------------
log("MODULE 4: similar-task recommender")
# The public datasets contain NO assignee names, so we recommend (a) the most
# similar past tasks with their effort, and (b) the project/team whose history
# fits the new task best. Project is used as a proxy for team.
vec4 = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2,
                       sublinear_tf=True, stop_words="english")
Xtr4 = vec4.fit_transform(sp_train["text"])
Xte4 = vec4.transform(sp_test["text"])
nn = NearestNeighbors(n_neighbors=10, metric="cosine").fit(Xtr4)
dist, idx = nn.kneighbors(Xte4)
proj_tr = sp_train["project"].values
sim = 1 - dist
votes_pred, top1_pred = [], []
for i in range(len(sp_test)):
    w = {}
    for j, s in zip(idx[i], sim[i]):
        w[proj_tr[j]] = w.get(proj_tr[j], 0) + s
    votes_pred.append(max(w, key=w.get))
    top1_pred.append(proj_tr[idx[i][0]])
true_proj = sp_test["project"].values
majority = pd.Series(proj_tr).value_counts().idxmax()
m4 = {
    "team_routing_accuracy_weighted_kNN(k=10)": float(np.mean(np.array(votes_pred) == true_proj)),
    "team_routing_accuracy_top1": float(np.mean(np.array(top1_pred) == true_proj)),
    "majority_baseline_accuracy": float(np.mean(true_proj == majority)),
    "n_projects(teams)": int(sp.project.nunique()),
    "note": "Public data has no assignee names; project is used as a proxy for team.",
}
# also: does the mean effort of the 10 neighbours predict story points?
nb_est = np.array([sp_train["storypoint"].values[idx[i]].mean() for i in range(len(sp_test))])
m4["neighbour_mean_effort_MAE"] = float(mean_absolute_error(y_te, nb_est))
metrics["task_recommender"] = m4
log(f"  routing acc={m4['team_routing_accuracy_weighted_kNN(k=10)']:.3f} "
    f"(baseline {m4['majority_baseline_accuracy']:.3f})")
# deploy on all data
vec4_full = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2,
                            sublinear_tf=True, stop_words="english")
X_full = vec4_full.fit_transform(sp["text"])
nn_full = NearestNeighbors(n_neighbors=10, metric="cosine").fit(X_full)
joblib.dump({"vec": vec4_full, "nn": nn_full,
             "meta": sp[["issuekey", "project", "title", "storypoint"]].reset_index(drop=True)},
            MODELS / "recommender.joblib")

# ----------------------------------------------------------------------------
# MODULE 2 - Delay prediction (classification)
# ----------------------------------------------------------------------------
log("MODULE 2: delay prediction")
dl = D.load_delay()
num_cols = [c for c in D.DELAY_NUMERIC if dl[c].nunique() > 1]   # drop constant columns
cat_cols = D.DELAY_CATEGORICAL
feat_cols = num_cols + cat_cols
log(f"  rows={len(dl)}, delayed={dl.delayed.mean():.3f}, numeric features kept={num_cols}")
Xd, yd = dl[feat_cols], dl["delayed"]
Xd_tr, Xd_te, yd_tr, yd_te, dl_tr, dl_te = train_test_split(
    Xd, yd, dl, test_size=0.2, random_state=SEED, stratify=yd)

pre_lin = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)])
pre_tree = ColumnTransformer([
    ("num", "passthrough", num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)])

models2 = {
    "Majority baseline": Pipeline([("p", pre_lin), ("m", DummyClassifier(strategy="most_frequent"))]),
    "Logistic Regression": Pipeline([("p", pre_lin), ("m", LogisticRegression(
        max_iter=2000, class_weight="balanced"))]),
    "Random Forest": Pipeline([("p", pre_tree), ("m", RandomForestClassifier(
        n_estimators=150, min_samples_leaf=5, class_weight="balanced_subsample",
        n_jobs=-1, random_state=SEED))]),
    "Gradient Boosting": Pipeline([("p", pre_tree), ("m", HistGradientBoostingClassifier(
        random_state=SEED, class_weight="balanced"))]),
}
# Hyper-parameter tuning for Gradient Boosting (GridSearchCV, 3-fold, optimise F1)
grid = GridSearchCV(
    models2["Gradient Boosting"],
    {"m__learning_rate": [0.05, 0.1], "m__max_depth": [4, None], "m__max_iter": [150, 300]},
    scoring="f1", cv=3, n_jobs=-1)
grid.fit(Xd_tr, yd_tr)
models2["Gradient Boosting (tuned)"] = grid.best_estimator_
log(f"  GridSearch best params: {grid.best_params_}")

m2, fitted2 = {}, {}
for name, pipe in models2.items():
    if name != "Gradient Boosting (tuned)":
        pipe.fit(Xd_tr, yd_tr)
    pr = pipe.predict(Xd_te)
    prob = pipe.predict_proba(Xd_te)[:, 1]
    m2[name] = {"accuracy": float((pr == yd_te).mean()),
                "precision": float(precision_score(yd_te, pr, zero_division=0)),
                "recall": float(recall_score(yd_te, pr)),
                "F1": float(f1_score(yd_te, pr)),
                "ROC_AUC": float(roc_auc_score(yd_te, prob))}
    fitted2[name] = (pipe, pr, prob)
    log(f"  {name}: F1={m2[name]['F1']:.3f} AUC={m2[name]['ROC_AUC']:.3f} "
        f"P={m2[name]['precision']:.3f} R={m2[name]['recall']:.3f}")
cand = [k for k in m2 if k != "Majority baseline"]
best2 = max(cand, key=lambda k: m2[k]["ROC_AUC"])
m2["best_model"] = best2
m2["best_params_GridSearch"] = {k: (v if v is not None else "None") for k, v in grid.best_params_.items()}
m2["delayed_share_in_data"] = float(dl.delayed.mean())
metrics["delay_prediction"] = m2

best_pipe2, best_pr2, best_prob2 = fitted2[best2]
fig, ax = plt.subplots(figsize=(4.8, 4.2))
ConfusionMatrixDisplay.from_predictions(yd_te, best_pr2, ax=ax, cmap="Blues",
                                        display_labels=["On time", "Delayed"], colorbar=False)
ax.set_title(f"Delay prediction: {best2}")
fig.tight_layout(); fig.savefig(FIGS / "delay_confusion.png", dpi=130); plt.close(fig)

names2 = [k for k in m2 if isinstance(m2[k], dict) and "F1" in m2[k]]
fig, ax = plt.subplots(figsize=(7, 4))
w = 0.2
for i, met in enumerate(["precision", "recall", "F1", "ROC_AUC"]):
    ax.bar(np.arange(len(names2)) + i * w, [m2[n][met] for n in names2], w, label=met)
ax.set_xticks(np.arange(len(names2)) + 1.5 * w)
ax.set_xticklabels(names2, rotation=15, ha="right"); ax.legend(); ax.set_ylim(0, 1)
ax.set_title("Delay prediction: model comparison")
fig.tight_layout(); fig.savefig(FIGS / "delay_model_comparison.png", dpi=130); plt.close(fig)

# Explainability: permutation importance on the raw input columns (test set)
pi = permutation_importance(best_pipe2, Xd_te, yd_te, scoring="roc_auc",
                            n_repeats=5, random_state=SEED, n_jobs=-1)
imp = pd.Series(pi.importances_mean, index=feat_cols).sort_values()
metrics["delay_prediction"]["permutation_importance_AUC_drop"] = {
    k: float(v) for k, v in imp.sort_values(ascending=False).items()}
fig, ax = plt.subplots(figsize=(6.5, 4.6))
imp.tail(12).plot.barh(ax=ax, color="#4c78a8")
ax.set_xlabel("Drop in ROC-AUC when feature is shuffled")
ax.set_title("Why does the model predict delay? (permutation importance)")
fig.tight_layout(); fig.savefig(FIGS / "delay_feature_importance.png", dpi=130); plt.close(fig)

# ----------------------------------------------------------------------------
# MODULE 3 - Risk scoring = P(delay) x expected delay length, scaled 0-100
# ----------------------------------------------------------------------------
log("MODULE 3: risk scoring")
delayed_tr = dl_tr[dl_tr.delayed == 1]
sev = Pipeline([("p", pre_tree), ("m", HistGradientBoostingRegressor(
    max_iter=200, learning_rate=0.06, random_state=SEED))])
sev.fit(delayed_tr[feat_cols], np.log1p(delayed_tr["delaydays"]))
exp_days_te = np.expm1(np.clip(sev.predict(dl_te[feat_cols]), 0, None))
risk_raw_te = best_prob2 * exp_days_te
risk_tr_raw = (fitted2[best2][0].predict_proba(dl_tr[feat_cols])[:, 1]
               * np.expm1(np.clip(sev.predict(dl_tr[feat_cols]), 0, None)))
# 0-100 score = percentile of raw risk among training issues
sorted_ref = np.sort(risk_tr_raw)
score_te = 100 * np.searchsorted(sorted_ref, risk_raw_te) / len(sorted_ref)
q1, q2 = np.percentile(score_te, [60, 85])
level = np.where(score_te >= q2, "High", np.where(score_te >= q1, "Medium", "Low"))
actual = dl_te["delaydays"].values
rho = spearmanr(score_te, actual).correlation
top10 = score_te >= np.percentile(score_te, 90)
m3 = {
    "spearman_corr_risk_vs_actual_delay_days": float(rho),
    "share_delayed_in_top10pct_risk": float(dl_te["delayed"].values[top10].mean()),
    "share_delayed_overall": float(dl_te["delayed"].mean()),
    "avg_actual_delay_days_High": float(actual[level == "High"].mean()),
    "avg_actual_delay_days_Medium": float(actual[level == "Medium"].mean()),
    "avg_actual_delay_days_Low": float(actual[level == "Low"].mean()),
    "level_thresholds_score": [float(q1), float(q2)],
}
metrics["risk_scoring"] = m3
log(f"  spearman={rho:.3f} top10% delayed share={m3['share_delayed_in_top10pct_risk']:.3f} "
    f"vs {m3['share_delayed_overall']:.3f}")
fig, ax = plt.subplots(figsize=(5, 3.8))
vals = [m3["avg_actual_delay_days_Low"], m3["avg_actual_delay_days_Medium"], m3["avg_actual_delay_days_High"]]
ax.bar(["Low", "Medium", "High"], vals, color=["#59a14f", "#edc948", "#e15759"])
ax.set_ylabel("Average actual delay (days)")
ax.set_title("Risk level vs. real delay (test set)")
fig.tight_layout(); fig.savefig(FIGS / "risk_levels_vs_actual.png", dpi=130); plt.close(fig)

# Deploy: refit delay + severity on all data
delay_full = models2[best2].fit(Xd, yd) if best2 != "Gradient Boosting (tuned)" else \
    Pipeline(grid.best_estimator_.steps).fit(Xd, yd)
sev_full = Pipeline([("p", pre_tree), ("m", HistGradientBoostingRegressor(
    max_iter=200, learning_rate=0.06, random_state=SEED))]).fit(
        dl[dl.delayed == 1][feat_cols], np.log1p(dl[dl.delayed == 1]["delaydays"]))
raw_all = (delay_full.predict_proba(Xd)[:, 1]
           * np.expm1(np.clip(sev_full.predict(Xd), 0, None)))
joblib.dump({"delay": delay_full, "severity": sev_full, "num_cols": num_cols,
             "cat_cols": cat_cols, "feat_cols": feat_cols,
             "risk_reference": np.sort(raw_all), "thresholds": [float(q1), float(q2)],
             "categories": {c: sorted(dl[c].dropna().unique().tolist()) for c in cat_cols},
             "num_defaults": {c: float(dl[c].median()) for c in num_cols},
             "num_max": {c: float(dl[c].quantile(.99)) for c in num_cols}},
            MODELS / "delay_risk.joblib")

# ----------------------------------------------------------------------------
# EDA figures for the report
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 3.8))
sp["storypoint"].value_counts().sort_index().head(14).plot.bar(ax=ax, color="#4c78a8")
ax.set_xlabel("Story points"); ax.set_ylabel("Number of tasks")
ax.set_title("Distribution of story points (all 16 projects)")
fig.tight_layout(); fig.savefig(FIGS / "eda_storypoints.png", dpi=130); plt.close(fig)
fig, ax = plt.subplots(figsize=(6, 3.8))
(dl.groupby("project")["delayed"].mean() * 100).sort_values().plot.barh(ax=ax, color="#e15759")
ax.set_xlabel("% of issues delayed"); ax.set_title("Delay rate by project")
fig.tight_layout(); fig.savefig(FIGS / "eda_delay_by_project.png", dpi=130); plt.close(fig)

metrics["data_summary"] = {
    "story_point_tasks": int(len(sp)), "story_point_projects": int(sp.project.nunique()),
    "story_points_capped_at_20": N_CAPPED,
    "delay_issues": int(len(dl)), "delay_projects": int(dl.project.nunique()),
    "train_test_split": "80/20, random_state=42 (stratified)"}
(ROOT / "reports" / "metrics.json").write_text(json.dumps(metrics, indent=2))
log("DONE. metrics saved to reports/metrics.json")
