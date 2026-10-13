# Machine-Learning Decision Support for Project Management

Python + machine learning on **real open-source Jira data** to support (not replace) a project manager.

| # | Module | Task | Model(s) | Key result (held-out 20% test set) |
|---|--------|------|----------|------------------------------------|
| 1 | Effort estimation | Regression: story points from task text | Ridge, Random Forest, Gradient Boosting vs. median baseline | MAE 2.92 vs 3.26 baseline (-10%) |
| 2 | Delay prediction | Classification: will the issue be late? | Logistic Reg., Random Forest, Gradient Boosting (+GridSearchCV) | F1 0.67, ROC-AUC 0.885 |
| 3 | Risk scoring | Risk = P(delay) x expected delay days, scaled 0-100 | Gradient Boosting classifier + regressor | Top-10% risk issues delayed 65% of the time vs 21% overall |
| 4 | Task recommender | Similar past tasks + best-fit team | TF-IDF + cosine k-NN | 86.7% team-routing accuracy vs 20.5% baseline |
| 5 | NLP | Classify task complexity (Small/Medium/Large) from text | Naive Bayes, Logistic Reg., Linear SVM | Macro-F1 0.54 vs 0.19 baseline |
| 6 | Dashboard | Streamlit app combining 1-5 | - | `streamlit run app.py` |

## Data (real, public)
From the SEAnalytics repository (Choetkiertikul et al.):
* **Story points** - IEEE TSE 2018, 23,313 tasks, 16 projects (`data/raw/storypoint/`)
* **Delayed issues** - EMSE 2017, 64,747 issues, 8 projects (`data/raw/delayed issues/`)

Source: https://github.com/SEAnalytics/datasets - please cite the original papers (see `REPORT.md`).

## Run it
```bash
pip install -r requirements.txt
python src/train.py        # trains all models (~4 min), writes models/, reports/
streamlit run app.py       # opens the dashboard
```
Trained models are already included in `models/`, so you can run the dashboard immediately.

## Folder structure
```
app.py                 Streamlit dashboard (Module 6)
src/data.py            data loading and labels
src/train.py           training + evaluation for Modules 1-5
src/predict.py         inference helpers used by the app
models/                saved models (.joblib)
reports/metrics.json   all metrics
reports/figures/       charts for the report
data/raw/              public datasets
REPORT.md              report draft
```

## Honest limitations
* Open-source Jira data is not your organisation's data.
* Story points above 20 were capped at 20 (3.5% of tasks) because some projects use very large scales.
* The public datasets contain **no assignee names**, so Module 4 recommends similar tasks and the best-fit *project (team proxy)*, not an individual person.
* Delay model uses only issues that have a due date; `RemainingDay` (days until the due date) is its strongest feature.
* Effort estimation from text alone is hard: the gain over a median baseline is modest (about 10%), in line with published work.
