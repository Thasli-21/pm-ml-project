# Machine-Learning Decision Support for Project Management
*College project report (draft - add your name, course, guide, date)*

## 1. Problem
Project managers spend much of their time on data-heavy judgements: how long will this task take, will this issue be late, what should I watch first, and who or which team fits it. This project builds a **decision-support system** that automates these predictions using Python and machine learning. It supports the manager; it does not replace the human parts of the job (people, negotiation, priorities).

## 2. Objectives
1. Estimate task effort (story points) from the task text.
2. Predict whether an issue will be delayed.
3. Score and rank issues by risk.
4. Recommend similar past tasks and the best-fit team.
5. Classify task complexity from text (NLP).
6. Present everything in an interactive dashboard.

## 3. Data
Real public datasets from Choetkiertikul et al. (SEAnalytics):
* Story points (IEEE TSE 2018): 23,313 tasks, 16 projects. Title + description with a story-point label.
* Delayed issues (EMSE 2017): 64,747 issues, 8 projects. Features at issue creation and the delay in days. 21.4% of issues are delayed.

Cleaning: dropped empty/duplicate tasks, merged title and description, capped story points at 20 (814 tasks, 3.5%), removed delay features that are constant at creation time, excluded LDA topic columns (not reproducible from a form).

Figures: `reports/figures/eda_storypoints.png`, `eda_delay_by_project.png`.

## 4. Method
* **Split:** 80/20 train/test, `random_state=42`, stratified. The final models are refit on all data for the dashboard.
* **Text features:** TF-IDF (uni+bi-grams, sublinear TF, stop words removed); SVD for tree models.
* **Module 1 (regression):** Ridge, Random Forest, Gradient Boosting vs. a median baseline. Metrics: MAE, RMSE; 3-fold cross-validation for the best model.
* **Module 2 (classification):** Logistic Regression, Random Forest, Gradient Boosting; Gradient Boosting tuned with GridSearchCV (3-fold, optimising F1). Class imbalance handled with class weights. Metrics: precision, recall, F1, ROC-AUC, confusion matrix (not accuracy alone).
* **Module 3 (risk):** risk = P(delay) x expected delay days (a second model trained on delayed issues, log-scale), converted to a 0-100 percentile score with Low/Medium/High bands.
* **Module 4 (recommender):** cosine k-NN (k=10) over TF-IDF vectors; similarity-weighted vote of neighbours' projects gives the best-fit team (project is a proxy for team).
* **Module 5 (NLP):** Naive Bayes, Logistic Regression, Linear SVM classify tasks into Small (<=2 points), Medium (3-5), Large (>5).
* **Explainability:** permutation importance on the delay model.

## 5. Results (test set)
**Module 1 - effort**

| Model | MAE | RMSE |
|---|---|---|
| Median baseline | 3.255 | 4.927 |
| Ridge (TF-IDF) | **2.923** | **4.042** |
| Random Forest | 3.099 | 4.299 |
| Gradient Boosting | 3.018 | 4.213 |

Best model improves MAE by 10.2% over the baseline; 3-fold CV MAE 3.31 +/- 0.30. The gain is modest, consistent with the literature: text alone carries limited information about effort.

**Module 2 - delay**

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Majority baseline | 0 | 0 | 0 | 0.500 |
| Logistic Regression | 0.370 | 0.627 | 0.465 | 0.725 |
| Random Forest | 0.602 | 0.731 | 0.660 | 0.884 |
| Gradient Boosting | 0.600 | 0.741 | 0.663 | 0.881 |
| Gradient Boosting (tuned) | **0.607** | **0.742** | **0.668** | **0.885** |

Most important features: `RemainingDay` (time between creation and due date), `project`, `reporterrep`, `no_fixversion`, `no_issuelink`, `type` (`delay_feature_importance.png`).

**Module 3 - risk:** Spearman correlation between risk score and real delay days 0.446. The top 10% highest-risk issues were delayed 64.9% of the time vs 21.4% overall. Average real delay: High 70.5 days, Medium 14.2, Low 2.9.

**Module 4 - recommender:** team-routing accuracy 86.7% (k=10 weighted vote), 82.1% (top-1 neighbour), vs 20.5% majority baseline. The mean effort of the 10 nearest tasks gives MAE 2.87, comparable to the best regression model.

**Module 5 - NLP:** best Linear SVM, accuracy 0.537, macro-F1 0.539 vs 0.188 for the majority baseline (3 classes; `complexity_confusion.png`).

## 6. ML vs. a simple manual rule
The "median" baseline stands in for a simple manual estimate (always guess the typical size). The models beat it on every module, but only modestly for effort. This shows where ML helps (ranking risk, flagging likely delays, routing) and where it does not yet help much (precise effort from text alone).

## 7. Limitations
* Open-source data, not company data; results may not transfer.
* No assignee names in the public data, so no individual-person recommendation.
* Delay data only contains issues with due dates; `RemainingDay` dominates the model.
* Story points capped at 20; random (not time-based) split, so results may be optimistic versus real deployment.
* Predictions are aids for prioritisation, not guarantees.

## 8. Future work
Time-based validation, deep-learning text models, per-developer data with assignees, sprint-level velocity prediction (the sprint dataset in the same repository), SHAP explanations, deployment with live Jira integration.

## 9. References
* M. Choetkiertikul, H. K. Dam, T. Tran, T. Pham, A. Ghose, T. Menzies. "A deep learning model for estimating story points." IEEE Trans. Software Engineering, 2018.
* M. Choetkiertikul, H. K. Dam, T. Tran, A. Ghose. "Predicting the delay of issues with due dates in software projects." Empirical Software Engineering 22(3), 2017.
* Datasets: https://github.com/SEAnalytics/datasets
* scikit-learn: Pedregosa et al., JMLR 12, 2011.
