"""Streamlit dashboard: ML decision support for project managers.

Run:  streamlit run app.py
"""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import predict as P  # noqa: E402

st.set_page_config(page_title="PM Decision Support (ML)", page_icon="📊", layout="wide")
st.title("📊 Project Manager Decision Support")
st.caption("Machine learning on real open-source project data (Jira issues). "
           "Supports the project manager - it does not replace human judgement.")

metrics_path = ROOT / "reports" / "metrics.json"
metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
figs = ROOT / "reports" / "figures"

tab_task, tab_delay, tab_perf, tab_about = st.tabs(
    ["🧩 Task analysis", "⏱ Delay & risk", "📈 Model performance", "ℹ️ About"])

# ---------------------------------------------------------------- Task analysis
with tab_task:
    st.subheader("Describe a task")
    st.write("Gets: effort estimate (Module 1), complexity class (Module 5), "
             "similar past tasks and best-fit team (Module 4).")
    example = ("Add retry logic to the file upload service so that failed uploads "
               "are retried three times with exponential backoff and logged.")
    text = st.text_area("Task title + description", value=example, height=140)
    if st.button("Analyse task", type="primary") and text.strip():
        c1, c2, c3 = st.columns(3)
        pts = P.predict_effort(text)
        comp = P.predict_complexity(text)
        sim = P.similar_tasks(text)
        c1.metric("Estimated effort", f"{pts:.1f} story points")
        c2.metric("Complexity", comp["label"])
        c3.metric("Best-fit team (project)", sim["best_team"])
        if comp["scores"]:
            st.write("**Complexity confidence**")
            st.bar_chart(pd.Series(comp["scores"]))
        st.write(f"**Most similar past tasks** (neighbours average "
                 f"{sim['neighbour_avg_points']:.1f} points)")
        st.dataframe(sim["tasks"].rename(columns={
            "issuekey": "Issue", "project": "Project", "title": "Title",
            "storypoint": "Story points", "similarity": "Similarity"}),
            use_container_width=True, hide_index=True)
        st.write("**Team fit scores**")
        st.bar_chart(sim["team_scores"])

# ---------------------------------------------------------------- Delay & risk
with tab_delay:
    st.subheader("Will this issue be delayed?")
    st.write("Gets: delay probability (Module 2) and risk score (Module 3).")
    opt = P.delay_options()
    cats = opt["categories"]
    a, b, c = st.columns(3)
    project = a.selectbox("Project", cats["project"])
    itype = b.selectbox("Issue type", cats["type"],
                        index=cats["type"].index("Bug") if "Bug" in cats["type"] else 0)
    prio = c.selectbox("Priority", cats["priority"],
                       index=cats["priority"].index("Major") if "Major" in cats["priority"] else 0)
    row = {"project": project, "type": itype, "priority": prio}
    st.write("**Issue activity & planning**")
    cols = st.columns(4)
    for i, col in enumerate(opt["num_cols"]):
        mx = max(int(opt["num_max"][col]), 5)
        row[col] = cols[i % 4].number_input(
            col.replace("_", " "), min_value=0, max_value=max(mx * 3, 10),
            value=int(opt["defaults"][col]), step=1, key=f"n_{col}")
    if st.button("Predict delay & risk", type="primary"):
        r = P.predict_delay_and_risk(row)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Delay probability", f"{r['delay_probability']*100:.0f}%")
        m2.metric("Expected delay if late", f"{r['expected_delay_days_if_late']:.0f} days")
        m3.metric("Risk score (0-100)", f"{r['risk_score']:.0f}")
        m4.metric("Risk level", r["risk_level"])
        icon = {"High": "🔴", "Medium": "🟠", "Low": "🟢"}[r["risk_level"]]
        st.info(f"{icon} Risk level **{r['risk_level']}**. The score ranks this issue against "
                "issues in the training data; it is a prioritisation aid, not a guarantee.")

    st.divider()
    st.write("**Batch scoring** - upload a CSV with columns: "
             + ", ".join(["project", "type", "priority"] + opt["num_cols"]))
    up = st.file_uploader("CSV file", type="csv")
    if up is not None:
        df = pd.read_csv(up)
        out = [P.predict_delay_and_risk(r.to_dict()) for _, r in df.iterrows()]
        res = pd.concat([df, pd.DataFrame(out)], axis=1).sort_values("risk_score", ascending=False)
        st.dataframe(res, use_container_width=True, hide_index=True)
        st.download_button("Download scored CSV", res.to_csv(index=False), "scored_issues.csv")

# ---------------------------------------------------------------- Performance
with tab_perf:
    st.subheader("How good are the models?")
    if not metrics:
        st.warning("Run `python src/train.py` first to generate reports/metrics.json.")
    else:
        e = metrics["effort_estimation"]
        st.write(f"**Effort estimation** - best: {e['best_model']}, MAE "
                 f"{e[e['best_model']]['MAE']:.2f} vs baseline {e['Median baseline']['MAE']:.2f}")
        d = metrics["delay_prediction"]
        dm = {k: v for k, v in d.items() if isinstance(v, dict) and "F1" in v}
        st.write(f"**Delay prediction** - best: {d['best_model']}")
        st.dataframe(pd.DataFrame(dm).T.round(3))
        r = metrics["risk_scoring"]
        st.write(f"**Risk scoring** - Spearman correlation with real delay: "
                 f"{r['spearman_corr_risk_vs_actual_delay_days']:.2f}; "
                 f"top-10% risk issues were delayed {r['share_delayed_in_top10pct_risk']*100:.0f}% "
                 f"of the time vs {r['share_delayed_overall']*100:.0f}% overall.")
        n = metrics["nlp_complexity"]
        st.write(f"**Complexity NLP** - best: {n['best_model']}, macro-F1 "
                 f"{n[n['best_model']]['macro_F1']:.2f}")
        t = metrics["task_recommender"]
        st.write(f"**Team routing** - accuracy "
                 f"{t['team_routing_accuracy_weighted_kNN(k=10)']*100:.0f}% vs "
                 f"{t['majority_baseline_accuracy']*100:.0f}% majority baseline")
        imgs = sorted(figs.glob("*.png"))
        cols = st.columns(3)
        for i, f in enumerate(imgs):
            cols[i % 3].image(str(f), caption=f.stem.replace("_", " "))

# ---------------------------------------------------------------- About
with tab_about:
    st.markdown("""
**Data (real, public):** Choetkiertikul et al. - story point dataset (IEEE TSE 2018, 23k tasks,
16 projects) and delayed-issue dataset (EMSE 2017, 64k issues, 8 projects).

**Modules:** 1 effort estimation - 2 delay prediction - 3 risk scoring -
4 similar-task / team recommender - 5 NLP complexity classifier - 6 this dashboard.

**Limitations:** open-source Jira data is not your company's data; the public datasets contain no
assignee names, so "team" means project; predictions support, not replace, the project manager.
""")
