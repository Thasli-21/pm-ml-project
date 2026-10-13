"""Data loading for the PM-ML project.

Real public data from SEAnalytics (Choetkiertikul et al.):
  * Story points  : IEEE TSE 2018  -> task text + story point label (16 projects)
  * Delayed issues: EMSE 2017      -> issue features at creation time + delay in days (8 projects)
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
SP_DIR = RAW / "storypoint" / "IEEE TSE2018" / "dataset"
DELAY_DIR = RAW / "delayed issues" / "EMSE2017" / "datasets"

# Delay features that are known when the issue is created / given a due date.
# LDA topic columns are excluded: they need the authors' topic model, which a
# manager could not reproduce from a simple form.
DELAY_NUMERIC = [
    "discussion", "repetition", "perofdelay", "workload", "no_comment",
    "no_priority_change", "no_fixversion", "no_fixversion_change", "no_issuelink",
    "no_blocking", "no_blockedby", "no_affectversion", "reporterrep",
    "no_des_change", "ProgressTime", "RemainingDay",
]
DELAY_CATEGORICAL = ["project", "type", "priority"]


def load_storypoints() -> pd.DataFrame:
    frames = []
    for f in sorted(SP_DIR.glob("*.csv")):
        d = pd.read_csv(f)
        d["project"] = f.stem
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["title"] = df["title"].fillna("")
    df["description"] = df["description"].fillna("")
    df["text"] = (df["title"] + ". " + df["description"]).str.strip()
    df = df[df["text"].str.len() > 3].copy()
    df = df.dropna(subset=["storypoint"])
    df = df.drop_duplicates(subset=["issuekey"])
    df["text_len"] = np.log1p(df["text"].str.len())
    df["title_len"] = np.log1p(df["title"].str.len())
    return df.reset_index(drop=True)


def load_delay() -> pd.DataFrame:
    frames = []
    for f in sorted(DELAY_DIR.glob("*_creation.csv")):
        d = pd.read_csv(f)
        d["project"] = f.stem.replace("_creation", "")
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["openeddate"] = pd.to_datetime(df["openeddate"], errors="coerce")
    df["delayed"] = (df["delaydays"] > 0).astype(int)
    df["year"] = df["openeddate"].dt.year
    return df.reset_index(drop=True)


def complexity_label(points: float) -> str:
    """Bucket story points into Small / Medium / Large."""
    if points <= 2:
        return "Small"
    if points <= 5:
        return "Medium"
    return "Large"
