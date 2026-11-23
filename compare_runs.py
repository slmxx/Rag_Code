"""Compare plusieurs expériences run_eval (métriques de recherche) dans un seul tableau.

Usage (à la racine du projet) :
    python compare_runs.py voyage_dev gemini_dev hybrid_dev voyage_bge_dev gemini_bge_dev hybrid_bge_dev

Lit llamaindex_pipeline/evaluation/runs/results_<label>.csv pour chaque label.
"""
import sys

import pandas as pd

from rag_pipeline import config

RUNS = config.EVAL_DIR / "runs"
labels = sys.argv[1:]
if len(labels) < 2:
    sys.exit(__doc__)

dfs = {}
for lab in labels:
    path = RUNS / f"results_{lab}.csv"
    if not path.exists():
        sys.exit(f"Introuvable : {path}\nLabels disponibles : {sorted(p.stem[8:] for p in RUNS.glob('results_*.csv'))}")
    dfs[lab] = pd.read_csv(path)

first = next(iter(dfs.values()))
hit_col = next(c for c in first.columns if c.startswith("hit@"))
rec_col = next(c for c in first.columns if c.startswith("recall@"))

print("=== Global (questions hors corpus exclues) ===")
glob = pd.DataFrame({
    lab: {
        hit_col: df[hit_col].mean(),
        rec_col: df[rec_col].mean(),
        "mrr": df["mrr"].mean(),
        "N": int(df["mrr"].notna().sum()),
        "retrieval_median_s": df["retrieval_s"].median(),
    } for lab, df in dfs.items()
}).T
print(glob.round(3).to_string())

print("\n=== MRR par type de question ===")
print(pd.DataFrame({lab: df.groupby("type")["mrr"].mean() for lab, df in dfs.items()}).round(3).to_string())

print("\n=== MRR question par question (seulement les questions où les systèmes diffèrent) ===")
mat = pd.DataFrame({lab: df.set_index("id")["mrr"] for lab, df in dfs.items()})
types = first.set_index("id")["type"]
mat.insert(0, "type", types)
vals = mat.drop(columns="type")
diff = mat[vals.nunique(axis=1, dropna=True) > 1]
print(diff.round(2).to_string() if len(diff) else "(aucune différence)")

print("\n=== Questions manquées (MRR = 0) par expérience ===")
for lab, df in dfs.items():
    missed = df.loc[df["mrr"] == 0, "id"].tolist()
    print(f"{lab:22s} {missed if missed else '—'}")