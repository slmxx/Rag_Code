"""Compare deux expériences de recherche (Voyage vs Gemini) sur le même split.

Usage:
    python compare.py          # split dev
    python compare.py test     # split test
"""
import sys

import pandas as pd

split = sys.argv[1] if len(sys.argv) > 1 else "dev"
PATH = "llamaindex_pipeline/evaluation/runs/results_{}.csv"
COLS = ["hit@5", "recall@5", "mrr"]

v = pd.read_csv(PATH.format(f"voyage_{split}"))
g = pd.read_csv(PATH.format(f"gemini_{split}"))

if set(v["id"]) != set(g["id"]):
    sys.exit("Les deux fichiers n'ont pas les mêmes questions : comparaison impossible.")

print(f"##### Split : {split} ({len(v)} questions, hors-corpus exclues des moyennes) #####")

print("\n=== Global ===")
print(pd.DataFrame({"voyage": v[COLS].mean(), "gemini": g[COLS].mean()}).round(3))

print("\n=== Par type ===")
print(pd.concat({"voyage": v.groupby("type")[COLS].mean(),
                 "gemini": g.groupby("type")[COLS].mean()}, axis=1).round(3))

m = v.merge(g, on="id", suffixes=("_v", "_g"))
m = m[m["mrr_v"].notna()]                      # questions hors corpus : pas de MRR
print("\n=== Question par question (MRR) ===")
wins_g = int((m["mrr_g"] > m["mrr_v"]).sum())
wins_v = int((m["mrr_v"] > m["mrr_g"]).sum())
ties = int((m["mrr_v"] == m["mrr_g"]).sum())
print(f"Gemini meilleur : {wins_g} | Voyage meilleur : {wins_v} | égalité : {ties}")
diff = m[m["mrr_v"] != m["mrr_g"]]
print(diff[["id", "type_v", "mrr_v", "mrr_g"]]
      .rename(columns={"type_v": "type", "mrr_v": "mrr_voyage", "mrr_g": "mrr_gemini"})
      .to_string(index=False))

print("\n=== Questions ratées (hit@5 = 0) ===")
for name, d in (("voyage", v), ("gemini", g)):
    missed = d.loc[d["hit@5"] == 0, "id"].tolist()
    print(f"{name} : {missed if missed else 'aucune'}")

print("\n=== Latence de retrieval médiane (s) ===")
print({"voyage": round(float(v["retrieval_s"].median()), 3), "gemini": round(float(g["retrieval_s"].median()), 3)})
print("(attention : une latence proche de 0,01 s vient du cache, pas de l'API)")