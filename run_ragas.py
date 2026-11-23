"""RAGAS à part, à partir des réponses déjà générées par run_eval (sans relancer Mistral).

Usage :
    python run_ragas.py judge   gemini_gen_test       
    python run_ragas.py compare gemini_gen_test voyage_gen_test

`judge` lit  rag_pipeline/evaluation/runs/ragas_input_<label>.json  (créé par run_eval, sans --no-generation)
et écrit     rag_pipeline/evaluation/runs/ragas_<label>.csv
"""
import json
import sys

import pandas as pd

from rag_pipeline import config

RUNS = config.EVAL_DIR / "runs"
REFUS_MOTS = ("ne contient pas", "pas d'information", "aucune information", "ne mentionne pas",
              "n'est pas mentionn", "pas indiqu", "pas précis", "ne dit pas", "impossible de répondre")


def cmd_judge(label: str) -> None:
    src = RUNS / f"ragas_input_{label}.json"
    if not src.exists():
        sys.exit(f"Fichier introuvable : {src}\nLance d'abord run_eval sans --no-generation.")
    data = json.loads(src.read_text(encoding="utf-8"))
    ids, types, rows = data["ids"], data["types"], data["rows"]

    # 1. questions hors corpus : on lit les réponses (RAGAS n'a pas de sens ici), refus = tentative de détection
    print("=== Questions hors corpus : la réponse refuse-t-elle ? (détection indicative, lis les réponses) ===")
    n_ref = n_tot = 0
    for i, t, r in zip(ids, types, rows):
        if t == "hors_corpus":
            n_tot += 1
            refuse = any(m in r["response"].lower() for m in REFUS_MOTS)
            n_ref += refuse
            print(f"\n[{i}] {'REFUS' if refuse else 'A RÉPONDU ?'} | {r['user_input']}\n    {r['response'][:300]}")
    if n_tot:
        print(f"\nRefus détectés : {n_ref}/{n_tot}")

    # 2. RAGAS sur les autres questions, en écartant les générations vides (timeout Ollama)
    keep = [k for k, (t, r) in enumerate(zip(types, rows)) if t != "hors_corpus" and r["response"].strip()]
    skipped = [ids[k] for k, (t, r) in enumerate(zip(types, rows)) if t != "hors_corpus" and not r["response"].strip()]
    if skipped:
        print(f"\nGénérations vides ignorées (timeout ?) : {skipped}")
    print(f"\nRAGAS sur {len(keep)} questions (peut durer plusieurs minutes)...")

    from rag_pipeline.evaluation.judge import judge
    df = judge([rows[k] for k in keep])
    df.insert(0, "id", [ids[k] for k in keep])
    df.insert(1, "type", [types[k] for k in keep])
    df = df.drop(columns=[c for c in ("retrieved_contexts",) if c in df.columns])

    out = RUNS / f"ragas_{label}.csv"
    df.to_csv(out, index=False, encoding="utf-8")
    metrics = [c for c in df.columns if c not in ("id", "type", "user_input", "response", "reference")
               and pd.api.types.is_numeric_dtype(df[c])]
    print("\n=== Moyennes RAGAS ===")
    print(df[metrics].mean().round(3).to_string())
    print("\n=== Par type ===")
    print(df.groupby("type")[metrics].mean().round(3).to_string())
    print(f"\nFichier : {out}")


def cmd_compare(a: str, b: str) -> None:
    da = pd.read_csv(RUNS / f"ragas_{a}.csv")
    db = pd.read_csv(RUNS / f"ragas_{b}.csv")
    metrics = [c for c in da.columns if c in db.columns and c not in ("id", "type", "user_input", "response", "reference")
               and pd.api.types.is_numeric_dtype(da[c])]
    print("=== Moyennes RAGAS ===")
    print(pd.DataFrame({a: da[metrics].mean(), b: db[metrics].mean()}).round(3))
    print("\n=== Par type ===")
    print(pd.concat({a: da.groupby("type")[metrics].mean(), b: db.groupby("type")[metrics].mean()}, axis=1).round(3))
    m = da.merge(db, on="id", suffixes=("_a", "_b"))
    if "faithfulness_a" in m.columns:
        print("\n=== Faithfulness question par question ===")
        print(m[["id", "type_a", "faithfulness_a", "faithfulness_b"]]
              .rename(columns={"type_a": "type", "faithfulness_a": a, "faithfulness_b": b}).round(2).to_string(index=False))
    print(f"\nNombre de questions : {a} = {len(da)}, {b} = {len(db)}")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "judge":
        cmd_judge(sys.argv[2])
    elif len(sys.argv) == 4 and sys.argv[1] == "compare":
        cmd_compare(sys.argv[2], sys.argv[3])
    else:
        sys.exit(__doc__)