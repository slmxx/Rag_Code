"""Évaluation du pipeline RAG : qualité + latence, paramétrable .
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime

import pandas as pd

from rag_pipeline import config
from rag_pipeline.evaluation.retrieval_metrics import compute_retrieval_metrics
from rag_pipeline.generation.answer import generate_answer
from rag_pipeline.retrieval import retriever


NON_QUALITY_COLS = ("retrieval_s", "generation_s", "total_s", "generation_failed")


def load_golden_set(path=None, split: str = "all") -> list[dict]:
    with open(path or config.GOLDEN_SET_FILE, encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]
    if split != "all":
        items = [it for it in items if it.get("split", "dev") == split]
    return items


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--embedder", default=None, help="voyage_code_4 | gemini_embedding | fake (défaut : config)")
    rr = p.add_mutually_exclusive_group()
    rr.add_argument("--reranker", dest="reranker", action="store_true", default=None)
    rr.add_argument("--no-reranker", dest="reranker", action="store_false")
    p.add_argument("--k", type=int, default=None, help="TOP_K_FINAL (défaut : config)")
    p.add_argument("--label", default=None, help="nom de l'expérience (défaut : généré)")
    p.add_argument("--no-ragas", action="store_true", help="ne lance pas le juge RAGAS (pas d'appel Gemini)")
    p.add_argument("--no-generation", action="store_true",
                   help="évalue seulement la recherche (hit@k, recall@k, MRR) : pas de Mistral, pas de RAGAS")
    p.add_argument("--golden", default=None, help="fichier JSONL des questions (défaut : GOLDEN_SET_FILE de config.py)")
    p.add_argument("--split", choices=["all", "dev", "test"], default="all",
                   help="dev = questions pour régler les paramètres ; test = mesure finale, à lancer une seule fois")
    p.add_argument("--no-warmup", action="store_true", help="ne fait pas d'appel de chauffe")
    p.add_argument("--manual", default=None, help="CSV des temps manuels (question,manual_seconds)")
    return p.parse_args()


def apply_config(args: argparse.Namespace) -> None:
    if args.embedder:
        config.ACTIVE_EMBEDDER = args.embedder
    if args.reranker is not None:
        config.USE_RERANKER = args.reranker
    if args.k:
        config.TOP_K_FINAL = args.k
    # le retriever garde l'embedder en mémoire : on le force à se recréer avec la bonne config
    retriever._embedder = None
    retriever._collection = None


def warm_up(generate: bool = True) -> None:
    """Le premier appel charge le client, la collection et le modèle Ollama : on ne le compte pas."""
    print("Chauffe (non comptée dans les latences)...")
    nodes = retriever.retrieve("warm-up")
    if generate:
        generate_answer("Réponds par OK.", [n.node.get_content() for n in nodes[:1]])


def run_questions(golden: list[dict], generate: bool = True) -> tuple[list[dict], pd.DataFrame]:
    ragas_rows, perf_rows = [], []
    k = config.TOP_K_FINAL
    for item in golden:
        q = item["question"]

        t0 = time.perf_counter()
        nodes = retriever.retrieve(q)
        t1 = time.perf_counter()
        contexts = [n.node.get_content() for n in nodes]
        paths = [n.node.metadata.get("path", "") for n in nodes]
        failed = False
        answer = ""
        if generate:
            try:
                answer = generate_answer(q, contexts)
            except Exception as exc:                  # timeout Ollama, etc. : on garde les autres questions
                failed = True
                print(f"✗ génération échouée ({exc.__class__.__name__}) : {q[:60]}")
        t2 = time.perf_counter()

        ragas_rows.append({
            "user_input": q,
            "retrieved_contexts": contexts,
            "response": answer,
            "reference": item["reference"],
        })
        relevant = item.get("relevant_files", [])
        if relevant:
            retrieval_metrics = compute_retrieval_metrics(paths, relevant, k)
        else:                                          # question hors corpus : pas de « bon fichier » à retrouver
            retrieval_metrics = {f"hit@{k}": float("nan"), f"recall@{k}": float("nan"), "mrr": float("nan")}
        perf_rows.append({
            "id": item.get("id", ""),
            "type": item.get("type", ""),
            "split": item.get("split", ""),
            "question": q,
            "retrieval_s": round(t1 - t0, 3),
            "generation_s": round(t2 - t1, 3),
            "total_s": round(t2 - t0, 3),
            "generation_failed": failed,
            "answer": answer,
            "retrieved_paths": " ; ".join(paths),
            **retrieval_metrics,
        })
        print(f"✓ {q[:70]}  ({t2 - t0:.1f} s)")
    return ragas_rows, pd.DataFrame(perf_rows)


def latency_stats(df: pd.DataFrame) -> dict:
    out = {}
    for col in ("retrieval_s", "generation_s", "total_s"):
        out[f"{col}_median"] = round(df[col].median(), 3)
        out[f"{col}_p95"] = round(df[col].quantile(0.95), 3)
    return out


def manual_comparison(df: pd.DataFrame, path: str) -> dict | None:
    manual = pd.read_csv(path)
    merged = df.merge(manual, on="question", how="inner")
    if merged.empty:
        print("--manual : aucune question en commun (le texte doit être identique au golden set).")
        return None
    med_manual = merged["manual_seconds"].median()
    med_rag = merged["total_s"].median()
    return {
        "n_manual": len(merged),
        "manual_median_s": round(float(med_manual), 1),
        "rag_median_s": round(float(med_rag), 1),
        "time_saved_pct": round(100 * (1 - med_rag / med_manual), 1),
        "speedup_x": round(float(med_manual / med_rag), 1),
    }


def write_summary(label: str, cfg: dict, final: pd.DataFrame, lat: dict, manual: dict | None) -> str:
    quality_cols = [c for c in final.columns
                    if c not in NON_QUALITY_COLS and pd.api.types.is_numeric_dtype(final[c])]
    lines = [
        f"# Expérience : {label}",
        "",
        f"- Embedder : `{cfg['embedder']}` | Reranker : `{cfg['reranker']}` | top-k final : `{cfg['k']}`",
        f"- MAX_CHUNK_CHARS : `{cfg['max_chunk_chars']}` | Questions : `{cfg['n_questions']}` | Split : `{cfg['split']}`",
        "",
        "## Qualité (moyenne ; les questions hors corpus sont exclues des métriques de recherche)",
        "",
        "| Métrique | Valeur |",
        "|---|---|",
    ]
    for c in quality_cols:
        lines.append(f"| {c} | {final[c].mean():.3f} |")
    retrieval_cols = [c for c in quality_cols if c.startswith(("hit@", "recall@")) or c == "mrr"]
    if "type" in final.columns and retrieval_cols and final["type"].astype(str).str.len().gt(0).any():
        lines += ["", "## Recherche par type de question", "",
                  "| Type | N | " + " | ".join(retrieval_cols) + " |",
                  "|---|---|" + "---|" * len(retrieval_cols)]
        for t, grp in final.groupby("type"):
            vals = " | ".join("n/a" if grp[c].isna().all() else f"{grp[c].mean():.3f}" for c in retrieval_cols)
            lines.append(f"| {t} | {len(grp)} | {vals} |")
    lines += [
        "",
        "## Latence par question",
        "",
        "| Étape | Médiane (s) | p95 (s) |",
        "|---|---|---|",
        f"| Retrieval (embedding requête + Chroma + rerank) | {lat['retrieval_s_median']} | {lat['retrieval_s_p95']} |",
        f"| Génération (Mistral 7B, Ollama) | {lat['generation_s_median']} | {lat['generation_s_p95']} |",
        f"| **Total** | **{lat['total_s_median']}** | **{lat['total_s_p95']}** |",
    ]
    if manual:
        lines += [
            "",
            "## Temps manuel vs RAG",
            "",
            "| | Médiane (s) |",
            "|---|---|",
            f"| Recherche manuelle ({manual['n_manual']} questions) | {manual['manual_median_s']} |",
            f"| RAG | {manual['rag_median_s']} |",
            "",
            f"**Gain : {manual['time_saved_pct']} %** (x{manual['speedup_x']} plus rapide, médiane).",
        ]
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    apply_config(args)

    label = args.label or (
        f"{config.ACTIVE_EMBEDDER}_{'rerank' if config.USE_RERANKER else 'norerank'}"
        f"_k{config.TOP_K_FINAL}_c{config.MAX_CHUNK_CHARS}"
        + ("" if args.split == "all" else f"_{args.split}"))
    cfg = {
        "embedder": config.ACTIVE_EMBEDDER, "reranker": config.USE_RERANKER,
        "k": config.TOP_K_FINAL, "max_chunk_chars": config.MAX_CHUNK_CHARS, "split": args.split,
    }
    print(f"Expérience : {label}\n")

    golden = load_golden_set(args.golden, args.split)
    if not golden:
        raise SystemExit("Aucune question à évaluer (vérifie --golden et --split).")
    cfg["n_questions"] = len(golden)

    generate = not args.no_generation
    if not args.no_warmup:
        warm_up(generate)

    ragas_rows, perf_df = run_questions(golden, generate)

    if generate:
        # sauvegarde des réponses AVANT le juge : si RAGAS échoue (limite de débit), rien n'est perdu
        runs_dir_early = config.EVAL_DIR / "runs"
        runs_dir_early.mkdir(parents=True, exist_ok=True)
        payload = {"label": label, "ids": [it.get("id", "") for it in golden],
                   "types": [it.get("type", "") for it in golden], "rows": ragas_rows}
        (runs_dir_early / f"ragas_input_{label}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    if args.no_ragas or not generate:
        final = perf_df
    else:
        from rag_pipeline.evaluation.judge import judge
        ragas_df = judge(ragas_rows)
        keep = [c for c in ragas_df.columns if c not in ("user_input", "retrieved_contexts", "reference")]
        final = pd.concat([perf_df, ragas_df[keep]], axis=1)

    lat = latency_stats(final)
    manual = manual_comparison(final, args.manual) if args.manual else None

    # --- sorties ---
    out_dir = config.EVAL_DIR                      # experiments.csv reste ici
    runs_dir = out_dir / "runs"                    # un fichier par expérience, rangé ici
    runs_dir.mkdir(parents=True, exist_ok=True)
    final.to_csv(runs_dir / f"results_{label}.csv", index=False, encoding="utf-8")
    summary = write_summary(label, cfg, final, lat, manual)
    (runs_dir / f"summary_{label}.md").write_text(summary, encoding="utf-8")

    numeric_means = {f"mean_{c}": round(final[c].mean(), 3)
                     for c in final.columns
                     if c not in NON_QUALITY_COLS and pd.api.types.is_numeric_dtype(final[c])}
    row = {"date": datetime.now().isoformat(timespec="seconds"), "label": label,
           **cfg, **lat, **numeric_means, **(manual or {})}
    log_path = out_dir / "experiments.csv"
    log_df = pd.DataFrame([row])
    if log_path.exists():
        log_df = pd.concat([pd.read_csv(log_path), log_df], ignore_index=True)
    log_df.to_csv(log_path, index=False, encoding="utf-8")

    print("\n" + summary)
    print(f"Fichiers : runs/results_{label}.csv, runs/summary_{label}.md, experiments.csv (dans {out_dir})")


if __name__ == "__main__":
    main()