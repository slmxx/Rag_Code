"""Rapport par question : classement de plusieurs systèmes (voyage, gemini, hybrid), rang du fichier
attendu et texte des chunks envoyés à Mistral.

Usage :
    python retrieval_report.py dev                    # voyage + gemini + hybrid, SANS reranker
    python retrieval_report.py dev --rerank           # mêmes systèmes AVEC le reranker de config.py (bge en local)
    python retrieval_report.py dev --rerank --full 5  # texte complet des 5 chunks envoyés à Mistral
    python retrieval_report.py dev --systems voyage,hybrid
    python retrieval_report.py test --rerank          # split test 

Sorties (à la racine) :
    retrieval_report_<split>_<norerank|rerank_local>.md    
    retrieval_report_<split>_<...>.csv             

Les rangs sont des rangs de CHUNKS : un fichier présent 3 fois occupe 3 places.
Avec --rerank, le reranker trie le pool de TOP_K_RETRIEVE candidats (20) : un fichier hors de ce pool est « >20 ».
Les chiffres sont les mêmes que ceux de run_eval (même pool, mêmes top-5).
"""
import argparse
import csv
import json

from rag_pipeline import config

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("split", nargs="?", default="dev", choices=["dev", "test", "all"])
ap.add_argument("--full", type=int, default=2, help="nombre de chunks affichés en texte complet (défaut 2)")
ap.add_argument("--depth", type=int, default=None, help="profondeur de recherche, sans reranker seulement (défaut : pool de config)")
ap.add_argument("--rerank", action="store_true", help="applique le reranker de config.py (RERANK_BACKEND)")
ap.add_argument("--systems", default="voyage,gemini,hybrid", help="liste séparée par des virgules")
args = ap.parse_args()

ALL_SYSTEMS = {"voyage": "voyage_code_4", "gemini": "gemini_embedding", "hybrid": "hybrid"}
unknown = [s for s in args.systems.split(",") if s not in ALL_SYSTEMS]
if unknown:
    raise SystemExit(f"Système inconnu : {unknown}. Choix : {list(ALL_SYSTEMS)}")
SYSTEMS = {s: ALL_SYSTEMS[s] for s in args.systems.split(",")}

SPLIT = args.split
GOLDEN = "llamaindex_pipeline/evaluation/golden_set_v2.jsonl"
TOPN = 5                                           # chunks envoyés à Mistral
FULL = max(0, min(args.full, TOPN))
POOL = config.TOP_K_RETRIEVE                       # pool de candidats, comme run_eval
BACKEND = getattr(config, "RERANK_BACKEND", "local")
TAG = f"rerank_{BACKEND}" if args.rerank else "norerank"

config.USE_RERANKER = args.rerank
if args.rerank:
    DEPTH = max(POOL, TOPN)                        # le reranker renvoie tout le pool, trié
    if args.depth and args.depth != DEPTH:
        print(f"(--depth ignoré avec --rerank : le reranker trie le pool de {DEPTH})")
    config.TOP_K_FINAL = DEPTH
else:
    DEPTH = max(args.depth or POOL, TOPN)
    config.TOP_K_RETRIEVE = DEPTH
    config.TOP_K_FINAL = DEPTH
from rag_pipeline.retrieval import retriever  # noqa: E402


def norm(p):
    return p.replace("\\", "/")


def rank_of(hits, target):
    for i, h in enumerate(hits, 1):
        if target in h[0]:
            return i
    return None


def fmt_rank(r):
    return str(r) if r else f">{DEPTH}"


def is_expected(path, expected):
    return any(t in path for t in expected)


def body_of(text):
    return text.split("\n---\n", 1)[-1]


def preview(text, n=140):
    return " ".join(body_of(text).split())[:n]


def fence(text):
    return ["~~~~~~text", text.strip(), "~~~~~~"]


def details(summary, lines):
    return ["<details>", f"<summary>{summary}</summary>", ""] + lines + ["", "</details>", ""]


_file_cache = {}


def file_chunks(system, path):
    """Chunks indexés d'un fichier (hybrid utilise la collection Voyage : même chunker)."""
    emb = "voyage_code_4" if SYSTEMS[system] == "hybrid" else SYSTEMS[system]
    key = (emb, path)
    if key not in _file_cache:
        try:
            from rag_pipeline.store import get_collection
            r = get_collection(emb).get(where={"path": path}, include=["documents", "metadatas"])
            items = sorted(zip(r["metadatas"], r["documents"]), key=lambda x: x[0].get("start_line") or 0)
            _file_cache[key] = [(m.get("start_line"), m.get("end_line"), d) for m, d in items]
        except Exception as exc:                          # noqa: BLE001
            _file_cache[key] = f"lecture impossible : {exc}"
    return _file_cache[key]


questions = []
for line in open(GOLDEN, encoding="utf-8"):
    if line.strip():
        q = json.loads(line)
        if SPLIT == "all" or q.get("split") == SPLIT:
            questions.append(q)
if not questions:
    raise SystemExit(f"Aucune question pour le split « {SPLIT} ».")

# --- 1. recherche pour chaque système ---
results = {}
for name, emb in SYSTEMS.items():
    config.ACTIVE_EMBEDDER = emb
    retriever._embedder = None
    retriever._collection = None
    results[name] = {}
    print(f"Recherche : {name} ({len(questions)} questions, reranker={'oui, ' + BACKEND if args.rerank else 'non'})...")
    for i, q in enumerate(questions, 1):
        nodes = retriever.retrieve(q["question"])
        results[name][q["id"]] = [
            (norm(n.node.metadata.get("path", "")),
             n.node.metadata.get("start_line"), n.node.metadata.get("end_line"),
             n.node.get_content(), n.score)
            for n in nodes
        ]
        print(f"  {i}/{len(questions)}", end="\r")
    print()

# --- 2. rapport ---
names = list(SYSTEMS)
key = lambda r: r if r else 10 ** 6                        # noqa: E731
rows, summary = [], []
best_by_sys = {n: [] for n in names}                       # meilleur rang par question (questions avec fichier attendu)
md = [f"# Rapport de recherche ({', '.join(names)}) : split {SPLIT}, reranker : {BACKEND if args.rerank else 'aucun'}", "",
      "Comment lire ce rapport :",
      "- ✅ = fichier attendu (vérité terrain du golden set), ❌ = fichier non attendu.",
      f"- Mistral reçoit les {TOPN} premiers chunks ; le texte complet est affiché pour les {FULL} premiers "
      "(clique sur la flèche pour ouvrir).",
      f"- Le fichier attendu est cherché jusqu'au rang {DEPTH}. S'il n'est pas remonté, on liste ses chunks indexés.",
      "- `hybrid` = fusion RRF des classements Voyage et Gemini.", ""]

for q in questions:
    expected = [norm(t) for t in q.get("relevant_files", [])]
    info = {}
    for name in names:
        hits = results[name][q["id"]]
        ranks = {t: rank_of(hits, t) for t in expected}
        found = [r for r in ranks.values() if r]
        info[name] = {"hits": hits, "ranks": ranks, "best": min(found) if found else None}
        if expected:
            best_by_sys[name].append(info[name]["best"])

    if not expected:
        winner = "—"
    else:
        keys = {n: key(info[n]["best"]) for n in names}
        m = min(keys.values())
        leaders = [n for n in names if keys[n] == m]
        if m == 10 ** 6:
            winner = f"aucun (tous >{DEPTH})"
        elif len(leaders) == len(names):
            winner = "égalité"
        else:
            winner = " + ".join(leaders)

    row = {"id": q["id"], "type": q.get("type", ""), "split": q.get("split", ""), "question": q["question"],
           "fichiers_attendus": " ; ".join(expected) if expected else "(hors corpus : aucun)"}
    for name in names:
        i = info[name]
        row[f"{name}_meilleur_rang"] = fmt_rank(i["best"]) if expected else ""
        row[f"{name}_rang_par_fichier"] = " ; ".join(f"{t.split('/')[-1]}={fmt_rank(r)}" for t, r in i["ranks"].items())
        row[f"{name}_top{TOPN}"] = " | ".join(f"{k}. {h[0]}" for k, h in enumerate(i["hits"][:TOPN], 1))
    row["meilleur"] = winner
    rows.append(row)
    summary.append((q["id"], q.get("type", ""), [fmt_rank(info[n]["best"]) if expected else "—" for n in names], winner))

    md += [f"## {q['id']} · {q.get('type', '')}", "", f"**Question :** {q['question']}", ""]
    if expected:
        md.append("**Fichier(s) attendu(s) (le premier de la liste est en général le principal) :**")
        md += [f"- `{t}`" for t in expected]
    else:
        md.append("**Question hors corpus : aucun fichier attendu.** La bonne réponse est de dire que le contexte ne contient pas l'information.")
    md += ["", "| Rang | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for k in range(TOPN):
        cells = []
        for name in names:
            hits = info[name]["hits"]
            if k < len(hits):
                p, s, e = hits[k][0], hits[k][1], hits[k][2]
                lines = f" (l.{s}-{e})" if s is not None else ""
                cells.append(f"`{p}`{lines}" + (" ✅" if is_expected(p, expected) else ""))
            else:
                cells.append("")
        md.append(f"| {k + 1} | " + " | ".join(cells) + " |")
    if expected:
        md += ["", f"**Rang du fichier attendu (sur {DEPTH} chunks) :**"]
        for name in names:
            parts = ", ".join(f"`{t.split('/')[-1]}` = {fmt_rank(r)}" for t, r in info[name]["ranks"].items())
            md.append(f"- {name} : {parts}")
        md.append(f"- **Meilleur :** {winner}")
    md.append("")

    for name in names:
        hits = info[name]["hits"]
        shown = hits[:FULL]
        if shown:
            body = []
            for k, (p, s, e, text, sc) in enumerate(shown, 1):
                mark = "" if not expected else (" ✅ attendu" if is_expected(p, expected) else " ❌ non attendu")
                body += [f"**Rang {k}** · `{p}` (l.{s}-{e}) · score {sc:.3f}{mark}", ""] + fence(text) + [""]
            md += details(f"{name} : texte complet des {len(shown)} premiers chunks (sur les {TOPN} envoyés à Mistral)", body)
        rest = hits[FULL:TOPN]
        if rest:
            md += details(f"{name} : aperçu des chunks de rang {FULL + 1} à {TOPN}",
                          [f"- {FULL + 1 + j}. `{h[0]}` (l.{h[1]}-{h[2]}) · score {h[4]:.3f} : {preview(h[3])}"
                           for j, h in enumerate(rest)])
        for t in expected:
            r = info[name]["ranks"][t]
            base = t.split("/")[-1]
            if r and r > FULL:
                p, s, e, text, sc = hits[r - 1]
                where = "" if r <= TOPN else " (hors des 5 envoyés à Mistral)"
                md += details(f"{name} : chunk attendu `{base}` trouvé au rang {r}{where}",
                              [f"`{p}` (l.{s}-{e}) · score {sc:.3f}", ""] + fence(text))
            elif not r:
                chunks = file_chunks(name, t)
                if isinstance(chunks, str):
                    body = [chunks]
                else:
                    body = [f"- l.{s}-{e} : {preview(d)}" for s, e, d in chunks] or ["(aucun chunk indexé pour ce fichier : fichier non indexé ?)"]
                md += details(f"{name} : `{base}` absent des {DEPTH} premiers : voici ses chunks indexés", body)

# --- 3. récapitulatif ---
md += ["## Récapitulatif", "", "| Question | Type | " + " | ".join(f"Rang {n}" for n in names) + " | Meilleur |",
       "|---|---|" + "---|" * (len(names) + 1)]
md += [f"| {a} | {b} | " + " | ".join(c) + f" | {d} |" for a, b, c, d in summary]

stats = []
for n in names:
    bests = best_by_sys[n]
    if bests:
        hit = sum(1 for b in bests if b and b <= TOPN) / len(bests)
        mrr = sum((1 / b) if b and b <= TOPN else 0 for b in bests) / len(bests)
        at1 = sum(1 for b in bests if b == 1) / len(bests)
        stats.append((n, hit, mrr, at1, len(bests)))
md += ["", f"### Métriques (top {TOPN}, questions avec fichier attendu)", "",
       f"| Système | hit@{TOPN} | MRR | rang 1 | N |", "|---|---|---|---|---|"]
md += [f"| {n} | {h:.3f} | {m:.3f} | {a:.3f} | {c} |" for n, h, m, a, c in stats]
md.append("")

stem = f"retrieval_report_{SPLIT}_{TAG}"
with open(f"{stem}.md", "w", encoding="utf-8") as f:
    f.write("\n".join(md))
with open(f"{stem}.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter=";")
    w.writeheader()
    w.writerows(rows)

print("\nQuestion | Type | " + " | ".join(names) + " | Meilleur")
for a, b, c, d in summary:
    print(f"{a} | {b} | " + " | ".join(c) + f" | {d}")
print(f"\nMétriques (top {TOPN}) :")
for n, h, m, a, c in stats:
    print(f"  {n:8s} hit@{TOPN}={h:.3f}  MRR={m:.3f}  rang1={a:.3f}  (N={c})")
print(f"\nFichiers créés : {stem}.md et {stem}.csv")