"""Latence réelle de la recherche (sans cache d'embeddings) pour chaque configuration.

Usage:
    python latency_check.py          # 6 questions par configuration
    python latency_check.py 8        # 8 questions

Chaque question reçoit un suffixe aléatoire : le texte est nouveau, donc le cache SQLite ne répond pas
et on mesure un vrai appel API. Pause de 21 s entre deux appels pour les configs qui utilisent Voyage
(limite gratuite de 3 requêtes/minute) : compte environ 10 minutes en tout.
La chauffe (chargement du modèle, de bge) n'est pas comptée.
"""
import json
import statistics
import sys
import time
import uuid

from rag_pipeline import config
from rag_pipeline.retrieval import retriever

N = int(sys.argv[1]) if len(sys.argv) > 1 else 6
GOLDEN = "rag_pipeline/evaluation/golden_set_v2.jsonl"
CONFIGS = [("voyage_code_4", False), ("gemini_embedding", False), ("hybrid", False)]
qs = []
for line in open(GOLDEN, encoding="utf-8"):
    if line.strip():
        q = json.loads(line)
        if q.get("split") == "dev" and q.get("type") != "hors_corpus":
            qs.append(q["question"])
qs = qs[:N]


def fresh(text):
    return f"{text} ({uuid.uuid4().hex[:8]})"


uses_voyage = lambda emb: emb in ("voyage_code_4", "hybrid")           # noqa: E731
results = []
for emb, rr in CONFIGS:
    config.ACTIVE_EMBEDDER = emb
    config.USE_RERANKER = rr
    retriever._embedder = None
    retriever._collection = None
    name = f"{emb}{' + rerank' if rr else ''}"
    print(f"\n{name} : chauffe...")
    retriever.retrieve(fresh("warm-up"))
    times = []
    for q in qs:
        if uses_voyage(emb):
            time.sleep(21)
        t0 = time.perf_counter()
        retriever.retrieve(fresh(q))
        times.append(time.perf_counter() - t0)
        print(f"  {times[-1]:.2f} s")
    results.append((name, statistics.median(times), sorted(times)[min(len(times) - 1, int(0.95 * len(times)))], max(times)))
print("\n=== Latence réelle de la recherche (sans cache) ===")
print(f"{'Configuration':34s} {'médiane (s)':>12s} {'p95 (s)':>9s} {'max (s)':>9s}")
for name, med, p95, mx in results:
    print(f"{name:34s} {med:12.2f} {p95:9.2f} {mx:9.2f}")
