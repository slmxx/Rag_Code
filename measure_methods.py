"""Mesure la taille des méthodes Java / TypeScript du corpus (en caractères non blancs).
Sert à choisir MAX_CHUNK_CHARS.   Lancer :  python measure_methods.py
"""
import math
import statistics

from rag_pipeline.ingestion.chunker import _Lines, _SymbolIndex
from rag_pipeline.ingestion.loader import load_corpus

SEUILS = (1000, 1500, 2000, 3000, 4000)


def centile(sorted_sizes, pct):
    """Rang le plus proche : le résultat est toujours une vraie taille mesurée."""
    return sorted_sizes[max(0, math.ceil(pct / 100 * len(sorted_sizes)) - 1)]


docs = load_corpus()
for lang in ("java", "typescript"):
    for is_test in (False, True):
        rows = []                                    # (taille, fichier, méthode)
        for d in docs:
            if d.metadata["language"] != lang or d.metadata["is_test"] != is_test:
                continue
            lines, index = _Lines(d.text), _SymbolIndex(d.text, lang)
            rows += [(lines.size(m.start, m.end), d.metadata["path"], m.name) for m in index.methods]
        if not rows:
            continue
        sizes = sorted(r[0] for r in rows)
        print(f"\n=== {lang} - {'TESTS' if is_test else 'PRODUCTION'} : {len(rows)} méthodes")
        print(f"médiane {int(statistics.median(sizes))} | p90 {centile(sizes, 90)} | p95 {centile(sizes, 95)} | max {sizes[-1]}")
        for seuil in SEUILS:
            n = sum(s > seuil for s in sizes)
            print(f"  plus grandes que {seuil:>4} : {n:>3} méthodes ({100 * n / len(sizes):.1f} %)")
        print("  les 5 plus grosses :")
        for size, path, name in sorted(rows, reverse=True)[:5]:
            print(f"    {size:>5}  {name}  ({path.split('/')[-1]})")