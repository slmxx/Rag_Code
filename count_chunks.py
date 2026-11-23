"""Combien de fichiers chargés et combien de chunks, par extension.
    python count_chunks.py          -> tableau par extension
    python count_chunks.py .java    -> la liste des fichiers .java avec leur nombre de chunks
"""
import sys
from collections import defaultdict
from pathlib import PurePosixPath

from rag_pipeline.ingestion.chunker import chunk_document
from rag_pipeline.ingestion.loader import load_corpus

docs = load_corpus()
rows = defaultdict(list)                       # extension -> [(chemin, nb de lignes, nb de chunks)]
for d in docs:
    ext = PurePosixPath(d.metadata["path"]).suffix.lower()
    rows[ext].append((d.metadata["path"], d.metadata["n_lines"], len(chunk_document(d))))

if len(sys.argv) > 1:
    ext = sys.argv[1] if sys.argv[1].startswith(".") else "." + sys.argv[1]
    print(f"{len(rows[ext])} fichiers {ext}\n")
    print(f"{'chunks':>7}{'lignes':>8}  fichier")
    for path, n_lines, n in sorted(rows[ext], key=lambda r: -r[2]):
        print(f"{n:>7}{n_lines:>8}  {path}")
else:
    print(f"{'ext':12}{'fichiers':>9}{'chunks':>8}{'chunks/fichier':>16}{'fichiers à 1 chunk':>20}  fichier avec le plus de chunks")
    for ext, lst in sorted(rows.items(), key=lambda kv: -sum(r[2] for r in kv[1])):
        total = sum(r[2] for r in lst)
        top = max(lst, key=lambda r: r[2])
        print(f"{ext:12}{len(lst):>9}{total:>8}{total / len(lst):>16.1f}{sum(r[2] == 1 for r in lst):>20}"
              f"  {top[0].split('/')[-1]} ({top[2]})")
    n_files = sum(len(v) for v in rows.values())
    n_chunks = sum(r[2] for v in rows.values() for r in v)
    print(f"\nTOTAL : {n_files} fichiers -> {n_chunks} chunks")