"""Chargement du corpus avec LlamaIndex (SimpleDirectoryReader).

Ce que fait LlamaIndex : parcourir les dossiers et lire les fichiers.
Ce que j'ai ajouté : métadonnées, nettoyage du texte, tri, rejet des fichiers illisibles.
"""
from __future__ import annotations

import hashlib
import logging
from collections import Counter
from functools import partial
from pathlib import Path

from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import Document

from rag_pipeline import config

log = logging.getLogger(__name__)



# On ajoute d'autres metadonnées dans la fct build metadata, comme file_path, language, kind(is it code ,service or config)
def build_metadata(file_path: str, root: Path) -> dict:
    path = Path(file_path).resolve()
    rel = path.relative_to(root)
    language, kind = config.INCLUDE_EXTS[path.suffix.lower()]
    if path.name in config.AGENT_INSTRUCTION_FILES:
        kind = "agent_instructions"
    folders = {p.lower() for p in rel.parts[:-1]}

    return {
        "path": rel.as_posix(),      # toujours avec "/"
        "file_name": path.name,
        "language": language,
        "kind": kind,                # code | template | config | doc
        "service": rel.parts[0] if len(rel.parts) > 1 else "(racine)",
        "is_test": bool(folders & {"test", "tests"})
                   or path.name.endswith(("Test.java", "Tests.java", "IT.java", ".spec.ts")),
        "size_bytes": path.stat().st_size,
    }


def _clean(doc: Document) -> Document:
    """Texte identique partout : sans BOM, fins de ligne \\n."""
    text = doc.text.lstrip("\ufeff").replace("\r\n", "\n")
    doc.set_content(text)
    doc.metadata["sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    doc.metadata["n_lines"] = text.count("\n") + 1
    doc.id_ = doc.metadata["path"]                                # ID stable = chemin
    return doc


def load_corpus(root: Path | None = None) -> list[Document]:
    root = Path(root or config.CORPUS_DIR).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Corpus introuvable : {root}")

    reader = SimpleDirectoryReader(
        input_dir=str(root),
        recursive=True,                                           
        required_exts=sorted(config.INCLUDE_EXTS),
        exclude=[f"**/{d}/**" for d in sorted(config.EXCLUDE_DIRS)],
        exclude_hidden=False,
        encoding="utf-8",
        errors="replace",                                         # par défaut: "ignore" (supprime des caractères EN SILENCE)
        raise_on_error=False,
        file_metadata=partial(build_metadata, root=root),         
    )
    docs = [_clean(d) for d in reader.load_data()]

    kept = []
    for d in docs:
        m = d.metadata
        if m["file_name"] in config.EXCLUDE_FILES or m["file_name"].endswith(".d.ts"):
            continue
        if m["size_bytes"] > config.MAX_FILE_BYTES:
            log.info("Trop gros, ignoré : %s", m["path"])
            continue
        if not d.text.strip():
            continue
        if "\ufffd" in d.text:                                    # octet illisible remplacé par "�"
            log.warning("Fichier non UTF-8, ignoré : %s", m["path"])
            continue
        kept.append(d)

    kept.sort(key=lambda d: d.metadata["path"])                   # ordre fixe, indépendant de l'OS
    return kept


def summarize(docs: list[Document]) -> None:
    print(f"{len(docs)} fichiers, {sum(d.metadata['n_lines'] for d in docs)} lignes, "
          f"{sum(d.metadata['is_test'] for d in docs)} fichiers de test")
    for title in ("kind", "language", "service"):
        print(f"\nPar {title} :")
        for name, n in Counter(d.metadata[title] for d in docs).most_common():
            print(f"  {name:28} {n}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    summarize(load_corpus())