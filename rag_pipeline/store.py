"""Quatrième étape de l'indexation : stocker chunks + vecteurs dans ChromaDB.

    python -m llamaindex_pipeline.store            -> indexe (seuls les chunks nouveaux sont vectorisés)
"""
from __future__ import annotations

import logging

import chromadb

from rag_pipeline import config
from rag_pipeline.ingestion.chunker import chunk_corpus
from rag_pipeline.ingestion.embedder import get_embedder
from rag_pipeline.ingestion.loader import load_corpus

log = logging.getLogger(__name__)
BATCH = 500


def collection_name(embedder_name: str | None = None) -> str:
    # une collection par modèle d'embedding : on ne mélange jamais deux espaces vectoriels
    return f"ecommerce_{embedder_name or config.ACTIVE_EMBEDDER}"


def get_collection(embedder_name: str | None = None, create: bool = False):
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    name = collection_name(embedder_name)
    if create:
        return client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})
    return client.get_collection(name)


def build_index(embedder_name: str | None = None) -> None:
    embedder = get_embedder(embedder_name)
    chunks = chunk_corpus(load_corpus())
    col = get_collection(embedder.name, create=True)

    # l'id contient le hash du contenu : même id = chunk inchangé
    existing = set(col.get(include=[])["ids"])
    current = {c.id for c in chunks}

    stale = list(existing - current)
    for i in range(0, len(stale), BATCH):
        col.delete(ids=stale[i:i + BATCH])

    todo = [c for c in chunks if c.id not in existing]
    print(f"{len(chunks)} chunks | {len(existing & current)} déjà indexés | "
          f"{len(todo)} à ajouter | {len(stale)} obsolètes supprimés")

    for i in range(0, len(todo), BATCH):
        part = todo[i:i + BATCH]
        vectors = embedder.embed_documents([c.embed_text for c in part])   # cache + limites gérés
        col.upsert(
            ids=[c.id for c in part],
            embeddings=vectors,
            documents=[c.embed_text for c in part],      # en-tête + code : Mistral voit le chemin du fichier
            metadatas=[c.to_metadata() for c in part],
        )
    print(f"Collection '{col.name}' : {col.count()} chunks")


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.WARNING)
    build_index(sys.argv[1] if len(sys.argv) > 1 else None)