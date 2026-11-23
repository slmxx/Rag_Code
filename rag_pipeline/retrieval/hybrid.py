"""Recherche hybride : fusion des classements Voyage et Gemini par RRF (Reciprocal Rank Fusion).

Chaque chunk reçoit  sum( 1 / (RRF_K + rang) )  sur les deux classements : un chunk bien classé par
l'un OU par l'autre remonte, un chunk bien classé par les deux remonte encore plus.
Les deux index viennent du même chunker : l'id du chunk (path::index::hash) est donc identique
dans les deux collections, c'est ce qui permet de fusionner.

Activation : config.ACTIVE_EMBEDDER = "hybrid"  (retriever.retrieve délègue ici).
"""
from llama_index.core.schema import NodeWithScore, TextNode

from rag_pipeline import config
from rag_pipeline.ingestion.embedder import get_embedder
from rag_pipeline.retrieval.reranker import rerank
from rag_pipeline.store import get_collection

HYBRID_MODELS = ["voyage_code_4", "gemini_embedding"]   
RRF_K = 60                                              # constante standard de la RRF
_parts: dict = {}


def _get(model_name: str):
    if model_name not in _parts:
        old = config.ACTIVE_EMBEDDER
        config.ACTIVE_EMBEDDER = model_name
        try:
            emb = get_embedder()
        finally:
            config.ACTIVE_EMBEDDER = old
        _parts[model_name] = (emb, get_collection(emb.name))
    return _parts[model_name]


def retrieve_hybrid(question: str, use_reranker: bool | None = None) -> list[NodeWithScore]:
    fused: dict[str, float] = {}
    content: dict[str, tuple[str, dict]] = {}
    for name in HYBRID_MODELS:
        emb, col = _get(name)
        res = col.query(
            query_embeddings=[emb.embed_query(question)],
            n_results=config.TOP_K_RETRIEVE,
            include=["documents", "metadatas"],
        )
        for rank, (cid, doc, meta) in enumerate(zip(res["ids"][0], res["documents"][0], res["metadatas"][0]), 1):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank)
            content[cid] = (doc, meta)

    best = sorted(fused, key=fused.get, reverse=True)[: config.TOP_K_RETRIEVE]   # pool de candidats
    nodes = [NodeWithScore(node=TextNode(text=content[c][0], metadata=content[c][1]), score=fused[c]) for c in best]
    use = config.USE_RERANKER if use_reranker is None else use_reranker
    return rerank(question, nodes) if use else nodes[: config.TOP_K_FINAL]