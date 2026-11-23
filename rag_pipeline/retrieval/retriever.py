from llama_index.core.schema import NodeWithScore, TextNode

from rag_pipeline import config
from rag_pipeline.ingestion.embedder import get_embedder
from rag_pipeline.retrieval.reranker import rerank
from rag_pipeline.store import get_collection
from rag_pipeline.retrieval.hybrid import retrieve_hybrid
_embedder = None
_collection = None


def retrieve(question: str, use_reranker: bool | None = None) -> list[NodeWithScore]:
    global _embedder, _collection
    if config.ACTIVE_EMBEDDER == "hybrid":
        return retrieve_hybrid(question, use_reranker)
    if _embedder is None:
        _embedder = get_embedder()
        _collection = get_collection(_embedder.name)

    res = _collection.query(
        query_embeddings=[_embedder.embed_query(question)],
        n_results=config.TOP_K_RETRIEVE,
        include=["documents", "metadatas", "distances"],
    )
    nodes = [
        NodeWithScore(node=TextNode(text=doc, metadata=meta), score=1 - dist)
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]
    use = config.USE_RERANKER if use_reranker is None else use_reranker
    return rerank(question, nodes) if use else nodes[: config.TOP_K_FINAL]


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or input("Question : ")
    for n in retrieve(q):
        m = n.node.metadata
        print(f"{n.score:.3f}  {m['path']}  (lignes {m['start_line']}-{m['end_line']})  [{m['context']}]")