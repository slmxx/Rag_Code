"""Reranker : réordonne les candidats du retriever et garde les TOP_K_FINAL meilleurs.

Deux backends, choisis par config.RERANK_BACKEND :
  "local"  : cross-encoder Hugging Face qui tourne sur mon PC (gratuit, sans limite, rien n'est envoyé dehors)
  "voyage" : API Voyage (limitée à 3 requêtes/minute en gratuit)

Le reranker est INDÉPENDANT de l'embedder : on peut appliquer exactement le même aux deux modèles,
ce qui rend la comparaison Voyage vs Gemini équitable.
"""
from llama_index.core.schema import NodeWithScore

from rag_pipeline import config

_voyage_client = None
_cross_encoder = None


def _rerank_voyage(question: str, nodes: list[NodeWithScore]) -> list[NodeWithScore]:
    global _voyage_client
    import voyageai
    if _voyage_client is None:
        _voyage_client = voyageai.Client(api_key=config.get_key("VOYAGE_API_KEY"))
    res = _voyage_client.rerank(
        query=question,
        documents=[n.node.get_content() for n in nodes],
        model=config.RERANK_MODEL,
        top_k=config.TOP_K_FINAL,
    )
    return [NodeWithScore(node=nodes[r.index].node, score=r.relevance_score) for r in res.results]


def _rerank_local(question: str, nodes: list[NodeWithScore]) -> list[NodeWithScore]:
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder     
        name = getattr(config, "LOCAL_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
        print(f"[reranker] chargement de {name} (1er lancement : téléchargement ~2 Go)...")
        _cross_encoder = CrossEncoder(name, max_length=getattr(config, "LOCAL_RERANK_MAX_LEN", 512))
    pairs = [(question, n.node.get_content()) for n in nodes]
    scores = _cross_encoder.predict(pairs, batch_size=8)
    order = sorted(range(len(nodes)), key=lambda i: -float(scores[i]))[: config.TOP_K_FINAL]
    return [NodeWithScore(node=nodes[i].node, score=float(scores[i])) for i in order]


def rerank(question: str, nodes: list[NodeWithScore]) -> list[NodeWithScore]:
    if not nodes:
        return nodes
    backend = getattr(config, "RERANK_BACKEND", "local")
    if backend == "voyage":
        return _rerank_voyage(question, nodes)
    return _rerank_local(question, nodes)
