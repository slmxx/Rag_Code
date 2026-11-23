# Expérience : voyage_recherche

- Embedder : `voyage_code_4` | Reranker : `False` | top-k final : `5`
- MAX_CHUNK_CHARS : `1500` | Questions : `10`

## Qualité (moyenne)

| Métrique | Valeur |
|---|---|
| generation_failed | 0.000 |
| hit@5 | 0.900 |
| recall@5 | 0.800 |
| mrr | 0.567 |

## Latence par question

| Étape | Médiane (s) | p95 (s) |
|---|---|---|
| Retrieval (embedding requête + Chroma + rerank) | 0.009 | 0.011 |
| Génération (Mistral 7B, Ollama) | 0.0 | 0.0 |
| **Total** | **0.009** | **0.011** |
