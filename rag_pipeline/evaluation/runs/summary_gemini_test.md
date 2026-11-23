# Expérience : gemini_test

- Embedder : `gemini_embedding` | Reranker : `False` | top-k final : `5`
- MAX_CHUNK_CHARS : `1500` | Questions : `17` | Split : `test`

## Qualité (moyenne ; les questions hors corpus sont exclues des métriques de recherche)

| Métrique | Valeur |
|---|---|
| hit@5 | 0.714 |
| recall@5 | 0.637 |
| mrr | 0.548 |

## Recherche par type de question

| Type | N | hit@5 | recall@5 | mrr |
|---|---|---|---|---|
| architecture | 3 | 0.333 | 0.333 | 0.333 |
| code | 7 | 0.857 | 0.857 | 0.619 |
| hors_corpus | 3 | n/a | n/a | n/a |
| multi_fichiers | 4 | 0.750 | 0.479 | 0.583 |

## Latence par question

| Étape | Médiane (s) | p95 (s) |
|---|---|---|
| Retrieval (embedding requête + Chroma + rerank) | 0.333 | 0.439 |
| Génération (Mistral 7B, Ollama) | 0.0 | 0.0 |
| **Total** | **0.333** | **0.439** |
