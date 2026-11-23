# Expérience : hybrid_test

- Embedder : `hybrid` | Reranker : `False` | top-k final : `5`
- MAX_CHUNK_CHARS : `1500` | Questions : `17` | Split : `test`

## Qualité (moyenne ; les questions hors corpus sont exclues des métriques de recherche)

| Métrique | Valeur |
|---|---|
| hit@5 | 0.929 |
| recall@5 | 0.815 |
| mrr | 0.839 |

## Recherche par type de question

| Type | N | hit@5 | recall@5 | mrr |
|---|---|---|---|---|
| architecture | 3 | 0.667 | 0.667 | 0.667 |
| code | 7 | 1.000 | 1.000 | 1.000 |
| hors_corpus | 3 | n/a | n/a | n/a |
| multi_fichiers | 4 | 1.000 | 0.604 | 0.688 |

## Latence par question

| Étape | Médiane (s) | p95 (s) |
|---|---|---|
| Retrieval (embedding requête + Chroma + rerank) | 0.01 | 0.013 |
| Génération (Mistral 7B, Ollama) | 0.0 | 0.0 |
| **Total** | **0.01** | **0.013** |
