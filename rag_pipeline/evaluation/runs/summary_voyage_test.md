# Expérience : voyage_test

- Embedder : `voyage_code_4` | Reranker : `False` | top-k final : `5`
- MAX_CHUNK_CHARS : `1500` | Questions : `17` | Split : `test`

## Qualité (moyenne ; les questions hors corpus sont exclues des métriques de recherche)

| Métrique | Valeur |
|---|---|
| hit@5 | 1.000 |
| recall@5 | 0.917 |
| mrr | 0.824 |

## Recherche par type de question

| Type | N | hit@5 | recall@5 | mrr |
|---|---|---|---|---|
| architecture | 3 | 1.000 | 1.000 | 0.833 |
| code | 7 | 1.000 | 1.000 | 0.886 |
| hors_corpus | 3 | n/a | n/a | n/a |
| multi_fichiers | 4 | 1.000 | 0.708 | 0.708 |

## Latence par question

| Étape | Médiane (s) | p95 (s) |
|---|---|---|
| Retrieval (embedding requête + Chroma + rerank) | 1.095 | 59.126 |
| Génération (Mistral 7B, Ollama) | 0.0 | 0.0 |
| **Total** | **1.095** | **59.126** |
