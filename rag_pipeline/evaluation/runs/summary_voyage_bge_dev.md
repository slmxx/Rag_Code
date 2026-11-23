# Expérience : voyage_bge_dev

- Embedder : `voyage_code_4` | Reranker : `True` | top-k final : `5`
- MAX_CHUNK_CHARS : `1500` | Questions : `23` | Split : `dev`

## Qualité (moyenne ; les questions hors corpus sont exclues des métriques de recherche)

| Métrique | Valeur |
|---|---|
| hit@5 | 0.789 |
| recall@5 | 0.693 |
| mrr | 0.506 |

## Recherche par type de question

| Type | N | hit@5 | recall@5 | mrr |
|---|---|---|---|---|
| architecture | 4 | 0.500 | 0.500 | 0.250 |
| code | 10 | 0.900 | 0.900 | 0.658 |
| hors_corpus | 4 | n/a | n/a | n/a |
| multi_fichiers | 5 | 0.800 | 0.433 | 0.407 |

## Latence par question

| Étape | Médiane (s) | p95 (s) |
|---|---|---|
| Retrieval (embedding requête + Chroma + rerank) | 49.701 | 58.107 |
| Génération (Mistral 7B, Ollama) | 0.0 | 0.0 |
| **Total** | **49.701** | **58.107** |
