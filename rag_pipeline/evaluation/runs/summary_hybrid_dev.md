# Expérience : hybrid_dev

- Embedder : `hybrid` | Reranker : `False` | top-k final : `5`
- MAX_CHUNK_CHARS : `1500` | Questions : `23` | Split : `dev`

## Qualité (moyenne ; les questions hors corpus sont exclues des métriques de recherche)

| Métrique | Valeur |
|---|---|
| hit@5 | 0.947 |
| recall@5 | 0.886 |
| mrr | 0.765 |

## Recherche par type de question

| Type | N | hit@5 | recall@5 | mrr |
|---|---|---|---|---|
| architecture | 4 | 1.000 | 1.000 | 0.750 |
| code | 10 | 1.000 | 1.000 | 0.833 |
| hors_corpus | 4 | n/a | n/a | n/a |
| multi_fichiers | 5 | 0.800 | 0.567 | 0.640 |

## Latence par question

| Étape | Médiane (s) | p95 (s) |
|---|---|---|
| Retrieval (embedding requête + Chroma + rerank) | 0.019 | 0.024 |
| Génération (Mistral 7B, Ollama) | 0.0 | 0.0 |
| **Total** | **0.019** | **0.024** |
