# Expérience : hybrid_bge_dev

- Embedder : `hybrid` | Reranker : `True` | top-k final : `5`
- MAX_CHUNK_CHARS : `1500` | Questions : `23` | Split : `dev`

## Qualité (moyenne ; les questions hors corpus sont exclues des métriques de recherche)

| Métrique | Valeur |
|---|---|
| hit@5 | 0.842 |
| recall@5 | 0.737 |
| mrr | 0.546 |

## Recherche par type de question

| Type | N | hit@5 | recall@5 | mrr |
|---|---|---|---|---|
| architecture | 4 | 0.750 | 0.750 | 0.500 |
| code | 10 | 0.900 | 0.900 | 0.603 |
| hors_corpus | 4 | n/a | n/a | n/a |
| multi_fichiers | 5 | 0.800 | 0.400 | 0.467 |

## Latence par question

| Étape | Médiane (s) | p95 (s) |
|---|---|---|
| Retrieval (embedding requête + Chroma + rerank) | 84.458 | 90.65 |
| Génération (Mistral 7B, Ollama) | 0.0 | 0.0 |
| **Total** | **84.458** | **90.65** |
