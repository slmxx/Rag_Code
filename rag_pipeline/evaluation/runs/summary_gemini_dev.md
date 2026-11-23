# Expérience : gemini_dev

- Embedder : `gemini_embedding` | Reranker : `False` | top-k final : `5`
- MAX_CHUNK_CHARS : `1500` | Questions : `23` | Split : `dev`

## Qualité (moyenne ; les questions hors corpus sont exclues des métriques de recherche)

| Métrique | Valeur |
|---|---|
| hit@5 | 1.000 |
| recall@5 | 0.904 |
| mrr | 0.688 |

## Recherche par type de question

| Type | N | hit@5 | recall@5 | mrr |
|---|---|---|---|---|
| architecture | 4 | 1.000 | 1.000 | 0.396 |
| code | 10 | 1.000 | 1.000 | 0.845 |
| hors_corpus | 4 | n/a | n/a | n/a |
| multi_fichiers | 5 | 1.000 | 0.633 | 0.607 |

## Latence par question

| Étape | Médiane (s) | p95 (s) |
|---|---|---|
| Retrieval (embedding requête + Chroma + rerank) | 0.311 | 0.328 |
| Génération (Mistral 7B, Ollama) | 0.0 | 0.0 |
| **Total** | **0.311** | **0.328** |
