# Expérience : gen_dev_voyage

- Embedder : `voyage_code_4` | Reranker : `False` | top-k final : `5`
- MAX_CHUNK_CHARS : `1500` | Questions : `23` | Split : `dev`

## Qualité (moyenne ; les questions hors corpus sont exclues des métriques de recherche)

| Métrique | Valeur |
|---|---|
| hit@5 | 1.000 |
| recall@5 | 0.921 |
| mrr | 0.758 |

## Recherche par type de question

| Type | N | hit@5 | recall@5 | mrr |
|---|---|---|---|---|
| architecture | 4 | 1.000 | 1.000 | 0.613 |
| code | 10 | 1.000 | 1.000 | 0.770 |
| hors_corpus | 4 | n/a | n/a | n/a |
| multi_fichiers | 5 | 1.000 | 0.700 | 0.850 |

## Latence par question

| Étape | Médiane (s) | p95 (s) |
|---|---|---|
| Retrieval (embedding requête + Chroma + rerank) | 0.023 | 0.042 |
| Génération (Mistral 7B, Ollama) | 151.559 | 261.736 |
| **Total** | **151.578** | **261.762** |
