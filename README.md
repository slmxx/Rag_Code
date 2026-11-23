# 🛒 CodeRAG : assistant RAG pour codebases e-commerce (Java / TypeScript)

> Pose des questions en langage naturel sur une architecture microservices **Spring Boot + Angular** et obtiens des réponses **sourcées**, générées en local par **Mistral 7B**. Le pipeline est **évalué de bout en bout** : retrieval (hit@5, recall@5, MRR) puis génération (RAGAS).

**Mots-clés :** RAG · AST chunking (tree-sitter, cAST) · embeddings de code · Voyage Code 4 · Gemini Embedding · ChromaDB · Reciprocal Rank Fusion · cross-encoder reranking · Mistral 7B · Ollama · RAGAS · LLM-as-a-judge · golden set dev/test · hit@5 / recall@5 / MRR · cache d'embeddings · indexation incrémentale · Streamlit

---

## 🎬 Démo

<!-- TODO : GIF ou vidéo courte de l'interface Streamlit -->

_(démo à venir)_

---

## ✨ Ce que fait le projet:

- **Ingestion** d'un dépôt de code : Java, TypeScript, HTML, YAML, properties, XML, SQL, GraphQL, conf, Markdown, shell.
- **Chunking structurel** : découpe selon la structure du code (classes, méthodes), pas à longueur fixe.
- **Indexation vectorielle** dans ChromaDB, avec deux modèles d'embedding comparables côte à côte.
- **Question → top-5 extraits → Mistral 7B** (local, Ollama), avec un prompt qui impose le **refus** quand le contexte ne contient pas la réponse.
- **Évaluation reproductible** : 40 questions annotées, séparation dev/test, métriques de retrieval puis RAGAS.
- **Interface Streamlit** aux couleurs Mistral.

## 🧱 Architecture:

```
Dépôt Git
   │
   ▼
Loader (LlamaIndex SimpleDirectoryReader + métadonnées custom, SHA-256, exclusion des tests)
   │
   ▼
Chunker hybride ── Java / TypeScript : AST (cAST + tree-sitter)
   │              └ YAML, XML, HTML, MD, SQL, GraphQL, conf… : découpe par structure
   ▼
Embeddings (Voyage Code 4 | Gemini Embedding)  ←── cache SQLite, rate limiting, retry
   │
   ▼
ChromaDB (une collection par embedder)
   │
   ▼
Retrieval ──(option)──►  RRF hybride  /  reranker bge-reranker-v2-m3
   │
   ▼
Mistral 7B (Ollama, T = 0,1) ──► réponse sourcée ou refus
   │
   ▼
Évaluation : hit@5 · recall@5 · MRR  +  RAGAS (juge : Gemini 3.5 Flash Lite)
```

## 🔬 Choix techniques

### Loading

LlamaIndex (`SimpleDirectoryReader`) avec une fonction de métadonnées personnalisée : chemin relatif portable, langage, type (`code` / `template` / `config` / `doc`), service, `is_test`, hash **SHA-256** du contenu normalisé (BOM retiré, fins de ligne uniformisées). Les fichiers de test sont exclus de l'index pour ne pas polluer le top-k. Les fichiers de plus de 100 Ko (généralement générés ou de données) sont ignorés. Couverture vérifiée contre la liste des fichiers Git : **264 fichiers**.

### Chunking structurel (AST)

- **Java / TypeScript** : découpe par AST (`ASTChunker`, cAST) avec `tree-sitter` pour retrouver classes, méthodes et lignes.
- **Autres formats** : stratégies dédiées (clés YAML, éléments XML, titres Markdown, instructions SQL, blocs à accolades, balises HTML, blocs séparés par lignes vides pour `.properties` et `.sh`).
- **Tailles** : plafond de 2 048 tokens (contrainte de Gemini, pour que les deux embedders reçoivent exactement les mêmes chunks), maximum de 1 500 caractères non blancs, minimum de 150. Choix appuyé sur l'analyse de **432 méthodes** : 99,1 % tiennent sous 1 500 caractères non blancs.
- **Méthodes trop longues** : découpées en plusieurs chunks, chacun garde la **signature** dans son en-tête (`Début de` / `Suite de`).
- **En-tête vectorisé avec le code** : `Fichier / Service / Contexte / Symboles`. Les métadonnées (service, type, lignes, hash) servent à filtrer, pas à l'embedding.
- **Identifiant** `chemin::numéro::hash` → indexation **incrémentale** : seuls les chunks modifiés sont ré-embeddés (cache SQLite).
- **22 tests** (15 sur fichiers fabriqués, 7 sur les 264 fichiers réels) : aucune ligne perdue, signatures conservées, ids uniques, plafond respecté.
  Résultat : **757 chunks**.

### Embeddings et recherche

Deux modèles comparés sur les mêmes chunks : **voyage-code-4** et **gemini-embedding-001**, plus une fusion **RRF** (Reciprocal Rank Fusion) et un **reranker cross-encoder local** (`BAAI/bge-reranker-v2-m3`). Le pipeline récupère 20 candidats et en transmet **5** au LLM. Un cache SQLite et un rate limiter (offre gratuite Voyage : 3 RPM / 10K TPM) rendent les réindexations peu coûteuses.

### Génération

Mistral 7B via Ollama, température 0,1. Le prompt numérote les extraits et impose la phrase de refus _« Le contexte fourni ne contient pas cette information. »_ quand la réponse n'est pas dans le contexte.

## 📏 Méthodologie d'évaluation

- **Golden set : 40 questions annotées** (architecture 7, code 17, multi-fichiers 9, hors corpus 7), avec fichiers pertinents exacts et faits vérifiés dans le dépôt.
- **Split dev (23) / test (17)**, comme en ML classique : le **dev** sert à configurer, comparer et diagnostiquer ; le **test** sert de contrôle final.
- **Retrieval évalué seul, sans LLM** : gratuit, rapide, il isole l'effet de l'embedding. Les questions hors corpus sont exclues de ces métriques (19 questions dev, 14 test).
- **Génération évaluée ensuite avec RAGAS** (juge Gemini 3.5 Flash Lite) : faithfulness, answer relevancy, context precision, context recall. Les refus sur questions hors corpus sont comptés à part.

## 📊 Résultats

### Retrieval

| Dev (19 q.)                      | hit@5 | recall@5    | MRR         |
| -------------------------------- | ----- | ----------- | ----------- |
| Voyage Code 4                    | 0,947 | 0,868       | 0,745       |
| Gemini Embedding                 | 1,000 | 0,904       | 0,688       |
| Hybride RRF                      | 0,947 | 0,886       | 0,765       |
| + reranker bge (chaque embedder) | 0,842 | 0,72 – 0,75 | 0,55 – 0,56 |

| Test (14 q.)      | hit@5     | recall@5  | MRR       |
| ----------------- | --------- | --------- | --------- |
| **Voyage Code 4** | **1,000** | **0,917** | 0,824     |
| Gemini Embedding  | 0,714     | 0,637     | 0,548     |
| Hybride RRF       | 0,929     | 0,815     | **0,839** |

**Cumul 33 questions** : Voyage hit@5 ≈ 97 % / MRR ≈ 0,78 · Gemini ≈ 88 % / 0,63 · Hybride ≈ 94 % / 0,80.

**Configuration retenue : Voyage Code 4, sans reranker, top-5.** Gemini est excellent sur le lot dev mais nettement moins stable sur le test ; Voyage est le plus régulier d'un lot à l'autre, plus simple et moins coûteux que l'hybride, qui n'apporte pas de gain net. Le reranker local dégradait la recherche et coûtait 45 à 87 s par requête sur CPU : écarté.

### Génération (RAGAS)

|                   | Dev (19 q.) | Test (14 q.) | Moyenne (33 q.) |
| ----------------- | ----------- | ------------ | --------------- |
| Faithfulness      | 0,832       | 0,760        | 0,80            |
| Answer relevancy  | 0,833       | 0,749        | 0,80            |
| Context precision | 0,647       | 0,717        | 0,68            |
| Context recall    | 0,860       | 0,881        | 0,87            |

Même prompt, même modèle et même configuration entre dev et test pour la génération. **Questions hors corpus refusées 7 fois sur 7** (4/4 dev, 3/3 test ; détection par mots-clés).

## 🧠 Enseignements

1. **Évaluer la recherche seule avant le LLM** : plus simple, et l'embedding est isolé.
2. **Un classement peut s'inverser d'un échantillon à l'autre** (Gemini : 1er en dev, dernier en test). Cette instabilité est une information en soi ; décider sur plusieurs lots est plus fiable que sur un seul.
3. **Un reranker généraliste n'est pas automatiquement un gain** sur du code : ici il dégrade les résultats (hypothèses non testées : modèle généraliste, question en français face à du code en anglais, extraits tronqués).
4. **Mesurer avant d'optimiser** : analyse d'erreurs question par question pour savoir si le problème vient de la recherche ou de la génération.

## ⚠️ Limites

- 40 questions et un seul dépôt : un écart de quelques points est dans le bruit (sur 19 questions, une question = 5 points).
- Le juge RAGAS est un modèle léger : notes indicatives.
- Refus détectés par mots-clés.
- La configuration finale s'appuie sur les deux lots cumulés (33 questions).
- Piste : GraphRAG / graphe de dépendances pour compléter le vectoriel sur les questions multi-fichiers.

## 🚀 Installation et utilisation

**Prérequis :** Python 3.11, [Ollama](https://ollama.com), une clé Voyage AI et une clé Gemini (offres gratuites suffisantes).

```bash
git clone https://github.com/slmxx/Rag_Code.git
cd Rag_Code
python -m venv .venv
.venv\Scripts\activate            # Linux / macOS : source .venv/bin/activate
pip install -r requirements.txt

# Corpus : le dépôt analysé, à cloner dans data/ecommerce
git clone https://github.com/dancodingbr/ecommerce data/ecommerce

# Clés API : copier .env.exemple en .env puis remplir
#   VOYAGE_API_KEY=...
#   GEMINI_API_KEY=...

# Modèle de génération
ollama pull mistral:7b
```

```bash
# 1. Indexer (embedder choisi par ACTIVE_EMBEDDER dans rag_pipeline/config.py)
python -m rag_pipeline.ingestion.embedder

# 2. Interface
streamlit run app.py

# 3. Tests du chunker et de l'embedder
pytest rag_pipeline/tests

# 4. Évaluation du retrieval (sans LLM)
python -m rag_pipeline.evaluation.run_eval --embedder voyage_code_4 --no-generation --split dev --label dev_voyage

# 5. Évaluation avec génération, puis RAGAS
python -m rag_pipeline.evaluation.run_eval --embedder voyage_code_4 --split test --label gen_test_voyage
python run_ragas.py judge gen_test_voyage
```

## 📚 Corpus et licence:

Le corpus analysé est le dépôt public [`dancodingbr/ecommerce`](https://github.com/dancodingbr/ecommerce) (licence MIT), utilisé uniquement comme base de test et non redistribué ici. Le code de ce projet est sous licence MIT (voir `LICENSE`).

## 👩‍💻 Auteure:

**Salma** (@salmadd), élève-ingénieure en Data Science et IA, à la recherche d'un stage PFE en IA / NLP à partir de février 2027.
