import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

CORPUS_DIR = PROJECT_ROOT / "data" / "ecommerce"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
CACHE_DIR = PROJECT_ROOT / ".embedding_cache"

# Fichiers à indexer : extension -> (langage, type)
INCLUDE_EXTS = {
    ".java": ("java", "code"),
    ".ts": ("typescript", "code"),
    ".html": ("html", "template"),
    ".yml": ("yaml", "config"),
    ".yaml": ("yaml", "config"),
    ".properties": ("properties", "config"),
    ".xml": ("xml", "config"),
    ".sql": ("sql", "config"),
    ".graphqls": ("graphql", "config"),
    ".conf": ("conf", "config"),
    ".md": ("markdown", "doc"),
    ".sh": ("shell", "config"),
    ".py": ("python", "code"),
}
EXCLUDE_DIRS = {".git",".idea", ".vscode", "node_modules", "target", "dist", "build", ".angular"}
EXCLUDE_FILES = {"package-lock.json"}
AGENT_INSTRUCTION_FILES = {"copilot-instructions.md", "CLAUDE.md", "AGENTS.md"}
MAX_FILE_BYTES = 100_000   # au-delà : fichier généré ou données, on l'ignore

# Chunking:
MAX_CHUNK_CHARS = 1500
MIN_CHUNK_CHARS = 150
CHUNK_OVERLAP_NODES = 0
CHARS_PER_TOKEN = 3
# Embeddings:
EMBEDDERS = ("voyage_code_4", "gemini_embedding")
GEMINI_EMBED_MODEL = "gemini-embedding-001"
MAX_EMBED_TOKENS = 2048 
ACTIVE_EMBEDDER = "voyage_code_4"

RATE_LIMITS = {"voyage_code_4": {"rpm": 3, "tpm": 10_000}}

# Recherche:
TOP_K_RETRIEVE = 20  
TOP_K_FINAL = 5 

# Reranking:
RERANK_BACKEND = "local"
LOCAL_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

# Génération:
GENERATION_MODEL = "mistral:7b"        # Mistral 7B via Ollama
OLLAMA_BASE_URL = "http://localhost:11434"
GENERATION_TEMPERATURE = 0.1
GENERATION_TIMEOUT = 900.0

# Évaluation RAGAS:
JUDGE_MODEL = "gemini-3.5-flash-lite"
JUDGE_EMBED_MODEL = "models/gemini-embedding-001"

PIPELINE_DIR = Path(__file__).resolve().parent       # rag_pipeline/
EVAL_DIR = PIPELINE_DIR / "evaluation"
GOLDEN_SET_FILE = EVAL_DIR / "golden_set_v2.jsonl"

# Recherche 
USE_RERANKER = False
def get_key(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Clé manquante : ajoute {name} dans le fichier .env")
    return value