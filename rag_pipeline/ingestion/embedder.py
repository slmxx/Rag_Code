"""Troisième étape de l'indexation : transformer du texte en vecteurs (embeddings).

Un modèle = une classe, même interface pour tous :

    embedder.embed_documents(textes)  -> une liste de vecteurs   (pour les chunks)
    embedder.embed_query(texte)       -> un vecteur              (pour la question)

- cache sur disque (SQLite) : un texte déjà vectorisé par ce modèle n'est jamais renvoyé à l'API ;
- envoi par lots, bornés en nombre de textes et en tokens estimés ;
- respect des limites de débit (requêtes et tokens par minute) du palier gratuit ;
- reprise automatique après une erreur « trop de requêtes » (429) ou une panne réseau ;
- refus d'envoyer un texte trop long : Gemini le tronquerait en silence.

Modèles : "voyage_code_4" (Voyage AI), "gemini_embedding" (Google), "fake" (test, sans réseau).

"""
from __future__ import annotations

import hashlib
import logging
import math
import re
import sqlite3
import sys
import time
from array import array
from collections import deque
from pathlib import Path
from typing import Callable, Iterator

from rag_pipeline import config

log = logging.getLogger(__name__)

DOCUMENT, QUERY = "document", "query"

# Valeurs de départ PRUDENTES (rpm = requêtes/min, tpm = tokens/min). Les vraies limites de ton compte sont dans
# le tableau de bord Voyage (Organization > Rate Limits) : si elles sont plus hautes, règle-les dans config.py :
#   RATE_LIMITS = {"voyage_code_4": {"rpm": 2000, "tpm": 8_000_000}}
_DEFAULT_LIMITS = {
    "voyage_code_4": {"rpm": 3, "tpm": 10_000},
    "gemini_embedding": {"rpm": 90, "tpm": 28_000},
}
_DEFAULT_BATCH_TOKENS = {"voyage_code_4": 8_000, "gemini_embedding": 20_000}   # tokens estimés par requête
_DEFAULT_BATCH_ITEMS = {"voyage_code_4": 128, "gemini_embedding": 20}          # textes par requête


def est_tokens(text: str) -> int:
    """Estimation prudente (CHARS_PER_TOKEN dans config.py). Le vrai compte vient du fournisseur."""
    return math.ceil(len(text) / getattr(config, "CHARS_PER_TOKEN", 3.0))


class TextTooLongError(ValueError):
    """Un texte dépasse le plafond de tokens du modèle : il serait tronqué."""


# ======================================================================
#  Cache disque
# ======================================================================
def _f32(vector) -> list[float]:
    """Les vecteurs sont gardés en float32 : même valeur qu'ils viennent de l'API ou du cache."""
    return array("f", vector).tolist()


class EmbeddingCache:
    """Un fichier SQLite : clé = hash(modèle + type d'usage + texte), valeur = le vecteur."""

    def __init__(self, path: Path | str):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path))
        self._db.execute("CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, vec BLOB NOT NULL)")
        self._db.commit()

    @staticmethod
    def key(model: str, kind: str, text: str) -> str:
        return hashlib.sha256(f"{model}\x00{kind}\x00{text}".encode("utf-8")).hexdigest()

    def get_many(self, keys: list[str]) -> dict[str, list[float]]:
        found: dict[str, list[float]] = {}
        for i in range(0, len(keys), 500):                  # SQLite limite le nombre de paramètres
            part = keys[i:i + 500]
            sql = f"SELECT key, vec FROM vectors WHERE key IN ({','.join('?' * len(part))})"
            for key, blob in self._db.execute(sql, part):
                values = array("f")
                values.frombytes(blob)
                found[key] = values.tolist()
        return found

    def put_many(self, items: dict[str, list[float]]) -> None:
        rows = [(k, array("f", v).tobytes()) for k, v in items.items()]
        self._db.executemany("INSERT OR REPLACE INTO vectors VALUES (?, ?)", rows)
        self._db.commit()

    def __len__(self) -> int:
        return self._db.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]

    def close(self) -> None:
        self._db.close()


# ======================================================================
#  Limites de débit et reprise sur erreur
# ======================================================================
class Throttle:
    """Fenêtre glissante d'une minute : bloque tant que la prochaine requête dépasserait rpm ou tpm."""

    def __init__(self, rpm: int | None = None, tpm: int | None = None,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep):
        self.rpm, self.tpm = rpm, tpm
        self._clock, self._sleep = clock, sleep
        self._events: deque[tuple[float, int]] = deque()     # (instant, tokens)

    def wait(self, tokens: int) -> None:
        while True:
            now = self._clock()
            while self._events and now - self._events[0][0] >= 60:
                self._events.popleft()
            used = sum(t for _, t in self._events)
            rpm_ok = not self.rpm or len(self._events) < self.rpm
            tpm_ok = not self.tpm or not self._events or used + tokens <= self.tpm
            if rpm_ok and tpm_ok:
                return
            self._sleep(max(60 - (now - self._events[0][0]), 0) + 0.1)

    def record(self, tokens: int) -> None:
        self._events.append((self._clock(), tokens))


def call_with_retry(fn: Callable, is_retryable: Callable[[Exception], bool], *, tries: int = 6,
                    base_delay: float = 20.0, max_delay: float = 300.0,
                    sleep: Callable[[float], None] = time.sleep,
                    on_retry: Callable[[Exception, int, float], None] | None = None):
    """Appelle fn(). Si l'erreur est temporaire, attend 20 s, 40 s, 80 s... puis réessaie."""
    for attempt in range(1, tries + 1):
        try:
            return fn()
        except Exception as exc:
            if attempt == tries or not is_retryable(exc):
                raise
            delay = min(base_delay * 2 ** (attempt - 1), max_delay)
            if on_retry:
                on_retry(exc, attempt, delay)
            sleep(delay)


# ======================================================================
#  Classe de base
# ======================================================================
class Embedder:
    name = ""
    model = ""

    def __init__(self, cache: EmbeddingCache | None = None, throttle: Throttle | None = None,
                 sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic,
                 verbose: bool = True):
        self.verbose = verbose
        self._sleep = sleep
        cache_dir = Path(getattr(config, "CACHE_DIR", config.PROJECT_ROOT / ".embedding_cache"))
        self.cache = cache if cache is not None else EmbeddingCache(cache_dir / f"{self.name}.sqlite3")
        limits = {**_DEFAULT_LIMITS.get(self.name, {}), **getattr(config, "RATE_LIMITS", {}).get(self.name, {})}
        self.throttle = throttle if throttle is not None else Throttle(
            limits.get("rpm"), limits.get("tpm"), clock=clock, sleep=sleep)
        self.batch_tokens = getattr(config, "EMBED_BATCH_TOKENS", {}).get(
            self.name, _DEFAULT_BATCH_TOKENS.get(self.name, 8_000))
        self.batch_items = getattr(config, "EMBED_BATCH_ITEMS", {}).get(
            self.name, _DEFAULT_BATCH_ITEMS.get(self.name, 64))
        self.dim: int | None = None
        self.stats = {"api_calls": 0, "texts_embedded": 0, "cache_hits": 0, "tokens": 0}

    # ---- API publique -------------------------------------------------
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(list(texts), DOCUMENT)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], QUERY)[0]

    # ---- à fournir par chaque modèle ---------------------------------
    def _call_api(self, texts: list[str], kind: str) -> tuple[list[list[float]], int | None]:
        """Un appel réseau. Retourne (vecteurs, tokens facturés ou None si inconnu)."""
        raise NotImplementedError

    def _is_retryable(self, exc: Exception) -> bool:
        return False

    # ---- mécanique commune -------------------------------------------
    def _embed(self, texts: list[str], kind: str) -> list[list[float]]:
        if not texts:
            return []
        keys = [EmbeddingCache.key(self.model, kind, t) for t in texts]
        vectors = self.cache.get_many(list(dict.fromkeys(keys)))
        self.stats["cache_hits"] += len(vectors)
        todo: dict[str, str] = {}                           # textes manquants, sans doublon, dans l'ordre
        for key, text in zip(keys, texts):
            if key not in vectors and key not in todo:
                todo[key] = text
        self._check_lengths(todo)
        if todo:
            self._embed_missing(todo, kind, vectors)
        return [vectors[k] for k in keys]

    @staticmethod
    def _check_lengths(todo: dict[str, str]) -> None:
        too_long = [(t, est_tokens(t)) for t in todo.values() if est_tokens(t) > config.MAX_EMBED_TOKENS]
        if too_long:
            sample = "\n".join(f"  - ≈{n} tokens : {t[:70]!r}…" for t, n in too_long[:5])
            raise TextTooLongError(
                f"{len(too_long)} texte(s) dépassent {config.MAX_EMBED_TOKENS} tokens "
                f"(le modèle les tronquerait en silence) :\n{sample}")

    def _batches(self, items: list[tuple[str, str]]) -> Iterator[list[tuple[str, str]]]:
        batch: list[tuple[str, str]] = []
        tokens = 0
        for key, text in items:
            n = est_tokens(text)
            if batch and (len(batch) >= self.batch_items or tokens + n > self.batch_tokens):
                yield batch
                batch, tokens = [], 0
            batch.append((key, text))
            tokens += n
        if batch:
            yield batch

    def _embed_missing(self, todo: dict[str, str], kind: str, vectors: dict[str, list[float]]) -> None:
        items = list(todo.items())
        batches = list(self._batches(items))
        started, done = time.monotonic(), 0
        for number, batch in enumerate(batches, 1):
            texts = [t for _, t in batch]
            fresh_vectors = self._dispatch(texts, kind)
            if len(fresh_vectors) != len(batch):
                raise RuntimeError(f"{self.name} : {len(fresh_vectors)} vecteurs reçus pour {len(batch)} textes")
            fresh: dict[str, list[float]] = {}
            for (key, _), vector in zip(batch, fresh_vectors):
                vector = _f32(vector)
                if self.dim is None:
                    self.dim = len(vector)
                elif len(vector) != self.dim:
                    raise RuntimeError(f"{self.name} : taille de vecteur {len(vector)} au lieu de {self.dim}")
                fresh[key] = vector
            self.cache.put_many(fresh)                      # sauvegardé tout de suite : une coupure ne perd rien
            vectors.update(fresh)
            self.stats["texts_embedded"] += len(batch)
            done += len(batch)
            if self.verbose and len(batches) > 1:
                elapsed = time.monotonic() - started
                remaining = elapsed / number * (len(batches) - number)
                print(f"  [{self.name}] {done}/{len(items)} textes vectorisés (lot {number}/{len(batches)}) "
                      f"- {elapsed / 60:.1f} min écoulées, ≈ {remaining / 60:.1f} min restantes",
                      file=sys.stderr, flush=True)

    def _dispatch(self, texts: list[str], kind: str) -> list[list[float]]:
        """Attend son tour (limites de débit), envoie, enregistre la consommation."""
        estimate = sum(est_tokens(t) for t in texts)
        self.throttle.wait(estimate)
        vectors, used = self._send(texts, kind)
        self.throttle.record(used or estimate)
        self.stats["tokens"] += used or estimate
        return vectors

    def _send(self, texts: list[str], kind: str) -> tuple[list[list[float]], int | None]:
        def fn():
            self.stats["api_calls"] += 1
            return self._call_api(texts, kind)
        return call_with_retry(fn, self._is_retryable, sleep=self._sleep, on_retry=self._announce_retry)

    def _announce_retry(self, exc: Exception, attempt: int, delay: float) -> None:
        message = f"  [{self.name}] {exc.__class__.__name__} (limite atteinte ou panne passagère) : nouvel essai dans {delay:.0f} s"
        log.warning(message)
        if self.verbose:
            print(message, file=sys.stderr, flush=True)


# ======================================================================
#  Voyage AI
# ======================================================================
class VoyageEmbedder(Embedder):
    """voyage-code-4 : 200 millions de tokens gratuits d'après la page de tarifs de Voyage.
    (voyage-code-3 est classé « ancien modèle », sans tokens gratuits.)
    Pour un autre modèle Voyage : une sous-classe avec un autre `name` et un autre `model`."""
    name = "voyage_code_4"
    model = "voyage-code-4"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        import voyageai
        self._error = voyageai.error
        self._client = voyageai.Client(api_key=config.get_key("VOYAGE_API_KEY"), max_retries=0)

    def _call_api(self, texts, kind):
        # truncation=False : un texte trop long fait une erreur au lieu d'être coupé en silence
        res = self._client.embed(texts, model=self.model, input_type=kind, truncation=False)
        return res.embeddings, res.total_tokens

    def _is_retryable(self, exc):
        retryable = tuple(getattr(self._error, n) for n in (
            "RateLimitError", "ServiceUnavailableError", "ServerError", "Timeout", "TryAgain",
            "APIConnectionError") if hasattr(self._error, n))
        return isinstance(exc, retryable)


# ======================================================================
#  Google Gemini
# ======================================================================
class GeminiEmbedder(Embedder):
    """gemini-embedding-001. Un type d'usage différent pour les documents et pour les questions."""
    name = "gemini_embedding"

    def __init__(self, **kwargs):
        self.model = getattr(config, "GEMINI_EMBED_MODEL", "gemini-embedding-001")
        super().__init__(**kwargs)
        from google import genai
        from google.genai import errors, types
        self._errors, self._types = errors, types
        self._client = genai.Client(api_key=config.get_key("GEMINI_API_KEY"))
        self._single = False                                # passe à True si l'API refuse les lots

    def _task(self, kind: str) -> str:
        if kind == DOCUMENT:
            return getattr(config, "GEMINI_DOC_TASK", "RETRIEVAL_DOCUMENT")
        return getattr(config, "GEMINI_QUERY_TASK", "CODE_RETRIEVAL_QUERY")

    def _call_api(self, texts, kind):
        dim = getattr(config, "GEMINI_DIM", None)           # None = taille par défaut du modèle (3072)
        cfg = self._types.EmbedContentConfig(task_type=self._task(kind), output_dimensionality=dim)
        res = self._client.models.embed_content(model=self.model, contents=texts, config=cfg)
        vectors = [list(e.values) for e in res.embeddings]
        if dim and dim != 3072:                             # seule la taille 3072 est déjà normalisée
            vectors = [[x / (math.sqrt(sum(y * y for y in v)) or 1.0) for x in v] for v in vectors]
        if len(vectors) != len(texts):
            raise _BatchMismatch(f"{len(vectors)} vecteurs pour {len(texts)} textes")
        return vectors, None                                # l'API ne renvoie pas de nombre de tokens

    def _is_retryable(self, exc):
        import httpx
        if isinstance(exc, self._errors.ClientError):
            return exc.code == 429
        return isinstance(exc, (self._errors.ServerError, httpx.TransportError))

    def _dispatch(self, texts, kind):
        if not self._single and len(texts) > 1:
            try:
                return super()._dispatch(texts, kind)
            except _BatchMismatch:
                pass
            except self._errors.ClientError as exc:
                if exc.code != 400:
                    raise
            message = f"  [{self.name}] lots refusés par l'API : passage à un texte par requête"
            log.warning(message)
            if self.verbose:
                print(message, file=sys.stderr, flush=True)
            self._single = True
        return [super(GeminiEmbedder, self)._dispatch([t], kind)[0] for t in texts]


class _BatchMismatch(RuntimeError):
    pass


# ======================================================================
#  Faux modèle : sans réseau, pour tester le pipeline
# ======================================================================
class FakeEmbedder(Embedder):
    """Sacs de mots hachés (camelCase et snake_case découpés). Ne comprend pas le sens, seulement les mots :
    utile pour tester sans clé, et comme point de comparaison « lexical » dans l'évaluation."""
    name = "fake"
    model = "fake-hashed-bag-of-words-v1"
    DIM = 512

    def _call_api(self, texts, kind):
        return [self._vector(t) for t in texts], sum(est_tokens(t) for t in texts)

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.DIM
        spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
        for word in re.findall(r"[A-Za-z]{2,}", spaced):
            h = int(hashlib.md5(word.lower().encode()).hexdigest(), 16)
            vec[h % self.DIM] += 1.0 if (h >> 100) & 1 else -1.0
        vec = [math.copysign(math.log1p(abs(x)), x) for x in vec]
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]


# ======================================================================
#  Fabrique
# ======================================================================
EMBEDDER_CLASSES = {"voyage_code_4": VoyageEmbedder, "gemini_embedding": GeminiEmbedder, "fake": FakeEmbedder}


def get_embedder(name: str | None = None, **kwargs) -> Embedder:
    name = name or config.ACTIVE_EMBEDDER
    if name == "voyage_code_3":
        raise ValueError("voyage_code_3 n'a plus de tokens gratuits (page de tarifs de Voyage) : "
                         "utilise voyage_code_4 dans config.py (EMBEDDERS et ACTIVE_EMBEDDER).")
    if name not in EMBEDDER_CLASSES:
        raise ValueError(f"Modèle inconnu : {name!r}. Choix : {', '.join(EMBEDDER_CLASSES)}")
    return EMBEDDER_CLASSES[name](**kwargs)


# ======================================================================
#  Ligne de commande
# ======================================================================
def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / ((math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))) or 1.0)


def _estimate() -> None:
    from rag_pipeline.ingestion.chunker import chunk_corpus
    from rag_pipeline.ingestion.loader import load_corpus

    chunks = chunk_corpus(load_corpus())
    tokens = sum(est_tokens(c.embed_text) for c in chunks)
    print(f"{len(chunks)} chunks, ≈ {tokens:,} tokens estimés (en-tête compris, {config.CHARS_PER_TOKEN} car./token)")
    for name, lim in _DEFAULT_LIMITS.items():
        lim = {**lim, **getattr(config, "RATE_LIMITS", {}).get(name, {})}
        minutes = max(tokens / lim["tpm"], len(chunks) / _DEFAULT_BATCH_ITEMS[name] / lim["rpm"])
        print(f"  {name:18} limite {lim['tpm']:>6} tokens/min -> au moins ≈ {minutes:.0f} min pour tout indexer")
    print("(le cache évite de refaire ce travail : seuls les chunks nouveaux ou modifiés sont envoyés)")


def _smoke_test(name: str) -> None:
    embedder = get_embedder(name)
    code = "public double computeTotal(Order order) {\n    return order.getItems().stream().mapToDouble(Item::getPrice).sum();\n}"
    question = "Comment est calculé le total d'une commande ?"
    started = time.monotonic()
    doc_vec = embedder.embed_documents([code])[0]
    query_vec = embedder.embed_query(question)
    print(f"{name} ({embedder.model}) : OK en {time.monotonic() - started:.1f} s")
    print(f"  taille des vecteurs : {len(doc_vec)}")
    print(f"  similarité question/code : {_cosine(doc_vec, query_vec):.3f}")
    print(f"  appels API : {embedder.stats['api_calls']} | tokens : {embedder.stats['tokens']} | depuis le cache : {embedder.stats['cache_hits']}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    if sys.argv[1:2] == ["--estimate"]:
        _estimate()
    else:
        _smoke_test(sys.argv[1] if len(sys.argv) > 1 else config.ACTIVE_EMBEDDER)