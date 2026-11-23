"""Tests de l'embedder. Aucun appel réseau : Voyage et Gemini sont remplacés par de faux clients.

Lancer :  python -m pytest llamaindex_pipeline/tests/test_embedder.py -v
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from rag_pipeline import config
from rag_pipeline.ingestion import embedder as emb


def no_sleep(_seconds):
    pass


@pytest.fixture
def cache():
    return emb.EmbeddingCache(":memory:")


class CountingFake(emb.FakeEmbedder):
    """Faux modèle qui note chaque « appel réseau »."""

    def __init__(self, **kwargs):
        self.calls: list[tuple[str, list[str]]] = []
        super().__init__(**kwargs)

    def _call_api(self, texts, kind):
        self.calls.append((kind, list(texts)))
        return super()._call_api(texts, kind)


def make(cache, **kwargs) -> CountingFake:
    return CountingFake(cache=cache, verbose=False, sleep=no_sleep, **kwargs)


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b)) / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


# ----------------------------------------------------------------------
#  Cache, doublons, lots
# ----------------------------------------------------------------------
def test_second_call_comes_from_the_cache(cache):
    e = make(cache)
    first = e.embed_documents(["foo bar", "baz qux"])
    calls_after_first = len(e.calls)
    second = e.embed_documents(["foo bar", "baz qux"])
    assert first == second
    assert len(e.calls) == calls_after_first, "le cache aurait dû éviter tout nouvel appel"
    assert e.stats["cache_hits"] == 2


def test_duplicates_are_embedded_once_and_order_is_kept(cache):
    e = make(cache)
    out = e.embed_documents(["alpha", "beta", "alpha"])
    assert out[0] == out[2] and out[0] != out[1]
    assert sum(len(texts) for _, texts in e.calls) == 2


def test_document_and_query_are_cached_separately(cache):
    e = make(cache)
    e.embed_documents(["same text"])
    e.embed_query("same text")
    assert [kind for kind, _ in e.calls] == ["document", "query"]


def test_batches_respect_the_item_limit(cache):
    e = make(cache)
    e.batch_items = 3
    e.embed_documents([f"text number {i}" for i in range(10)])
    assert [len(texts) for _, texts in e.calls] == [3, 3, 3, 1]


def test_batches_respect_the_token_limit(cache, monkeypatch):
    monkeypatch.setattr(config, "CHARS_PER_TOKEN", 1.0)          # 1 caractère = 1 token
    e = make(cache)
    e.batch_tokens = 25
    e.embed_documents([f"xxxxxxxxx{i}" for i in range(5)])         # 10 tokens chacun
    assert [len(texts) for _, texts in e.calls] == [2, 2, 1]


def test_cache_survives_between_instances(tmp_path):
    path = tmp_path / "cache.sqlite3"
    first = CountingFake(cache=emb.EmbeddingCache(path), verbose=False)
    v1 = first.embed_documents(["texte persisté"])
    first.cache.close()
    second = CountingFake(cache=emb.EmbeddingCache(path), verbose=False)
    v2 = second.embed_documents(["texte persisté"])
    assert v1 == v2 and second.calls == []


# ----------------------------------------------------------------------
#  Garde-fous
# ----------------------------------------------------------------------
def test_too_long_text_is_refused_before_any_call(cache):
    e = make(cache)
    too_long = "x" * int(config.MAX_EMBED_TOKENS * config.CHARS_PER_TOKEN + 10)
    with pytest.raises(emb.TextTooLongError):
        e.embed_documents(["texte normal", too_long])
    assert e.calls == [], "rien ne doit partir vers l'API si un texte est trop long"


def test_wrong_number_of_vectors_is_an_error(cache):
    class Short(CountingFake):
        def _call_api(self, texts, kind):
            vectors, used = super()._call_api(texts, kind)
            return vectors[:-1], used

    with pytest.raises(RuntimeError):
        Short(cache=cache, verbose=False).embed_documents(["a b", "c d"])


def test_inconsistent_vector_size_is_an_error(cache):
    class Wobbly(CountingFake):
        def _call_api(self, texts, kind):
            vectors, used = super()._call_api(texts, kind)
            return [v[: 10 + len(self.calls)] for v in vectors], used

    e = Wobbly(cache=cache, verbose=False)
    e.batch_items = 1
    with pytest.raises(RuntimeError):
        e.embed_documents(["premier texte", "second texte"])


def test_fake_embedder_ranks_shared_words_higher(cache):
    e = make(cache)
    docs = e.embed_documents(["calculateOrderTotal sums the items price", "send email notification to user"])
    query = e.embed_query("how is the order total calculated")
    assert cosine(query, docs[0]) > cosine(query, docs[1])


# ----------------------------------------------------------------------
#  Limites de débit
# ----------------------------------------------------------------------
class FakeClock:
    def __init__(self):
        self.t, self.slept = 0.0, []

    def now(self):
        return self.t

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.t += seconds


def test_throttle_waits_when_requests_per_minute_is_reached():
    clock = FakeClock()
    throttle = emb.Throttle(rpm=2, tpm=None, clock=clock.now, sleep=clock.sleep)
    for _ in range(2):
        throttle.wait(10)
        throttle.record(10)
    throttle.wait(10)
    assert clock.t >= 60 and clock.slept


def test_throttle_waits_when_tokens_per_minute_is_reached():
    clock = FakeClock()
    throttle = emb.Throttle(rpm=None, tpm=100, clock=clock.now, sleep=clock.sleep)
    throttle.wait(60)
    throttle.record(60)
    throttle.wait(60)                                    # 60 + 60 > 100 : il faut attendre
    assert clock.t >= 60


def test_throttle_does_not_wait_under_the_limits():
    clock = FakeClock()
    throttle = emb.Throttle(rpm=3, tpm=1000, clock=clock.now, sleep=clock.sleep)
    for _ in range(3):
        throttle.wait(100)
        throttle.record(100)
    assert clock.slept == []


# ----------------------------------------------------------------------
#  Reprise sur erreur
# ----------------------------------------------------------------------
class Flaky(emb.FakeEmbedder):
    def __init__(self, fail_times, retryable=True, **kwargs):
        self.fail_times, self.retryable, self.attempts = fail_times, retryable, 0
        super().__init__(**kwargs)

    def _call_api(self, texts, kind):
        self.attempts += 1
        if self.attempts <= self.fail_times:
            raise TimeoutError("trop de requêtes")
        return super()._call_api(texts, kind)

    def _is_retryable(self, exc):
        return self.retryable and isinstance(exc, TimeoutError)


def test_retry_waits_with_growing_delays(cache):
    slept: list[float] = []
    e = Flaky(2, cache=cache, verbose=False, sleep=slept.append)
    assert len(e.embed_documents(["abc def"])) == 1
    assert slept == [20.0, 40.0] and e.attempts == 3


def test_non_retryable_error_is_raised_immediately(cache):
    slept: list[float] = []
    e = Flaky(5, retryable=False, cache=cache, verbose=False, sleep=slept.append)
    with pytest.raises(TimeoutError):
        e.embed_documents(["abc def"])
    assert slept == [] and e.attempts == 1


def test_gives_up_after_all_tries(cache):
    e = Flaky(99, cache=cache, verbose=False, sleep=no_sleep)
    with pytest.raises(TimeoutError):
        e.embed_documents(["abc def"])
    assert e.attempts == 6


# ----------------------------------------------------------------------
#  Voyage (faux client)
# ----------------------------------------------------------------------
@pytest.fixture
def voyage(monkeypatch):
    import voyageai
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")

    class DummyClient:
        instances: list = []

        def __init__(self, api_key=None, max_retries=0, timeout=None, base_url=None):
            self.api_key, self.max_retries, self.calls, self.fail = api_key, max_retries, [], []
            DummyClient.instances.append(self)

        def embed(self, texts, model, input_type=None, truncation=True, **kwargs):
            self.calls.append({"texts": list(texts), "model": model, "input_type": input_type,
                               "truncation": truncation})
            if self.fail:
                raise self.fail.pop(0)
            return SimpleNamespace(embeddings=[[float(len(t)), 1.0, 0.5] for t in texts],
                                   total_tokens=7 * len(texts))

    monkeypatch.setattr(voyageai, "Client", DummyClient)
    return DummyClient


def test_voyage_sends_the_right_arguments(voyage, cache):
    e = emb.VoyageEmbedder(cache=cache, verbose=False, sleep=no_sleep)
    e.embed_documents(["abc"])
    e.embed_query("une question")
    client = voyage.instances[-1]
    assert client.api_key == "test-key" and client.max_retries == 0
    doc_call, query_call = client.calls
    assert doc_call["input_type"] == "document" and query_call["input_type"] == "query"
    assert doc_call["model"] == "voyage-code-4"
    assert doc_call["truncation"] is False, "un texte trop long doit faire une erreur, pas être coupé en silence"
    assert e.stats["tokens"] == 14, "la consommation réelle renvoyée par l'API doit être comptée"


def test_voyage_retries_when_rate_limited(voyage, cache):
    import voyageai
    slept: list[float] = []
    e = emb.VoyageEmbedder(cache=cache, verbose=False, sleep=slept.append)
    voyage.instances[-1].fail = [voyageai.error.RateLimitError("trop vite")]
    assert len(e.embed_documents(["abc"])) == 1
    assert slept == [20.0]


def test_voyage_does_not_retry_a_bad_key(voyage, cache):
    import voyageai
    slept: list[float] = []
    e = emb.VoyageEmbedder(cache=cache, verbose=False, sleep=slept.append)
    voyage.instances[-1].fail = [voyageai.error.AuthenticationError("clé invalide")]
    with pytest.raises(voyageai.error.AuthenticationError):
        e.embed_documents(["abc"])
    assert slept == []


def test_missing_key_gives_a_clear_message(monkeypatch, cache):
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="VOYAGE_API_KEY"):
        emb.VoyageEmbedder(cache=cache, verbose=False)


# ----------------------------------------------------------------------
#  Gemini (faux client)
# ----------------------------------------------------------------------
@pytest.fixture
def gemini(monkeypatch):
    from google import genai
    from google.genai import errors
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    class DummyModels:
        def __init__(self):
            self.calls, self.fail, self.refuse_batches = [], [], False

        def embed_content(self, model, contents, config=None):
            self.calls.append({"model": model, "n": len(contents), "task": config.task_type})
            if self.fail:
                raise self.fail.pop(0)
            if self.refuse_batches and len(contents) > 1:
                raise errors.ClientError(400, {"error": {"message": "un seul texte par requête"}})
            return SimpleNamespace(embeddings=[SimpleNamespace(values=[float(len(t)), 2.0, 3.0]) for t in contents])

    class DummyClient:
        instances: list = []

        def __init__(self, api_key=None, **kwargs):
            self.api_key, self.models = api_key, DummyModels()
            DummyClient.instances.append(self)

    monkeypatch.setattr(genai, "Client", DummyClient)
    return DummyClient, errors


def test_gemini_uses_different_task_types_for_documents_and_questions(gemini, cache):
    client_class, _ = gemini
    e = emb.GeminiEmbedder(cache=cache, verbose=False, sleep=no_sleep)
    e.embed_documents(["abc"])
    e.embed_query("une question")
    calls = client_class.instances[-1].models.calls
    assert calls[0]["task"] == "RETRIEVAL_DOCUMENT"
    assert calls[1]["task"] == "CODE_RETRIEVAL_QUERY"
    assert calls[0]["model"] == "gemini-embedding-001"


def test_gemini_sends_several_texts_in_one_request(gemini, cache):
    client_class, _ = gemini
    e = emb.GeminiEmbedder(cache=cache, verbose=False, sleep=no_sleep)
    e.embed_documents([f"texte {i}" for i in range(5)])
    assert [c["n"] for c in client_class.instances[-1].models.calls] == [5]


def test_gemini_falls_back_to_one_text_per_request_when_batches_are_refused(gemini, cache):
    client_class, _ = gemini
    e = emb.GeminiEmbedder(cache=cache, verbose=False, sleep=no_sleep)
    models = client_class.instances[-1].models
    models.refuse_batches = True
    out = e.embed_documents(["un", "deux", "trois"])
    assert len(out) == 3
    assert [c["n"] for c in models.calls] == [3, 1, 1, 1]          # un essai en lot, puis un par un
    e.embed_documents(["quatre", "cinq"])
    assert [c["n"] for c in models.calls][4:] == [1, 1], "une fois le lot refusé, on ne réessaie plus en lot"


def test_gemini_retries_on_429(gemini, cache):
    client_class, errors = gemini
    slept: list[float] = []
    e = emb.GeminiEmbedder(cache=cache, verbose=False, sleep=slept.append)
    client_class.instances[-1].models.fail = [errors.ClientError(429, {"error": {"status": "RESOURCE_EXHAUSTED"}})]
    assert len(e.embed_documents(["abc"])) == 1
    assert slept == [20.0]


def test_gemini_does_not_retry_a_bad_request(gemini, cache):
    client_class, errors = gemini
    slept: list[float] = []
    e = emb.GeminiEmbedder(cache=cache, verbose=False, sleep=slept.append)
    client_class.instances[-1].models.fail = [errors.ClientError(400, {"error": {"message": "clé invalide"}})]
    with pytest.raises(errors.ClientError):
        e.embed_documents(["abc"])
    assert slept == []


# ----------------------------------------------------------------------
#  Fabrique
# ----------------------------------------------------------------------
def test_factory(monkeypatch):
    assert isinstance(emb.get_embedder("fake", verbose=False), emb.FakeEmbedder)
    with pytest.raises(ValueError, match="inconnu"):
        emb.get_embedder("nimporte-quoi")
    with pytest.raises(ValueError, match="voyage_code_4"):
        emb.get_embedder("voyage_code_3")                # l'ancien nom renvoie vers le nouveau
    monkeypatch.setattr(config, "ACTIVE_EMBEDDER", "fake")
    assert emb.get_embedder(verbose=False).name == "fake"