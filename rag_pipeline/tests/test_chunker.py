"""Tests d'intégrité du chunker.  Lancer :  python -m pytest llamaindex_pipeline/tests -v

Deux familles :
- cas synthétiques (petits fichiers fabriqués ici) : ils fixent le comportement attendu ;
- corpus réel (data/ecommerce) : ils vérifient les garanties sur tous les fichiers chargés.
"""
from __future__ import annotations

import pytest
from llama_index.core.schema import Document

from rag_pipeline import config
from rag_pipeline.ingestion import chunker
from rag_pipeline.ingestion.chunker import _Lines, _SymbolIndex, chunk_document, est_tokens, nws


def make_doc(text: str, path: str, language: str, kind: str = "code") -> Document:
    return Document(text=text, metadata={
        "path": path, "service": path.split("/")[0], "language": language,
        "kind": kind, "is_test": False, "sha256": "x"})


def covered_lines(chunks) -> set[int]:
    return {n for c in chunks for n in range(c.start_line, c.end_line + 1)}


def non_blank_lines(text: str) -> set[int]:
    return {i + 1 for i, line in enumerate(text.split("\n")) if line.strip()}


JAVA = """package com.shop;

import java.util.List;

@Service
public class PricingService {

    public double small(int a) { return a; }

    public double computeDiscount(Order o, User u) {
        double d = 0;
%s
        return d;
    }

    public void other() { }
}
""" % "\n".join(f'        if (u.rule{i}() && o.total() > {i}) {{ d += {i}.5; log.info("rule {i} applied"); }}'
                for i in range(40))


@pytest.fixture
def small_limits(monkeypatch):
    monkeypatch.setattr(config, "MAX_CHUNK_CHARS", 600)
    monkeypatch.setattr(config, "MIN_CHUNK_CHARS", 100)
    monkeypatch.setattr(config, "CHUNK_OVERLAP_NODES", 0)


# ----------------------------------------------------------------------
#  Cas synthétiques
# ----------------------------------------------------------------------
def test_long_method_pieces_carry_signature(small_limits):
    chunks = chunk_document(make_doc(JAVA, "svc/PricingService.java", "java"))
    method_chunks = [c for c in chunks if "computeDiscount" in c.header]
    assert len(method_chunks) >= 3, "la méthode longue doit être découpée en plusieurs morceaux"
    for c in method_chunks:
        assert "PricingService" in c.header
        assert "computeDiscount(Order o, User u)" in c.header
    assert any("Début de:" in c.header for c in method_chunks)
    assert any("Suite de:" in c.header and "partie 2/" in c.header for c in method_chunks)


def test_short_methods_are_never_split(small_limits):
    chunks = chunk_document(make_doc(JAVA, "svc/PricingService.java", "java"))
    index = _SymbolIndex(JAVA, "java")
    lines = _Lines(JAVA)
    for m in index.methods:
        if lines.size(m.start, m.end) <= config.MAX_CHUNK_CHARS:
            assert any(c.start_line - 1 <= m.start and m.end <= c.end_line - 1 for c in chunks), \
                f"la méthode {m.name} (assez petite) a été coupée"


def test_no_line_lost_java(small_limits):
    chunks = chunk_document(make_doc(JAVA, "svc/PricingService.java", "java"))
    assert non_blank_lines(JAVA) <= covered_lines(chunks)


def test_small_chunk_merged_with_next_same_file(small_limits):
    text = "package a;\n\nimport b.C;\n\npublic class X {\n  int f() { return 1; }\n}\n"
    chunks = chunk_document(make_doc(text, "svc/X.java", "java"))
    assert len(chunks) == 1
    assert non_blank_lines(text) <= covered_lines(chunks)


def test_yaml_split_by_service(small_limits):
    env = "".join(f"      VAR_{k}: valeur-{k}-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n" for k in range(8))
    svc = lambda n: f"  {n}:\n    image: img/{n}\n    environment:\n{env}"
    text = "version: '3'\nservices:\n" + "".join(svc(n) for n in ("alpha", "beta", "gamma")) + "volumes:\n  data: {}\n"
    chunks = chunk_document(make_doc(text, "docker-compose.yml", "yaml", "config"))
    ctxs = " ".join(c.context for c in chunks)
    for name in ("alpha", "beta", "gamma"):
        assert f"services > {name}" in ctxs
    assert non_blank_lines(text) <= covered_lines(chunks)
    # chaque service est dans un seul chunk
    for name in ("alpha", "beta", "gamma"):
        owners = [c for c in chunks if f"  {name}:" in c.body]
        assert len(owners) == 1


def test_yaml_block_scalar_is_not_split_on_inner_colons(small_limits):
    text = "job:\n  script: |\n    echo a: b\n    key: value\n  other: 1\n"
    chunks = chunk_document(make_doc(text, "ci.yml", "yaml", "config"))
    assert non_blank_lines(text) <= covered_lines(chunks)
    assert not any(c.context.endswith("key") for c in chunks)


def test_markdown_ignores_hash_inside_code_fence(small_limits):
    text = "# Titre\n\nintro\n\n```bash\n# pas un titre\nls\n```\n\n## Suite\n\ntexte\n"
    chunks = chunk_document(make_doc(text, "docs/a.md", "markdown", "doc"))
    assert "pas un titre" not in " ".join(c.context for c in chunks)
    assert any("Titre > Suite" in c.context for c in chunks)


def test_sql_semicolon_inside_string_does_not_split(small_limits):
    text = ("CREATE TABLE a (\n  s VARCHAR(10) DEFAULT 'x; y'\n);\n\n"
            "CREATE TABLE b (\n  id INT\n);\n" + "INSERT INTO b VALUES (1);\n" * 30)
    chunks = chunk_document(make_doc(text, "db/V1.sql", "sql", "config"))
    owner = [c for c in chunks if "x; y" in c.body][0]
    assert "CREATE TABLE a" in owner.body
    assert non_blank_lines(text) <= covered_lines(chunks)


def test_oversized_line_never_cut_in_the_middle(small_limits):
    text = "# titre\n\n" + ("mot " * 400) + "\n"
    chunks = chunk_document(make_doc(text, "docs/long.md", "markdown", "doc"))
    assert any(("mot " * 400).strip() in c.body for c in chunks)


def test_malformed_xml_falls_back_without_crashing(small_limits):
    text = "<project>\n  <a>\n  <b></a>\n</project>\n"
    chunks = chunk_document(make_doc(text, "pom.xml", "xml", "config"))
    assert chunks and non_blank_lines(text) <= covered_lines(chunks)


def test_java_with_syntax_error_still_chunked(small_limits):
    text = "public class A {\n  void f( {\n    int x = ;\n  }\n"
    chunks = chunk_document(make_doc(text, "svc/A.java", "java"))
    assert chunks and non_blank_lines(text) <= covered_lines(chunks)


def test_header_format_is_uniform(small_limits):
    for text, lang, path in ((JAVA, "java", "svc/P.java"), ("a:\n  b: 1\n", "yaml", "svc/a.yml")):
        c = chunk_document(make_doc(text, path, lang))[0]
        first = c.header.split("\n")[:3]
        assert first[0].startswith("Fichier: ") and first[1].startswith("Service: ") and first[2].startswith("Contexte: ")


def test_ids_are_unique(small_limits):
    chunks = chunk_document(make_doc(JAVA, "svc/PricingService.java", "java"))
    assert len({c.id for c in chunks}) == len(chunks)


def test_metadata_is_chroma_compatible(small_limits):
    c = chunk_document(make_doc(JAVA, "svc/PricingService.java", "java"))[0]
    for key, value in c.to_metadata().items():
        assert isinstance(value, (str, int, float, bool)), key


# ----------------------------------------------------------------------
#  Corpus réel : garanties sur tous les fichiers
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def corpus():
    if not config.CORPUS_DIR.is_dir():
        pytest.skip("corpus absent")
    from rag_pipeline.ingestion.loader import load_corpus
    docs = load_corpus()
    return [(d, chunk_document(d)) for d in docs]


def test_corpus_no_line_lost(corpus):
    for doc, chunks in corpus:
        missing = non_blank_lines(doc.text) - covered_lines(chunks)
        assert not missing, f"{doc.metadata['path']} : lignes perdues {sorted(missing)[:5]}"


def test_corpus_every_small_method_is_intact(corpus):
    for doc, chunks in corpus:
        lang = doc.metadata["language"]
        if lang not in ("java", "typescript"):
            continue
        lines, index = _Lines(doc.text), _SymbolIndex(doc.text, lang)
        for m in index.methods:
            if lines.size(m.start, m.end) <= config.MAX_CHUNK_CHARS:
                assert any(c.start_line - 1 <= m.start and m.end <= c.end_line - 1 for c in chunks), \
                    f"{doc.metadata['path']} : méthode {m.name} coupée alors qu'elle tient dans la limite"


def test_corpus_split_methods_keep_their_signature(corpus):
    for doc, chunks in corpus:
        lang = doc.metadata["language"]
        if lang not in ("java", "typescript"):
            continue
        index = _SymbolIndex(doc.text, lang)
        for c in chunks:
            m = index.enclosing_method(c.start_line - 1)
            if m is not None:
                assert "Suite de:" in c.header and m.name in c.header, \
                    f"{c.id} commence au milieu de {m.name} sans sa signature"


def test_corpus_no_chunk_above_token_ceiling(corpus):
    for _, chunks in corpus:
        for c in chunks:
            assert est_tokens(c.embed_text) <= config.MAX_EMBED_TOKENS, c.id


def test_corpus_small_chunks_are_unmergeable(corpus):
    for _, chunks in corpus:
        for i, c in enumerate(chunks):
            if c.n_nws >= config.MIN_CHUNK_CHARS or len(chunks) == 1:
                continue
            nxt = chunks[i + 1] if i + 1 < len(chunks) else None
            prev = chunks[i - 1] if i > 0 else None
            fits_next = nxt is not None and nws(c.body) + nxt.n_nws <= config.MAX_CHUNK_CHARS
            fits_prev = prev is not None and nws(c.body) + prev.n_nws <= config.MAX_CHUNK_CHARS
            assert not (fits_next or fits_prev), f"{c.id} est petit et aurait pu être fusionné"


def test_corpus_ids_are_unique(corpus):
    ids = [c.id for _, chunks in corpus for c in chunks]
    assert len(ids) == len(set(ids))