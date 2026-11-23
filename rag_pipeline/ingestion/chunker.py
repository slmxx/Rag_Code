"""Deuxième étape de l'indexation : découper chaque fichier en chunks qui portent un sens.

- Java / TypeScript : algorithme cAST (arbre syntaxique tree-sitter), réécrit ici + notre propre contexte
  (classe, méthode, suite). Réf. : Zhang et al., « cAST », arXiv 2506.15655.
- yaml, xml, html, md, sql, graphql, conf, properties, sh : découpage selon la structure du format.
- Tous les formats : un morceau trop petit est fusionné avec le suivant (même fichier uniquement),
  un morceau trop grand est redécoupé, et chaque chunk reçoit un en-tête.

Unités : MAX_CHUNK_CHARS et MIN_CHUNK_CHARS comptent des caractères NON BLANCS (comme l'article cAST).
"""
from __future__ import annotations

import hashlib
import logging
import math
import re
import statistics
import sys
from itertools import accumulate
from collections import Counter
from dataclasses import dataclass
from html.parser import HTMLParser
from xml.parsers import expat

import tree_sitter as ts
import tree_sitter_java
import tree_sitter_typescript
from llama_index.core.schema import Document

from rag_pipeline import config

log = logging.getLogger(__name__)


def _max() -> int:
    return config.MAX_CHUNK_CHARS


def _min() -> int:
    return config.MIN_CHUNK_CHARS


def nws(text: str) -> int:
    """Nombre de caractères non blancs."""
    return sum(1 for ch in text if not ch.isspace())


def est_tokens(text: str) -> int:
    """Estimation grossière : à calibrer avec CHARS_PER_TOKEN dans config.py."""
    return math.ceil(len(text) / getattr(config, "CHARS_PER_TOKEN", 3.0))


# ======================================================================
#  Structures de données
# ======================================================================
@dataclass
class Piece:
    """Un morceau de fichier, en numéros de ligne (base 0, bornes incluses)."""
    start: int
    end: int
    context: str = ""
    part: tuple[int, int] | None = None     # (k, n) quand un bloc trop grand a été redécoupé


@dataclass(frozen=True)
class Chunk:
    id: str
    path: str
    service: str
    language: str
    kind: str
    is_test: bool
    context: str
    symbols: str
    start_line: int          # base 1, comme GitHub et l'éditeur
    end_line: int
    part: int
    n_parts: int
    header: str
    body: str
    file_sha256: str

    @property
    def embed_text(self) -> str:
        """Le texte réellement envoyé au modèle d'embedding : en-tête + code."""
        return f"{self.header}\n---\n{self.body}"

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.embed_text.encode("utf-8")).hexdigest()

    @property
    def n_nws(self) -> int:
        return nws(self.body)

    def to_metadata(self) -> dict:
        """Uniquement des valeurs simples (texte, nombre, booléen) : compatible ChromaDB."""
        return {
            "path": self.path, "service": self.service, "language": self.language,
            "kind": self.kind, "is_test": self.is_test, "context": self.context,
            "symbols": self.symbols, "start_line": self.start_line, "end_line": self.end_line,
            "part": self.part, "n_parts": self.n_parts, "n_nws": self.n_nws,
            "sha256": self.sha256, "file_sha256": self.file_sha256,
        }


class _Lines:
    """Les lignes d'un fichier, avec des tailles calculées une fois pour toutes."""

    def __init__(self, text: str):
        self.lines = text.split("\n")
        self.cum = [0]
        for line in self.lines:
            self.cum.append(self.cum[-1] + nws(line))

    def __len__(self) -> int:
        return len(self.lines)

    def size(self, s: int, e: int) -> int:
        return self.cum[e + 1] - self.cum[s]

    def text(self, s: int, e: int) -> str:
        return "\n".join(self.lines[s:e + 1])

    def trim(self, s: int, e: int) -> tuple[int, int] | None:
        while s <= e and not self.lines[s].strip():
            s += 1
        while e >= s and not self.lines[e].strip():
            e -= 1
        return (s, e) if s <= e else None


def _ctx(path: list[str]) -> str:
    return " > ".join(p for p in path if p and p != "---") or "(racine)"


def _cap(text: str, n: int = 160) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"


def _attach_comments(L: _Lines, idx: int, floor: int, prefixes: tuple[str, ...]) -> int:
    """Un commentaire collé juste au-dessus d'un bloc appartient à ce bloc."""
    while idx - 1 >= floor and L.lines[idx - 1].strip().startswith(prefixes):
        idx -= 1
    return idx


def _segments(L: _Lines, bounds: list[tuple[int, str]]) -> list[Piece]:
    """Transforme une liste de débuts de blocs en morceaux contigus qui couvrent tout le fichier."""
    uniq: list[tuple[int, str]] = []
    for line, ctx in sorted(bounds, key=lambda b: b[0]):
        if uniq and uniq[-1][0] == line:
            continue
        uniq.append((line, ctx))
    if not uniq:
        uniq = [(0, "(racine)")]
    uniq[0] = (0, uniq[0][1])               # ce qui précède le premier bloc reste dans le premier
    pieces = []
    for i, (line, ctx) in enumerate(uniq):
        end = uniq[i + 1][0] - 1 if i + 1 < len(uniq) else len(L) - 1
        seg = L.trim(line, end)
        if seg:
            pieces.append(Piece(seg[0], seg[1], ctx))
    return pieces


# ======================================================================
#  Java / TypeScript : algorithme cAST + contexte
# ======================================================================
_CLASS_TYPES = {
    "java": {"class_declaration", "interface_declaration", "enum_declaration", "record_declaration"},
    "typescript": {"class_declaration", "abstract_class_declaration", "interface_declaration",
                   "enum_declaration"},
}
_METHOD_TYPES = {
    "java": {"method_declaration", "constructor_declaration"},
    "typescript": {"method_definition", "function_declaration"},
}
_PARSERS: dict[str, ts.Parser] = {}


def _parser(language: str) -> ts.Parser:
    if language not in _PARSERS:
        lang = (tree_sitter_java.language() if language == "java"
                else tree_sitter_typescript.language_typescript())
        _PARSERS[language] = ts.Parser(ts.Language(lang))
    return _PARSERS[language]


@dataclass
class _Sym:
    name: str
    start: int
    end: int
    sig: str = ""


class _SymbolIndex:
    """Les classes et méthodes d'un fichier, avec leurs lignes (via tree-sitter)."""

    def __init__(self, text: str, language: str):
        data = text.encode("utf-8")
        tree = _parser(language).parse(data)
        self.classes: list[_Sym] = []
        self.methods: list[_Sym] = []
        stack = [tree.root_node]
        while stack:
            node = stack.pop()
            stack.extend(node.children)
            is_class = node.type in _CLASS_TYPES[language]
            is_method = node.type in _METHOD_TYPES[language]
            if not (is_class or is_method):
                continue
            name_node = node.child_by_field_name("name")
            if name_node is None:
                continue
            name = data[name_node.start_byte:name_node.end_byte].decode("utf-8", "ignore")
            sym = _Sym(name, node.start_point[0], node.end_point[0])
            if is_method:
                body = node.child_by_field_name("body")
                stop = body.start_byte if body is not None else node.end_byte
                sig = " ".join(data[node.start_byte:stop].decode("utf-8", "ignore").split())
                sym.sig = _cap(sig)
                self.methods.append(sym)
            else:
                self.classes.append(sym)
        self.classes.sort(key=lambda s: s.start)
        self.methods.sort(key=lambda s: s.start)

    def chain(self, row: int) -> str:
        return ".".join(c.name for c in self.classes if c.start <= row <= c.end)

    def chain_for(self, s: int, e: int) -> str:
        """Classe englobante au début du morceau. Si le morceau commence avant toute classe
        (package, imports), on prend la classe du morceau seulement s'il n'y en a qu'une."""
        chain = self.chain(s)
        if chain:
            return chain
        tops = [c for c in self.classes
                if s <= c.start <= e and not any(o is not c and o.start <= c.start and c.end <= o.end
                                                 for o in self.classes)]
        return tops[0].name if len(tops) == 1 else ""

    def split_start(self, s: int, e: int) -> _Sym | None:
        """Première méthode qui commence dans ce morceau mais se termine dans un autre."""
        found = [m for m in self.methods if s <= m.start <= e and m.end > e]
        return min(found, key=lambda m: m.start) if found else None

    def symbols(self, s: int, e: int) -> str:
        names: list[str] = []
        for sym in sorted(self.classes + self.methods, key=lambda x: x.start):
            if s <= sym.start <= e and sym.name not in names:
                names.append(sym.name)
        return ", ".join(names[:12]) + (", …" if len(names) > 12 else "")

    def enclosing_method(self, row: int) -> _Sym | None:
        """La méthode la plus interne qui a COMMENCÉ avant cette ligne et la contient."""
        found = [m for m in self.methods if m.start < row <= m.end]
        return min(found, key=lambda m: m.end - m.start) if found else None


# ---- Algorithme cAST (découpage par arbre syntaxique) -------------------
# Principe : on remplit un morceau avec des nœuds ENTIERS, dans l'ordre, jusqu'à MAX_CHUNK_CHARS
# caractères non blancs. Un nœud qui dépasse la limite à lui seul est ouvert : on traite ses enfants
# (les instructions d'une méthode), et on regroupe les petits morceaux voisins qui en sortent.
_ASCII_SPACE = {b for b in range(128) if chr(b).isspace()}


@dataclass
class _Node:
    """Copie légère d'un nœud tree-sitter : après la copie, plus aucun appel à la bibliothèque native."""
    sb: int                  # octet de début
    eb: int                  # octet de fin
    sr: int                  # ligne de début (base 0)
    er: int                  # ligne de fin (base 0)
    kids: list


def _snapshot(root) -> _Node:
    top = _Node(root.start_byte, root.end_byte, root.start_point[0], root.end_point[0], [])
    work = [(root, top)]
    while work:
        node, rec = work.pop()
        for child in node.children:
            sub = _Node(child.start_byte, child.end_byte, child.start_point[0], child.end_point[0], [])
            rec.kids.append(sub)
            work.append((child, sub))
    return top


def _nws_prefix(text: str) -> list[int]:
    """prefix[i] = nombre de caractères non blancs situés avant l'octet i du texte encodé en UTF-8."""
    if text.isascii():
        flags = [0 if b in _ASCII_SPACE else 1 for b in text.encode("ascii")]
    else:
        flags = []
        for ch in text:
            flags.append(0 if ch.isspace() else 1)
            flags.extend([0] * (len(ch.encode("utf-8")) - 1))
    return [0, *accumulate(flags)]


def _assign(nodes: list[_Node], size, limit: int) -> list[list[_Node]]:
    windows: list[list[_Node]] = []
    cur: list[_Node] = []
    cur_size = 0
    for node in nodes:
        s = size(node)
        too_big = s > limit
        if (not cur and too_big) or cur_size + s > limit:
            if cur:
                windows.append(cur)
                cur, cur_size = [], 0
            if too_big:
                inner = _assign(node.kids, size, limit)
                if inner:
                    windows.extend(_merge_windows(inner, size, limit))
                else:                                  # feuille énorme sans enfants : on la garde entière
                    windows.append([node])
            else:
                cur.append(node)
                cur_size += s
        else:
            cur.append(node)
            cur_size += s
    if cur:
        windows.append(cur)
    return windows


def _merge_windows(windows: list[list[_Node]], size, limit: int) -> list[list[_Node]]:
    """Regroupe les morceaux voisins issus d'un même nœud tant que le total reste sous la limite."""
    merged = [list(windows[0])]
    for w in windows[1:]:
        if sum(size(n) for n in merged[-1]) + sum(size(n) for n in w) <= limit:
            merged[-1].extend(w)
        else:
            merged.append(list(w))
    return merged


def _ast_pieces(text: str, language: str) -> list[Piece]:
    data = text.encode("utf-8")
    tree = _parser(language).parse(data)
    root = _snapshot(tree.root_node)
    del tree
    prefix = _nws_prefix(text)
    size = lambda n: prefix[n.eb] - prefix[n.sb]
    limit = _max()
    windows = [[root]] if size(root) <= limit else _assign(root.kids, size, limit)
    return [Piece(w[0].sr, w[-1].er) for w in windows]


# ======================================================================
#  Autres formats : découpage selon la structure
# ======================================================================
# ---- YAML (.yml, .yaml) -------------------------------------------------
_YAML_KEY = re.compile(r"^(?P<indent> *)(?P<key>\"[^\"]+\"|'[^']+'|[^\s#\-\[\]{}&*!|>'\"%@`,?][^:#]*?)\s*:(?:\s|$)")
_YAML_BLOCK = re.compile(r":\s*[|>][+-]?\d*\s*(?:#.*)?$")


def _yaml_keys(L: _Lines, s: int, e: int) -> list[tuple[int, int, str]]:
    keys, skip_above = [], None
    for i in range(s, e + 1):
        line = L.lines[i]
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        if skip_above is not None:                 # contenu d'un bloc texte (| ou >)
            if indent > skip_above:
                continue
            skip_above = None
        if line.lstrip().startswith("#"):
            continue
        if line.strip() == "---":
            keys.append((i, 0, "---"))
            continue
        m = _YAML_KEY.match(line)
        if m:
            keys.append((i, indent, m.group("key").strip().strip("\"'")))
            if _YAML_BLOCK.search(line):
                skip_above = indent
    return keys


def _split_yaml(L: _Lines, s: int, e: int, path: list[str]) -> list[Piece]:
    keys = _yaml_keys(L, s, e)
    if not keys:
        seg = L.trim(s, e)
        return [Piece(seg[0], seg[1], _ctx(path))] if seg else []
    base = min(k[1] for k in keys)
    tops = [k for k in keys if k[1] == base]
    bounds, prev = [], s - 1
    for line, _, key in tops:
        bounds.append((_attach_comments(L, line, prev + 1, ("#",)), key, line))
        prev = line
    bounds[0] = (s, bounds[0][1], bounds[0][2])
    pieces: list[Piece] = []
    for i, (b, key, line) in enumerate(bounds):
        end = bounds[i + 1][0] - 1 if i + 1 < len(bounds) else e
        seg = L.trim(b, end)
        if not seg:
            continue
        a, z = seg
        sub = path + [key]
        if L.size(a, z) > _max():
            children = _split_yaml(L, line + 1, z, sub)
            if len(children) > 1:
                children[0].start = a               # garde la clé parent dans le premier enfant
                pieces.extend(children)
                continue
        pieces.append(Piece(a, z, _ctx(sub)))
    return pieces


# ---- Markdown (.md) -----------------------------------------------------
_MD_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")


def _split_md(L: _Lines) -> list[Piece]:
    bounds, stack, fence = [], [], None
    for i, line in enumerate(L.lines):
        st = line.lstrip()
        if st.startswith("```") or st.startswith("~~~"):
            mark = st[:3]
            if fence is None:
                fence = mark
            elif mark == fence:
                fence = None
            continue
        if fence:
            continue
        m = _MD_HEADING.match(line)
        if m:
            level, title = len(m.group(1)), m.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            bounds.append((i, " > ".join(t for _, t in stack)))
    if not bounds or bounds[0][0] > 0:
        bounds.insert(0, (0, "(introduction)"))
    return _segments(L, bounds)


# ---- XML (pom.xml, ...) -------------------------------------------------
_XML_NAMED = {"dependency", "plugin", "parent", "module", "profile"}


def _xml_elements(text: str) -> list[tuple[int, str, int, int]]:
    elems, stack = [], []
    parser = expat.ParserCreate()

    def start(name, attrs):
        stack.append((len(stack), name, parser.CurrentLineNumber - 1))

    def end(name):
        depth, tag, s = stack.pop()
        elems.append((depth, tag, s, parser.CurrentLineNumber - 1))

    parser.StartElementHandler, parser.EndElementHandler = start, end
    parser.Parse(text.encode("utf-8"), True)
    return elems


def _split_xml(L: _Lines, elems, s: int, e: int, depth: int, path: list[str]) -> list[Piece]:
    kids = sorted((el for el in elems if el[0] == depth and el[2] >= s and el[3] <= e), key=lambda x: x[2])
    if not kids:
        seg = L.trim(s, e)
        return [Piece(seg[0], seg[1], _ctx(path))] if seg else []
    bounds, prev = [], s - 1
    for _, tag, st, en in kids:
        if bounds and st == bounds[-1][2]:
            continue
        bounds.append((_attach_comments(L, st, prev + 1, ("<!--",)), tag, st, en))
        prev = st
    bounds[0] = (s,) + bounds[0][1:]
    pieces: list[Piece] = []
    for i, (b, tag, st, en) in enumerate(bounds):
        end = bounds[i + 1][0] - 1 if i + 1 < len(bounds) else e
        seg = L.trim(b, end)
        if not seg:
            continue
        a, z = seg
        label = tag
        if tag in _XML_NAMED:
            m = re.search(r"<artifactId>([^<]+)</artifactId>", L.text(st, min(en, st + 10)))
            if m:
                label = f"{tag}[{m.group(1).strip()}]"
        sub = path + [label]
        if L.size(a, z) > _max():
            children = _split_xml(L, elems, st, en, depth + 1, sub)
            if len(children) > 1:
                children[0].start = a
                children[-1].end = z
                pieces.extend(children)
                continue
        pieces.append(Piece(a, z, _ctx(sub)))
    return pieces


# ---- HTML (templates Angular) ------------------------------------------
class _HtmlScan(HTMLParser):
    VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta",
            "param", "source", "track", "wbr"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.depth = 0
        self.tags: list[tuple[int, str, int]] = []

    @staticmethod
    def _label(tag, attrs) -> str:
        d = dict(attrs)
        if d.get("id"):
            return f"{tag}#{d['id']}"
        cls = (d.get("class") or "").split()
        return f"{tag}.{cls[0]}" if cls else tag

    def handle_starttag(self, tag, attrs):
        self.tags.append((self.depth, self._label(tag, attrs), self.getpos()[0] - 1))
        if tag not in self.VOID:
            self.depth += 1

    def handle_startendtag(self, tag, attrs):
        self.tags.append((self.depth, self._label(tag, attrs), self.getpos()[0] - 1))

    def handle_endtag(self, tag):
        if tag not in self.VOID:
            self.depth = max(0, self.depth - 1)


def _split_html(L: _Lines, tags, s: int, e: int, depth: int, path: list[str]) -> list[Piece]:
    cands = [t for t in tags if t[0] == depth and s <= t[2] <= e]
    if not cands:
        seg = L.trim(s, e)
        return [Piece(seg[0], seg[1], _ctx(path))] if seg else []
    bounds, prev = [], s - 1
    for _, label, line in sorted(cands, key=lambda t: t[2]):
        if bounds and bounds[-1][2] == line:
            continue
        bounds.append((_attach_comments(L, line, prev + 1, ("<!--",)), label, line))
        prev = line
    bounds[0] = (s,) + bounds[0][1:]
    pieces: list[Piece] = []
    for i, (b, label, line) in enumerate(bounds):
        end = bounds[i + 1][0] - 1 if i + 1 < len(bounds) else e
        seg = L.trim(b, end)
        if not seg:
            continue
        a, z = seg
        sub = path + [label]
        if L.size(a, z) > _max():
            children = _split_html(L, tags, line + 1, z, depth + 1, sub)
            if len(children) > 1:
                children[0].start = a
                pieces.extend(children)
                continue
        pieces.append(Piece(a, z, _ctx(sub)))
    return pieces


# ---- SQL ----------------------------------------------------------------
def _sql_end_lines(text: str) -> list[int]:
    """Lignes où une instruction se termine (un ';' hors texte, commentaire ou $$)."""
    ends, i, line, n = [], 0, 0, len(text)
    in_s = in_d = in_lc = in_bc = in_dollar = False
    while i < n:
        c, two = text[i], text[i:i + 2]
        if c == "\n":
            line, in_lc, i = line + 1, False, i + 1
            continue
        if in_lc:
            i += 1
        elif in_bc:
            if two == "*/":
                in_bc, i = False, i + 2
            else:
                i += 1
        elif in_s:
            if c == "'":
                if text[i + 1:i + 2] == "'":
                    i += 1
                else:
                    in_s = False
            i += 1
        elif in_d:
            in_d = c != '"'
            i += 1
        elif in_dollar:
            if two == "$$":
                in_dollar, i = False, i + 2
            else:
                i += 1
        elif two == "--":
            in_lc, i = True, i + 2
        elif two == "/*":
            in_bc, i = True, i + 2
        elif two == "$$":
            in_dollar, i = True, i + 2
        else:
            if c == "'":
                in_s = True
            elif c == '"':
                in_d = True
            elif c == ";":
                ends.append(line)
            i += 1
    return ends


def _split_sql(L: _Lines, text: str) -> list[Piece]:
    ends = sorted(set(_sql_end_lines(text)))
    starts = [0] + [e + 1 for e in ends if e + 1 < len(L)]
    pieces = _segments(L, [(s, "") for s in starts])
    for p in pieces:
        code = "\n".join(ln for ln in L.lines[p.start:p.end + 1] if not ln.strip().startswith("--"))
        words = [w for w in re.findall(r"[A-Za-z_][\w.]*|`[^`]+`|\"[^\"]+\"", code)
                 if w.upper() not in {"IF", "NOT", "EXISTS"}]
        p.context = " ".join(words[:3]) or "(sql)"
    return pieces


# ---- Blocs entre accolades (GraphQL, nginx .conf) ----------------------
_STR = re.compile(r"\"[^\"]*\"|'[^']*'")


def _brace_delta(line: str) -> int:
    code = _STR.sub("", line).split("#", 1)[0]
    return code.count("{") - code.count("}")


def _brace_units(L: _Lines, s: int, e: int, base: int) -> list[tuple[int, int, int]]:
    """Blocs de plus haut niveau : (début avec ses commentaires, fin, ligne d'ouverture)."""
    units, depth, i, pending = [], base, s, None
    while i <= e:
        st = L.lines[i].strip()
        if depth == base:
            if not st:
                pending = None
                i += 1
                continue
            if st.startswith("#"):
                pending = i if pending is None else pending
                i += 1
                continue
            if st.startswith('"""'):               # description GraphQL sur plusieurs lignes
                j = i
                if st.count('"""') < 2:
                    j += 1
                    while j <= e and '"""' not in L.lines[j]:
                        j += 1
                pending = i if pending is None else pending
                i = j + 1
                continue
            if st.startswith('"'):
                pending = i if pending is None else pending
                i += 1
                continue
        start = pending if pending is not None else i
        pending, opener = None, i
        depth += _brace_delta(L.lines[i])
        j = i
        while depth > base and j < e:
            j += 1
            depth += _brace_delta(L.lines[j])
        depth = base                                 # bloc fini (ou fichier fini) : on repart du niveau de base
        units.append((start, j, opener))
        i = j + 1
    return units


def _unit_label(line: str) -> str:
    head = re.split(r"[{(;]", line.strip(), maxsplit=1)[0].strip()
    return _cap(head, 60) or "(bloc)"


def _split_braces(L: _Lines, s: int, e: int, base: int, path: list[str]) -> list[Piece]:
    units = _brace_units(L, s, e, base)
    if not units:
        seg = L.trim(s, e)
        return [Piece(seg[0], seg[1], _ctx(path))] if seg else []
    pieces: list[Piece] = []
    for i, (a, z, opener) in enumerate(units):
        seg_start = s if i == 0 else a
        seg_end = units[i + 1][0] - 1 if i + 1 < len(units) else e
        seg = L.trim(seg_start, seg_end)
        if not seg:
            continue
        sub = path + [_unit_label(L.lines[opener])]
        if L.size(*seg) > _max() and z > opener:
            children = _split_braces(L, opener + 1, z - 1, base + 1, sub)
            if len(children) > 1:
                children[0].start = seg[0]
                children[-1].end = seg[1]
                pieces.extend(children)
                continue
        pieces.append(Piece(seg[0], seg[1], _ctx(sub)))
    return pieces


# ---- Blocs séparés par des lignes vides (.properties, .sh, repli) -------
def _split_blocks(L: _Lines) -> list[Piece]:
    bounds, i, n = [], 0, len(L)
    while i < n:
        if not L.lines[i].strip():
            i += 1
            continue
        j = i
        while j + 1 < n and L.lines[j + 1].strip():
            j += 1
        first = next((ln.strip() for ln in L.lines[i:j + 1] if not ln.strip().startswith("#")), "")
        bounds.append((i, _cap(first, 60) or "(commentaire)"))
        i = j + 1
    return _segments(L, bounds)


# ======================================================================
#  Étapes communes : taille maximale, fusion des petits morceaux
# ======================================================================
def _line_groups(L: _Lines, a: int, b: int) -> list[tuple[int, int]]:
    groups, cur = [], [a, a]
    for i in range(a + 1, b + 1):
        if L.size(cur[0], i) <= _max():
            cur[1] = i
        else:
            groups.append((cur[0], cur[1]))
            cur = [i, i]
    groups.append((cur[0], cur[1]))
    return groups


def _enforce_max(L: _Lines, pieces: list[Piece]) -> list[Piece]:
    """Un morceau trop grand est coupé aux lignes vides, sinon ligne par ligne (jamais au milieu d'une ligne)."""
    out: list[Piece] = []
    for p in pieces:
        if L.size(p.start, p.end) <= _max():
            out.append(p)
            continue
        units: list[tuple[int, int]] = []
        i = p.start
        while i <= p.end:
            if not L.lines[i].strip():
                i += 1
                continue
            j = i
            while j + 1 <= p.end and L.lines[j + 1].strip():
                j += 1
            units.extend([(i, j)] if L.size(i, j) <= _max() else _line_groups(L, i, j))
            i = j + 1
        packed: list[list[int]] = []
        for a, b in units:
            if packed and L.size(packed[-1][0], b) <= _max():
                packed[-1][1] = b
            else:
                packed.append([a, b])
        packed[0][0], packed[-1][1] = p.start, p.end
        for k in range(1, len(packed)):             # les lignes vides entre unités restent dans le morceau précédent
            packed[k - 1][1] = packed[k][0] - 1
        n = len(packed)
        out.extend(Piece(a, b, p.context, (k + 1, n)) for k, (a, b) in enumerate(packed))
    return out


def _join_ctx(a: str, b: str) -> str:
    parts = [x for x in (a + " | " + b).split(" | ") if x]
    uniq = list(dict.fromkeys(parts))
    return _cap(" | ".join(uniq), 200)


def _join(a: Piece, b: Piece) -> Piece:
    return Piece(a.start, b.end, _join_ctx(a.context, b.context), a.part)


def _merge_small(L: _Lines, pieces: list[Piece]) -> list[Piece]:
    """Un morceau sous MIN_CHUNK_CHARS absorbe le suivant (même fichier), sans dépasser MAX_CHUNK_CHARS.
    S'il n'a pas de suivant utilisable, il est rattaché au précédent."""
    out: list[Piece] = []
    i = 0
    while i < len(pieces):
        cur = pieces[i]
        while (L.size(cur.start, cur.end) < _min() and i + 1 < len(pieces)
               and L.size(cur.start, pieces[i + 1].end) <= _max()):
            cur = _join(cur, pieces[i + 1])
            i += 1
        if (L.size(cur.start, cur.end) < _min() and out
                and L.size(out[-1].start, cur.end) <= _max()):
            out[-1] = _join(out[-1], cur)
        else:
            out.append(cur)
        i += 1
    return out


# ======================================================================
#  Assemblage : fichier -> chunks
# ======================================================================
def _header(path: str, service: str, context: str, symbols: str, suite: str, debut: str,
            part: tuple[int, int] | None) -> str:
    lines = [f"Fichier: {path}", f"Service: {service}", f"Contexte: {context}"]
    if symbols:
        lines.append(f"Symboles: {symbols}")
    if debut:
        lines.append(f"Début de: {debut}")
    if suite:
        lines.append(f"Suite de: {suite}")
    elif part and not debut:
        lines.append(f"Partie: {part[0]}/{part[1]}")
    return "\n".join(lines)


def chunk_document(doc: Document) -> list[Chunk]:
    m, text = doc.metadata, doc.text
    L = _Lines(text)
    lang = m["language"]
    index: _SymbolIndex | None = None
    try:
        if lang in ("java", "typescript"):
            pieces = _ast_pieces(text, lang)
            index = _SymbolIndex(text, lang)
        elif lang == "yaml":
            pieces = _split_yaml(L, 0, len(L) - 1, [])
        elif lang == "markdown":
            pieces = _split_md(L)
        elif lang == "xml":
            elems = _xml_elements(text)
            root = next((el for el in elems if el[0] == 0), None)
            pieces = _split_xml(L, elems, 0, len(L) - 1, 1, [root[1]] if root else [])
        elif lang == "html":
            scan = _HtmlScan()
            scan.feed(text)
            pieces = _split_html(L, scan.tags, 0, len(L) - 1, 0, [])
        elif lang == "sql":
            pieces = _split_sql(L, text)
        elif lang in ("graphql", "conf"):
            pieces = _split_braces(L, 0, len(L) - 1, 0, [])
        else:                                        # properties, shell, tout le reste
            pieces = _split_blocks(L)
    except Exception as exc:                         # un fichier mal formé ne doit pas arrêter l'indexation
        log.warning("Découpage structurel impossible pour %s (%s) : repli sur les blocs", m["path"], exc)
        pieces, index = _split_blocks(L), None

    pieces = [p for p in pieces if L.trim(p.start, p.end)]
    if not pieces:
        return []
    # Java / TypeScript : cAST a déjà borné les morceaux (en nœuds entiers). Les re-mesurer en lignes
    # entières les recouperait au milieu d'une méthode : on ne passe donc _enforce_max que sur le reste.
    pieces = _merge_small(L, pieces if index is not None else _enforce_max(L, pieces))

    def rank_in(meth: _Sym, piece: Piece) -> tuple[int, int]:
        inter = [q for q in pieces if q.end >= meth.start and q.start <= meth.end]
        return next(k for k, q in enumerate(inter) if q is piece) + 1, len(inter)

    chunks: list[Chunk] = []
    for i, p in enumerate(pieces):
        symbols = suite = debut = ""
        part = p.part
        context = p.context or "(fichier)"
        if index is not None:
            context = index.chain_for(p.start, p.end) or "(en-tête du fichier)"
            symbols = index.symbols(p.start, p.end)
            meth = index.enclosing_method(p.start)
            if meth is not None:                      # le morceau commence au milieu d'une méthode
                k, n = rank_in(meth, p)
                suite, part = f"{meth.sig} (partie {k}/{n})", (k, n)
            first = index.split_start(p.start, p.end)
            if first is not None:                     # le morceau contient le début d'une méthode coupée
                k, n = rank_in(first, p)
                debut = f"{first.sig} (partie {k}/{n})"
                part = part or (k, n)
        body = L.text(p.start, p.end)
        header = _header(m["path"], m["service"], context, symbols, suite, debut, part)
        digest = hashlib.sha256((header + body).encode("utf-8")).hexdigest()[:8]
        chunks.append(Chunk(
            id=f"{m['path']}::{i:03d}::{digest}", path=m["path"], service=m["service"],
            language=lang, kind=m["kind"], is_test=m["is_test"], context=context, symbols=symbols,
            start_line=p.start + 1, end_line=p.end + 1,
            part=part[0] if part else 1, n_parts=part[1] if part else 1,
            header=header, body=body, file_sha256=m["sha256"],
        ))
    return chunks


def chunk_corpus(docs: list[Document]) -> list[Chunk]:
    return [c for d in docs for c in chunk_document(d)]


# ======================================================================
#  Contrôle visuel
#     python -m llamaindex_pipeline.ingestion.chunker            -> statistiques + 3 exemples (java, yaml, xml)
#     python -m llamaindex_pipeline.ingestion.chunker Service.java -> tous les chunks de ce fichier
# ======================================================================
def _stats(chunks: list[Chunk]) -> None:
    sizes = [c.n_nws for c in chunks]
    toks = [est_tokens(c.embed_text) for c in chunks]
    q = statistics.quantiles(sizes, n=20) if len(sizes) > 1 else sizes
    print(f"{len(chunks)} chunks | taille en caractères non blancs : min {min(sizes)}, "
          f"médiane {int(statistics.median(sizes))}, p95 {int(q[-1])}, max {max(sizes)}")
    print(f"tokens estimés (en-tête compris) : max {max(toks)} | plafond {config.MAX_EMBED_TOKENS}")
    over = [c for c, t in zip(chunks, toks) if t > config.MAX_EMBED_TOKENS]
    small = [c for c in chunks if c.n_nws < _min()]
    print(f"au-dessus du plafond : {len(over)} | sous le minimum ({_min()}) : {len(small)}")
    for title, key in (("langage", lambda c: c.language), ("service", lambda c: c.service)):
        print(f"\nChunks par {title} :")
        for name, n in Counter(key(c) for c in chunks).most_common():
            print(f"  {name:28} {n}")
    for c in over[:10]:
        print("AU-DESSUS DU PLAFOND :", c.id, est_tokens(c.embed_text), "tokens")


def _print_chunk(c: Chunk, k: int, n: int, max_body_lines: int | None = 12) -> None:
    print(f"\n  ┌─ chunk {k}/{n}   id = {c.id}")
    print("  │ MÉTADONNÉES (stockées dans Chroma, utilisables comme filtres) :")
    for key, value in c.to_metadata().items():
        shown = value if key not in ("sha256", "file_sha256") else str(value)[:12] + "…"
        print(f"  │     {key:12} = {shown!r}")
    print("  │")
    print("  │ ── TEXTE ENVOYÉ AU MODÈLE D'EMBEDDING ─────────────────────")
    print("  │ ┄┄ EN-TÊTE (ajouté par le chunker, n'existe pas dans le fichier) ┄┄")
    for line in c.header.split("\n"):
        print(f"  │ {line}")
    print("  │ ┄┄ CODE (copié tel quel depuis le fichier) ┄┄")
    body_lines = c.body.split("\n")
    shown_lines = body_lines if max_body_lines is None else body_lines[:max_body_lines]
    for line in shown_lines:
        print(f"  │ {line}")
    if len(shown_lines) < len(body_lines):
        print(f"  │ … ({len(body_lines) - len(shown_lines)} lignes de plus, {c.n_nws} caractères non blancs au total)")
    print("  └" + "─" * 60)


def _examples(docs: list[Document]) -> None:
    for lang, label in (("java", "JAVA"), ("yaml", "YAML (.yml / .yaml)"), ("xml", "XML")):
        scored = []
        for d in docs:
            if d.metadata["language"] != lang or d.metadata["is_test"]:
                continue
            cs = chunk_document(d)
            has_split = any("Suite de:" in c.header or "Début de:" in c.header for c in cs)
            scored.append((has_split, len(cs), d, cs))
        print("\n" + "=" * 70)
        if not scored:
            print(f"EXEMPLE {label} : aucun fichier chargé")
            continue
        _, _, doc, cs = max(scored, key=lambda t: (t[0], t[1]))
        picked = [c for c in cs if "Suite de:" in c.header or "Début de:" in c.header][:2]
        picked += [c for c in cs if c not in picked][: 3 - len(picked)]
        picked.sort(key=lambda c: c.start_line)
        print(f"EXEMPLE {label} : {doc.metadata['path']}  ({len(cs)} chunks au total, {len(picked)} affichés)")
        for k, c in enumerate(picked, 1):
            _print_chunk(c, cs.index(c) + 1, len(cs))


if __name__ == "__main__":
    from rag_pipeline.ingestion.loader import load_corpus

    logging.basicConfig(level=logging.WARNING)
    docs = load_corpus()
    if len(sys.argv) > 1:
        target = sys.argv[1].replace("\\", "/")
        for d in (d for d in docs if d.metadata["path"].endswith(target)):
            cs = chunk_document(d)
            for k, c in enumerate(cs, 1):
                _print_chunk(c, k, len(cs), max_body_lines=None)
    else:
        _stats(chunk_corpus(docs))
        _examples(docs)