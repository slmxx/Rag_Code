"""Interface Streamlit du RAG e-commerce.

Lancer:
    pip install streamlit
    streamlit run app.py

- La recherche utilise le retriever (embedder de config.ACTIVE_EMBEDDER, sans reranker).
- La génération utilise Mistral via Ollama. 
- Le modèle et l'index sont chargés une seule fois (st.cache_resource) : seule la 1re question est plus lente.
"""
import time
from pathlib import Path

import streamlit as st

from rag_pipeline import config

st.set_page_config(page_title="Assistant RAG e-commerce", page_icon="🟧", layout="wide")

st.markdown(
    """
<style>
.stripe {height: 6px; border-radius: 3px; margin: 0 0 1rem 0;
         background: linear-gradient(90deg, #FFD800 0%, #FFAF00 25%, #FF8205 50%, #FA520F 75%, #E10500 100%);}
.title {font-size: 2rem; font-weight: 700; color: #1E1E1E; margin: 0;}
.title span {color: #FA520F;}
.subtitle {color: #6B5B4B; margin: .2rem 0 1rem 0;}
.src {background: #FFFFFF; border: 1px solid #F3DDBA; border-left: 5px solid #FA520F;
      border-radius: 8px; padding: .55rem .9rem; margin: .45rem 0;}
.src .path {font-family: monospace; font-weight: 600; color: #1E1E1E;}
.src .meta {color: #8A7560; font-size: .85rem;}
.pill {display: inline-block; background: #FFF0D4; color: #B33A00; border-radius: 999px;
       padding: .05rem .6rem; font-size: .8rem; margin-right: .4rem;}
</style>
<div class="stripe"></div>
""",
    unsafe_allow_html=True,
)

LANG = {".java": "java", ".ts": "typescript", ".html": "html", ".yml": "yaml", ".yaml": "yaml",
        ".xml": "xml", ".sql": "sql", ".md": "markdown", ".properties": "properties",
        ".sh": "bash", ".py": "python", ".graphqls": "graphql"}
EXAMPLES = [
    "Quel service utilise Redis et pourquoi ?",
    "Comment lancer le projet avec Docker ?",
    "Comment l'authentification est-elle configurée ?",
    "Comment fonctionnent les tentatives de renvoi des notifications ?",
]


@st.cache_resource(show_spinner="Chargement de l'index et du modèle de recherche...")
def load_retriever():
    from rag_pipeline.retrieval import retriever
    try:
        retriever.retrieve("warm-up")          # chauffe : charge le client et la collection
    except Exception:                          # noqa: BLE001 (quota, réseau) : on réessaiera à la 1re question
        pass
    return retriever


def search(question: str, k: int):
    retriever = load_retriever()
    config.TOP_K_FINAL = k
    t0 = time.perf_counter()
    nodes = retriever.retrieve(question)
    elapsed = time.perf_counter() - t0
    sources = [{
        "path": n.node.metadata.get("path", ""),
        "start": n.node.metadata.get("start_line"),
        "end": n.node.metadata.get("end_line"),
        "context": n.node.metadata.get("context", ""),
        "score": float(n.score),
        "text": n.node.get_content(),
    } for n in nodes]
    return sources, elapsed


def stream_answer(question: str, contexts: list[str]):
    from rag_pipeline.generation import answer
    from rag_pipeline.generation.prompts import build_prompt
    for part in answer._llm.stream_complete(build_prompt(question, contexts)):
        yield part.delta or ""


def show_sources(sources: list[dict], full: bool) -> None:
    with st.expander(f"Sources envoyées au modèle ({len(sources)} extraits)", expanded=full):
        for i, s in enumerate(sources, 1):
            lines = f"l.{s['start']}-{s['end']}" if s["start"] is not None else ""
            st.markdown(
                f"<div class='src'><span class='pill'>#{i}</span><span class='path'>{s['path']}</span>"
                f"<div class='meta'>{lines} · similarité {s['score']:.3f}"
                f"{' · ' + s['context'] if s['context'] else ''}</div></div>",
                unsafe_allow_html=True)
            header, sep, body = s["text"].partition("\n---\n")
            code = body if sep else s["text"]
            with st.expander("Voir le texte de l'extrait"):
                st.code(code.strip(), language=LANG.get(Path(s["path"]).suffix, "text"))


def show_timing(t_search: float, t_gen: float | None) -> None:
    txt = f"🔎 recherche {t_search:.2f} s"
    if t_gen is not None:
        txt += f" · 🤖 génération {t_gen:.1f} s"
    st.caption(txt)


# ---------------------------------------------------------------- barre latérale
with st.sidebar:
    st.markdown("### Réglages")
    use_llm = st.toggle("Générer avec Mistral", value=True)
    k = st.slider("Extraits envoyés au modèle", 1, 8, config.TOP_K_FINAL)
    show_full = st.checkbox("Ouvrir les sources automatiquement", value=False)
    st.markdown("---")
    st.markdown(f"**Embeddings** : `{config.ACTIVE_EMBEDDER}`  \n"
                f"**Génération** : `{config.GENERATION_MODEL}` (Ollama)  \n"
                f"**Reranker** : {'oui' if config.USE_RERANKER else 'non'}")
    st.markdown("---")
    st.markdown("**Essayer :**")
    for ex in EXAMPLES:
        if st.button(ex, use_container_width=True):
            st.session_state["pending"] = ex
    if st.button("Effacer la conversation", use_container_width=True):
        st.session_state["messages"] = []

# ---------------------------------------------------------------- en-tête
st.markdown("<p class='title'>Assistant <span>RAG</span> e-commerce</p>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Pose une question sur le code des microservices Spring et du frontend Angular. "
            "Les réponses s'appuient uniquement sur les extraits retrouvés.</p>", unsafe_allow_html=True)

st.session_state.setdefault("messages", [])
for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            show_sources(m["sources"], show_full)
            show_timing(m["t_search"], m["t_gen"])

question = st.chat_input("Pose ta question sur le code...") or st.session_state.pop("pending", None)
if question:
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Recherche des extraits pertinents..."):
                sources, t_search = search(question, k)
        except Exception as exc:                                     # noqa: BLE001
            st.error(f"La recherche a échoué ({exc.__class__.__name__}) : {exc}\n\n"
                     "Si c'est une erreur de limite de débit Voyage (3 requêtes/minute en gratuit), "
                     "attends une minute et réessaie.")
            st.stop()

        answer_text, t_gen = "", None
        if use_llm:
            try:
                t0 = time.perf_counter()
                answer_text = st.write_stream(stream_answer(question, [s["text"] for s in sources]))
                t_gen = time.perf_counter() - t0
            except Exception as exc:                                 # noqa: BLE001
                answer_text = ("La génération a échoué : Ollama est-il lancé et le modèle `"
                               f"{config.GENERATION_MODEL}` installé ? ({exc.__class__.__name__})")
                st.warning(answer_text)
        else:
            answer_text = "*Mode recherche seule : voici les extraits les plus proches de ta question.*"
            st.markdown(answer_text)
        show_sources(sources, show_full or not use_llm)
        show_timing(t_search, t_gen)

    st.session_state["messages"].append({"role": "assistant", "content": answer_text,
                                         "sources": sources, "t_search": t_search, "t_gen": t_gen})
