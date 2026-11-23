from llama_index.llms.ollama import Ollama

from rag_pipeline import config
from rag_pipeline.generation.prompts import build_prompt

_llm = Ollama(
    model=config.GENERATION_MODEL,
    base_url=config.OLLAMA_BASE_URL,
    temperature=config.GENERATION_TEMPERATURE,
    request_timeout=config.GENERATION_TIMEOUT,
)


def generate_answer(question: str, contexts: list[str]) -> str:
    response = _llm.complete(build_prompt(question, contexts))
    return response.text.strip()

if __name__ == "__main__":
    import sys

    from rag_pipeline.retrieval.retriever import retrieve

    question = " ".join(sys.argv[1:]) or input("Question : ")
    nodes = retrieve(question)
    contexts = [n.node.get_content() for n in nodes]

    print("\n=== CHUNKS RÉCUPÉRÉS ===")
    for n in nodes:
        m = n.node.metadata
        print(f"- {m['path']} (lignes {m['start_line']}-{m['end_line']})")

    print("\n=== RÉPONSE MISTRAL 7B ===")
    print(generate_answer(question, contexts))