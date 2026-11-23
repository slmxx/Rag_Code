"""Juge RAGAS : le LLM (et les embeddings) qui notent les réponses de Mistral 7B.
le LLM juge: Gemini
"""
from ragas import evaluate, EvaluationDataset
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
)

from rag_pipeline import config


def _build_judge():
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    api_key = config.get_key("GEMINI_API_KEY")
    llm = ChatGoogleGenerativeAI(model=config.JUDGE_MODEL, google_api_key=api_key, temperature=0)
    embeddings = GoogleGenerativeAIEmbeddings(model=config.JUDGE_EMBED_MODEL, google_api_key=api_key)
    return LangchainLLMWrapper(llm), LangchainEmbeddingsWrapper(embeddings)


def judge(rows: list[dict], max_workers: int = 1, timeout: int = 600):
    """rows: [{user_input, retrieved_contexts, response, reference}, ...]
    Retourne un DataFrame pandas avec les scores RAGAS par question."""
    llm, embeddings = _build_judge()
    dataset = EvaluationDataset.from_list(rows)

    result = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(),
            ResponseRelevancy(),
            LLMContextPrecisionWithReference(),
            LLMContextRecall(),
        ],
        llm=llm,
        embeddings=embeddings,
        run_config=RunConfig(max_workers=max_workers, timeout=timeout),
    )
    return result.to_pandas()
