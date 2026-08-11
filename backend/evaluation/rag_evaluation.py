from __future__ import annotations


# RAGAS 0.2.15 import fix — must happen BEFORE any ragas import

import sys
from unittest.mock import MagicMock

if "langchain_community.chat_models.vertexai" not in sys.modules:
    sys.modules["langchain_community.chat_models.vertexai"] = MagicMock()
if "langchain_community.llms.vertexai" not in sys.modules:
    sys.modules["langchain_community.llms.vertexai"] = MagicMock()


# Standard library

import json
import logging
import os
from pathlib import Path


# Third-party

import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# RAGAS 0.2.x
from ragas import EvaluationDataset, RunConfig, SingleTurnSample, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    AnswerCorrectness,
    AnswerRelevancy,
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
)


# Project

from backend.nodes.shared.rag_node import _SYSTEM_PROMPT as _RAG_SYSTEM_PROMPT
from backend.rag.retriever import get_retriever

load_dotenv()
logger = logging.getLogger(__name__)


# Paths

_TEST_CASES_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "backend"
    / "tests"
    / "rag-evaluation"
    / "rag_test_cases.json"
)


# Constants

# Must match the production bot (rag_node.py) — evaluates what we actually ship.
_BOT_MODEL = "gpt-4o-mini"

# Stronger judge for RAGAS metrics (Faithfulness, AnswerCorrectness, etc.).
# gpt-4o produces significantly more reliable scores for multi-step reasoning
# tasks than gpt-4o-mini. Evaluation is an offline job so the cost delta is
# negligible (≈ a few cents per run over ~100 judge calls).
_JUDGE_MODEL = "gpt-4o"

_RAGAS_MAX_WORKERS = 2   # conservative — avoids OpenAI rate-limit errors
_RAGAS_MAX_RETRIES = 3
_LOW_SCORE_THRESHOLD = 0.60



# Helpers


def _load_test_cases() -> list[dict]:
    if not _TEST_CASES_PATH.exists():
        raise FileNotFoundError(
            f"RAG test cases file not found: {_TEST_CASES_PATH}\n"
            "Expected location: data/evaluation/rag_test_cases.json"
        )
    with _TEST_CASES_PATH.open(encoding="utf-8") as f:
        cases = json.load(f)
    logger.info("Loaded %d RAG test cases from %s", len(cases), _TEST_CASES_PATH)
    return cases


def _generate_answer(question: str, context: str, llm: ChatOpenAI) -> str:

    result = llm.invoke([
        SystemMessage(content=_RAG_SYSTEM_PROMPT),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}"),
    ])
    return str(result.content).strip()


def _build_samples(
    test_cases: list[dict],
    retriever,
    llm: ChatOpenAI,
) -> tuple[list[SingleTurnSample], list[dict]]:

    samples: list[SingleTurnSample] = []
    failures: list[dict] = []

    for i, case in enumerate(test_cases):
        question: str = case.get("question", "").strip()
        ground_truth: str = case.get("ground_truth", "").strip()

        if not question or not ground_truth:
            logger.warning("Test case %d skipped — missing question or ground_truth", i)
            failures.append({
                "index": i,
                "question": question,
                "reason": "Missing question or ground_truth in test dataset",
            })
            continue

        logger.info(
            "RAG eval — case %d/%d: %r", i + 1, len(test_cases), question[:70]
        )

        try:
            # ---- Step 1: retrieve context from ChromaDB ----
            docs = retriever.invoke(question)
            if not docs:
                logger.warning("Case %d: retriever returned no documents", i)
                failures.append({
                    "index": i,
                    "question": question,
                    "reason": "Retriever returned no documents",
                })
                continue

            retrieved_contexts: list[str] = [
                doc.page_content for doc in docs if doc.page_content.strip()
            ]
            if not retrieved_contexts:
                failures.append({
                    "index": i,
                    "question": question,
                    "reason": "All retrieved documents had empty page_content",
                })
                continue

            context_blob = "\n\n".join(retrieved_contexts)

            # ---- Step 2: generate answer via production RAG prompt ----
            answer = _generate_answer(question, context_blob, llm)
            if not answer:
                failures.append({
                    "index": i,
                    "question": question,
                    "reason": "LLM returned empty answer",
                })
                continue

            # ---- Step 3: build RAGAS SingleTurnSample ----
            sample = SingleTurnSample(
                user_input=question,
                response=answer,
                retrieved_contexts=retrieved_contexts,
                reference=ground_truth,
            )
            samples.append(sample)

        except Exception as exc:  # noqa: BLE001
            logger.exception("RAG eval — case %d failed: %s", i, exc)
            failures.append({
                "index": i,
                "question": question,
                "reason": f"{type(exc).__name__}: {exc}",
            })

    logger.info(
        "Built %d RAGAS samples, %d failures", len(samples), len(failures)
    )
    return samples, failures


def _safe_float(value) -> float:
    try:
        if value is None:
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _nan_mean(values: list[float]) -> float:
    valid = [v for v in values if not (v != v)]  # NaN != NaN
    return sum(valid) / len(valid) if valid else 0.0



# Public API


def run_rag_evaluation() -> dict:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. "
            "Add it to your .env file before running RAG evaluation."
        )

    
    # 1. Load test cases
    
    test_cases = _load_test_cases()

    
    # 2. Initialise LLM + embeddings (reuse existing project config)
    
    logger.info(
        "RAG evaluation — initialising LLMs | bot=%s | judge=%s",
        _BOT_MODEL,
        _JUDGE_MODEL,
    )
    # Reproduces exactly what the production bot produces — must stay gpt-4o-mini.
    llm = ChatOpenAI(model=_BOT_MODEL, temperature=0, api_key=api_key)

    # Separate, stronger model used exclusively as the RAGAS judge.
    judge_llm = ChatOpenAI(model=_JUDGE_MODEL, temperature=0, api_key=api_key)

    # Wrap for RAGAS
    ragas_llm = LangchainLLMWrapper(judge_llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
    )

    
    # 3. Initialise RAGAS metrics
    
    faithfulness_metric = Faithfulness(llm=ragas_llm)
    answer_relevancy_metric = AnswerRelevancy(
        llm=ragas_llm, embeddings=ragas_embeddings
    )
    context_precision_metric = LLMContextPrecisionWithReference(llm=ragas_llm)
    context_recall_metric = LLMContextRecall(llm=ragas_llm)
    answer_correctness_metric = AnswerCorrectness(
        llm=ragas_llm, embeddings=ragas_embeddings
    )

    metrics = [
        faithfulness_metric,
        answer_relevancy_metric,
        context_precision_metric,
        context_recall_metric,
        answer_correctness_metric,
    ]

    
    # 4. Build retriever (production ChromaDB)
    
    logger.info("RAG evaluation — loading retriever")
    retriever = get_retriever()

    
    # 5. Run retrieval + generation for every test case
    
    samples, failures = _build_samples(test_cases, retriever, llm)

    if not samples:
        raise RuntimeError(
            "No valid RAGAS samples could be built. "
            f"All {len(test_cases)} test cases failed. "
            "Check that ChromaDB is populated (run the ingest script) "
            "and that OPENAI_API_KEY is valid."
        )

    
    # 6. Create RAGAS dataset and run evaluation
    
    logger.info(
        "RAG evaluation — running RAGAS evaluate() on %d samples", len(samples)
    )
    ragas_dataset = EvaluationDataset(samples=samples)

    run_config = RunConfig(
        max_workers=_RAGAS_MAX_WORKERS,
        max_retries=_RAGAS_MAX_RETRIES,
    )

    ragas_result = evaluate(
        dataset=ragas_dataset,
        metrics=metrics,
        run_config=run_config,
    )

    
    # 7. Convert result to DataFrame
    
    result_df: pd.DataFrame = ragas_result.to_pandas()

    # Normalise column names — RAGAS 0.2.x uses these exact names
    col_map = {
        "user_input":        "Question",
        "response":          "Answer",
        "reference":         "Ground Truth",
        "faithfulness":      "Faithfulness",
        "answer_relevancy":  "Answer Relevancy",
        "context_precision": "Context Precision",
        "context_recall":    "Context Recall",
        "answer_correctness":"Answer Correctness",
        # aliases used in some RAGAS 0.2 patch versions
        "llm_context_precision_with_reference": "Context Precision",
        "llm_context_recall":                   "Context Recall",
    }
    result_df = result_df.rename(columns=col_map)

    # Ensure the five metric columns exist (fill NaN if a column is absent)
    for col in [
        "Faithfulness",
        "Answer Relevancy",
        "Context Precision",
        "Context Recall",
        "Answer Correctness",
    ]:
        if col not in result_df.columns:
            result_df[col] = float("nan")

    
    # 8. Compute per-metric averages
    
    avg_faithfulness    = _nan_mean(result_df["Faithfulness"].tolist())
    avg_relevancy       = _nan_mean(result_df["Answer Relevancy"].tolist())
    avg_precision       = _nan_mean(result_df["Context Precision"].tolist())
    avg_recall          = _nan_mean(result_df["Context Recall"].tolist())
    avg_correctness     = _nan_mean(result_df["Answer Correctness"].tolist())

    overall_score = (
        avg_faithfulness
        + avg_relevancy
        + avg_precision
        + avg_recall
        + avg_correctness
    ) / 5

    
    # 9. Round for display
    
    for col in [
        "Faithfulness",
        "Answer Relevancy",
        "Context Precision",
        "Context Recall",
        "Answer Correctness",
    ]:
        result_df[col] = result_df[col].apply(
            lambda v: round(_safe_float(v), 3)
        )

    # Keep only the columns we want to display
    display_cols = [
        c for c in [
            "Question", "Answer", "Ground Truth",
            "Faithfulness", "Answer Relevancy",
            "Context Precision", "Context Recall",
            "Answer Correctness",
        ]
        if c in result_df.columns
    ]
    detail_df = result_df[display_cols].copy()

    logger.info(
        "RAG evaluation complete | overall=%.3f | faithfulness=%.3f | "
        "relevancy=%.3f | precision=%.3f | recall=%.3f | correctness=%.3f | "
        "samples=%d | failures=%d",
        overall_score,
        avg_faithfulness,
        avg_relevancy,
        avg_precision,
        avg_recall,
        avg_correctness,
        len(samples),
        len(failures),
    )

    return {
        "overall_score":       round(overall_score, 3),
        "overall_percentage":  round(overall_score * 100, 1),
        "faithfulness":        round(avg_faithfulness, 3),
        "answer_relevancy":    round(avg_relevancy, 3),
        "context_precision":   round(avg_precision, 3),
        "context_recall":      round(avg_recall, 3),
        "answer_correctness":  round(avg_correctness, 3),
        "total_questions":     len(samples) + len(failures),
        "evaluated_questions": len(samples),
        "failed_questions":    len(failures),
        "details":             detail_df,
        "failures":            failures,
    }
