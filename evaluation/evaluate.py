# """
# Offline RAGAS scoring script. Runs the RAG pipeline against a small labeled
# test set and reports faithfulness, answer relevancy, and context precision --
# turning "is our RAG actually grounded, not hallucinating" into a number
# instead of a hunch.

# Run from the project root:
#     python -m evaluation.evaluate
# """

# import json
# from dotenv import load_dotenv

# from ragas import SingleTurnSample, EvaluationDataset, evaluate
# from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision
# from ragas.llms import LangchainLLMWrapper
# from ragas.embeddings import LangchainEmbeddingsWrapper

# from retrieval.retriever import get_retriever
# from generation.llm import get_llm
# from generation.prompt import RAG_PROMPT
# from generation.parser import answer_parser
# from ingestion.embed_store import _get_embeddings


# def load_testset(path: str = "evaluation/testset.json"):
#     with open(path, "r", encoding="utf-8") as f:
#         return json.load(f)


# def run_pipeline_on_question(question: str, retriever, llm):
#     """
#     Runs the same document-route pipeline as backend/main.py: retrieve,
#     format context, generate a structured answer. Returns the plain-text
#     answer and the raw retrieved chunk texts (what RAGAS calls
#     retrieved_contexts).
#     """
#     docs = retriever.invoke(question)
#     contexts = [doc.page_content for doc in docs]
#     context_str = "\n\n".join(
#         f"[source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}" for d in docs
#     )
#     chain = RAG_PROMPT | llm | answer_parser
#     result = chain.invoke({"context": context_str, "question": question})
#     return result.answer, contexts


# def main():
#     load_dotenv()

#     testset = load_testset()
#     retriever = get_retriever()
#     llm = get_llm()

#     print(f"Running the RAG pipeline on {len(testset)} test questions...")
#     samples = []
#     for item in testset:
#         question = item["question"]
#         ground_truth = item["ground_truth"]
#         answer, contexts = run_pipeline_on_question(question, retriever, llm)
#         samples.append(
#             SingleTurnSample(
#                 user_input=question,
#                 response=answer,
#                 retrieved_contexts=contexts,
#                 reference=ground_truth,
#             )
#         )
#         print(f"  done: {question[:60]}...")

#     dataset = EvaluationDataset(samples=samples)

#     # RAGAS needs its own LLM/embeddings to *judge* the answers -- reusing
#     # the same chat model and the same local embeddings the project already
#     # uses, so no extra API key is needed just to run evaluation.
#     evaluator_llm = LangchainLLMWrapper(llm)
#     evaluator_embeddings = LangchainEmbeddingsWrapper(_get_embeddings())

#     metrics = [
#         Faithfulness(llm=evaluator_llm),
#         AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
#         ContextPrecision(llm=evaluator_llm),
#     ]

#     print("\nScoring with RAGAS (faithfulness, answer relevancy, context precision)...")
#     result = evaluate(dataset=dataset, metrics=metrics)

#     print("\n=== RAGAS Evaluation Results ===")
#     print(result)

#     df = result.to_pandas()
#     df.to_csv("evaluation/results.csv", index=False)
#     print("\nPer-question breakdown saved to evaluation/results.csv")


# if __name__ == "__main__":
#     main()



"""
Offline RAGAS scoring script. Runs the RAG pipeline against a small labeled
test set and reports faithfulness, answer relevancy, and context precision --
turning "is our RAG actually grounded, not hallucinating" into a number
instead of a hunch.

Run from the project root:
    python -m evaluation.evaluate
"""

import json

from dotenv import load_dotenv

from ragas import SingleTurnSample, EvaluationDataset, evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from retrieval.retriever import get_retriever
from generation.llm import get_llm
from generation.prompt import RAG_PROMPT
from generation.parser import answer_parser
from ingestion.embed_store import _get_embeddings


def load_testset(path: str = "evaluation/testset.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_pipeline_on_question(question: str, retriever, llm):
    """
    Runs the same document-route pipeline as backend/main.py:
    retrieve, format context, generate a structured answer.

    Returns:
        answer: Plain-text generated answer
        contexts: Raw retrieved chunk texts
    """

    docs = retriever.invoke(question)

    contexts = [doc.page_content for doc in docs]

    context_str = "\n\n".join(
        f"[source: {d.metadata.get('source', 'unknown')}]\n"
        f"{d.page_content}"
        for d in docs
    )

    chain = RAG_PROMPT | llm | answer_parser

    result = chain.invoke(
        {
            "context": context_str,
            "question": question,
        }
    )

    return result.answer, contexts


def main():
    load_dotenv()

    # ---------------------------------------------------------
    # CI/CD quality thresholds
    # ---------------------------------------------------------
    thresholds = {
        "faithfulness": 0.80,
        "answer_relevancy": 0.80,
        "context_precision": 0.80,
    }

    # ---------------------------------------------------------
    # Load test set
    # ---------------------------------------------------------
    testset = load_testset()

    # ---------------------------------------------------------
    # Initialize RAG components
    # ---------------------------------------------------------
    retriever = get_retriever()
    llm = get_llm()

    print(
        f"Running the RAG pipeline on "
        f"{len(testset)} test questions..."
    )

    # ---------------------------------------------------------
    # Run RAG pipeline on every test question
    # ---------------------------------------------------------
    samples = []

    for item in testset:
        question = item["question"]
        ground_truth = item["ground_truth"]

        answer, contexts = run_pipeline_on_question(
            question,
            retriever,
            llm,
        )

        samples.append(
            SingleTurnSample(
                user_input=question,
                response=answer,
                retrieved_contexts=contexts,
                reference=ground_truth,
            )
        )

        print(f"  done: {question[:60]}...")

    # ---------------------------------------------------------
    # Create RAGAS evaluation dataset
    # ---------------------------------------------------------
    dataset = EvaluationDataset(
        samples=samples
    )

    # ---------------------------------------------------------
    # Configure RAGAS evaluator
    # ---------------------------------------------------------
    # RAGAS uses its own LLM and embeddings to judge the
    # generated answers and retrieved contexts.
    evaluator_llm = LangchainLLMWrapper(llm)

    evaluator_embeddings = LangchainEmbeddingsWrapper(
        _get_embeddings()
    )

    metrics = [
        Faithfulness(
            llm=evaluator_llm
        ),
        AnswerRelevancy(
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
        ),
        ContextPrecision(
            llm=evaluator_llm
        ),
    ]

    # ---------------------------------------------------------
    # Run RAGAS evaluation
    # ---------------------------------------------------------
    print(
        "\nScoring with RAGAS "
        "(faithfulness, answer relevancy, context precision)..."
    )

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
    )

    # ---------------------------------------------------------
    # Display RAGAS results
    # ---------------------------------------------------------
    print("\n=== RAGAS Evaluation Results ===")
    print(result)

    # ---------------------------------------------------------
    # Convert results to DataFrame
    # ---------------------------------------------------------
    # RAGAS result contains per-question metric values.
    # We use the DataFrame to calculate aggregate mean scores.
    df = result.to_pandas()

    # ---------------------------------------------------------
    # Save per-question results
    # ---------------------------------------------------------
    df.to_csv(
        "evaluation/results.csv",
        index=False,
    )

    print(
        "\nPer-question breakdown saved to "
        "evaluation/results.csv"
    )

    # ---------------------------------------------------------
    # Calculate aggregate scores
    # ---------------------------------------------------------
    scores = {
        "faithfulness": df["faithfulness"].mean(),
        "answer_relevancy": df["answer_relevancy"].mean(),
        "context_precision": df["context_precision"].mean(),
    }

    # ---------------------------------------------------------
    # CI/CD Quality Gate
    # ---------------------------------------------------------
    print("\n=== CI/CD Quality Gate ===")

    failed = False

    for metric, score in scores.items():
        threshold = thresholds[metric]

        if score >= threshold:
            status = "PASS"
        else:
            status = "FAIL"
            failed = True

        print(
            f"{metric}: {score:.4f} "
            f"(threshold: {threshold:.2f}) -> {status}"
        )

    # ---------------------------------------------------------
    # Fail CI/CD if any metric is below threshold
    # ---------------------------------------------------------
    if failed:
        print("\nRAG evaluation failed quality gate.")
        raise SystemExit(1)

    print("\nRAG evaluation passed quality gate.")


if __name__ == "__main__":
    main()