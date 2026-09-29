"""
Evaluation suite for Project 5 (B2B hybrid RAG).

Four parts, run independently so you can debug one at a time:
  A. Router classification accuracy
  B. Structured path: golden-answer correctness + run-to-run consistency
  C. Unstructured path: Hit Rate@k, MRR, Context Precision, Recall@k,
     faithfulness (word-overlap), answer relevance
  D. Latency per path (structured / unstructured / hybrid)

NOTE on Part C: HAND_LABELED_RELEVANT_NOTES below is a starting point built
from notes we referenced earlier in this project's testing, NOT verified
against your actual retailer_notes.csv line by line. Open that file
yourself and confirm/correct the (retailer_name, note_date) pairs listed
as "relevant" for each question before trusting these numbers - hand
labeling this correctly is the whole point of Part C, and it only works
if the ground truth is actually right.
"""

import time

from router import answer_question as route_question
from structured_query import answer_question as structured_answer_question
from unstructured import answer_question as unstructured_answer_question, load_vectorstore


# ---------------------------------------------------------------------------
# PART A: Router classification accuracy
# ---------------------------------------------------------------------------

ROUTER_QUESTIONS = [
    {"question": "How many orders are currently backordered?", "expected_label": "structured"},
    {"question": "What is the average unit price for SKU-00001?", "expected_label": "structured"},
    {"question": "Which retailer has the highest backorders?", "expected_label": "structured"},
    {"question": "What have retailers said about pricing?", "expected_label": "unstructured"},
    {"question": "Are there any complaints about shipping delays?", "expected_label": "unstructured"},
    {"question": "Is Sunbelt School Supply a reliable retailer?", "expected_label": "hybrid"},
    {"question": "Which retailers complained about backorders and which are there backorders", "expected_label": "hybrid"},
]


def evaluate_router():
    print("=" * 60)
    print("PART A: Router classification accuracy")
    print("=" * 60)

    correct = 0
    for item in ROUTER_QUESTIONS:
        predicted = route_question(item["question"])
        is_correct = predicted == item["expected_label"]
        correct += int(is_correct)
        status = "PASS" if is_correct else "FAIL"
        print(f"[{status}] '{item['question']}' -> predicted={predicted!r}, expected={item['expected_label']!r}")

    accuracy = correct / len(ROUTER_QUESTIONS)
    print(f"\nRouter accuracy: {correct}/{len(ROUTER_QUESTIONS)} ({accuracy:.0%})")
    return accuracy


# ---------------------------------------------------------------------------
# PART B: Structured golden-answer + consistency
# ---------------------------------------------------------------------------

STRUCTURED_QUESTIONS = [
    {"question": "How many orders are currently backordered?", "expected_substring": "44879"},
    {"question": "What is the average unit price for SKU-00001?", "expected_substring": "51.9"},
    {"question": "How many orders have status 'cancelled'?", "expected_substring": "20160"},
]

RUNS_PER_QUESTION = 5


def evaluate_structured():
    print("\n" + "=" * 60)
    print("PART B: Structured golden-answer + consistency")
    print("=" * 60)

    for item in STRUCTURED_QUESTIONS:
        if item["expected_substring"] is None:
            print(f"\nSKIPPED (no expected_substring set yet): '{item['question']}'")
            continue

        passes = 0
        for run_num in range(1, RUNS_PER_QUESTION + 1):
            result = structured_answer_question(item["question"])
            output = result["execution_result"]["output"] if result["execution_result"]["success"] else ""
            run_pass = item["expected_substring"] in output
            passes += int(run_pass)
            print(f"  run {run_num}/{RUNS_PER_QUESTION}: {'PASS' if run_pass else 'FAIL'} (output={output.strip()!r})")

        print(f"'{item['question']}' -> {passes}/{RUNS_PER_QUESTION} passed")


# ---------------------------------------------------------------------------
# PART C: Unstructured retrieval + faithfulness + relevance
# ---------------------------------------------------------------------------

# TODO: verify these against your actual retailer_notes.csv - see module
# docstring above. Each question lists the (retailer_name, note_date) pairs
# that are ACTUALLY relevant to it, based on your own reading of the notes.
HAND_LABELED_RELEVANT_NOTES = [
    {
        "question": "What have retailers said about pricing?",
        "relevant": [
            ("Evergreen Sporting Goods", "2025-08-30"),
            ("Sunbelt School Supply", "2025-06-11"),
            ("Pacific Outfitters", "2025-07-02"),
            ("Empire Healthcare", "2025-05-05"),
        ],
    },
    {
        "question": "Are there any complaints about shipping delays?",
        "relevant": [
            ("Cascade Medical Group", "2025-04-01"),
            ("Riverbend Medical", "2025-03-14"),
            ("Evergreen Sporting Goods", "2025-04-22"),
        ],
    },
]

K = 5


def _note_key(doc):
    return (doc.metadata.get("retailer_name"), doc.metadata.get("note_date"))


STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "and", "or", "of", "to",
    "in", "on", "for", "with", "as", "that", "this", "it", "has", "have",
    "had", "be", "been", "at", "by", "from", "their", "they", "there",
}


def meaningful_words(text):
    return {w.strip(".,!?[]()'\"").lower() for w in text.split() if w.strip(".,!?[]()'\"").lower() not in STOPWORDS and len(w) > 2}


def evaluate_unstructured():
    print("\n" + "=" * 60)
    print("PART C: Unstructured retrieval + faithfulness + relevance")
    print("=" * 60)

    vectorstore = load_vectorstore()

    hits = 0
    reciprocal_ranks = []
    precisions = []
    recalls = []
    faithfulness_scores = []
    relevance_scores = []

    for item in HAND_LABELED_RELEVANT_NOTES:
        question = item["question"]
        relevant_keys = set(item["relevant"])

        retrieved_docs = vectorstore.similarity_search(question, k=K)
        retrieved_keys = [_note_key(d) for d in retrieved_docs]

        # Hit Rate + MRR
        rank = None
        for i, key in enumerate(retrieved_keys, start=1):
            if key in relevant_keys:
                rank = i
                break
        if rank is not None:
            hits += 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

        # Context Precision: fraction of retrieved that are relevant
        num_retrieved_relevant = sum(1 for key in retrieved_keys if key in relevant_keys)
        precision = num_retrieved_relevant / len(retrieved_keys) if retrieved_keys else 0.0
        precisions.append(precision)

        # Recall@k: fraction of all relevant notes that were retrieved
        recall = num_retrieved_relevant / len(relevant_keys) if relevant_keys else 0.0
        recalls.append(recall)

        # Faithfulness: word overlap between generated answer and retrieved notes
        full_result = unstructured_answer_question(question, k=K)
        answer_text = full_result["answer"]
        context_words = set()
        for doc in retrieved_docs:
            context_words |= meaningful_words(doc.page_content)
        answer_words = meaningful_words(answer_text)
        overlap = answer_words & context_words
        faithfulness = len(overlap) / len(answer_words) if answer_words else 0.0
        faithfulness_scores.append(faithfulness)

        # Answer relevance (simple heuristic): word overlap between question and answer
        question_words = meaningful_words(question)
        relevance_overlap = question_words & answer_words
        relevance = len(relevance_overlap) / len(question_words) if question_words else 0.0
        relevance_scores.append(relevance)

        print(f"\nQuestion: {question}")
        print(f"  retrieved: {retrieved_keys}")
        print(f"  relevant (hand-labeled): {sorted(relevant_keys)}")
        print(f"  hit_rank: {rank}, precision: {precision:.2f}, recall: {recall:.2f}")
        print(f"  faithfulness: {faithfulness:.2f}, relevance: {relevance:.2f}")

    n = len(HAND_LABELED_RELEVANT_NOTES)
    print(f"\nHit Rate@{K}: {hits / n:.2f}")
    print(f"MRR@{K}: {sum(reciprocal_ranks) / n:.2f}")
    print(f"Avg Context Precision: {sum(precisions) / n:.2f}")
    print(f"Avg Recall@{K}: {sum(recalls) / n:.2f}")
    print(f"Avg Faithfulness: {sum(faithfulness_scores) / n:.2f}")
    print(f"Avg Answer Relevance: {sum(relevance_scores) / n:.2f}")


# ---------------------------------------------------------------------------
# PART D: Latency per path
# ---------------------------------------------------------------------------

LATENCY_TEST_QUESTIONS = {
    "structured": "How many orders are currently backordered?",
    "unstructured": "What have retailers said about pricing?",
    "hybrid": "Is Sunbelt School Supply a reliable retailer?",
}


def evaluate_latency():
    print("\n" + "=" * 60)
    print("PART D: Latency per path")
    print("=" * 60)

    for path_name, question in LATENCY_TEST_QUESTIONS.items():
        start = time.time()

        label = route_question(question)
        if label == "structured":
            structured_answer_question(question)
        elif label == "unstructured":
            unstructured_answer_question(question)
        elif label == "hybrid":
            unstructured_result = unstructured_answer_question(question)
            structured_answer_question(question, context=unstructured_result["answer"])

        elapsed = time.time() - start
        print(f"{path_name:>14} path: {elapsed:.2f}s  (question: '{question}')")
        #print(len(structured_answer['code']))


if __name__ == "__main__":
    evaluate_router()
    evaluate_structured()
    evaluate_unstructured()
    evaluate_latency()