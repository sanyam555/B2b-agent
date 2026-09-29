from fastapi import FastAPI
from pydantic import BaseModel

from router import answer_question as route_question
from structured_query import answer_question as structured_answer_question
from unstructured import answer_question as unstructured_answer_question


def handle_question(query: str) -> str:
    label = route_question(query)

    if label == "structured":
        result = structured_answer_question(query)
        if result["execution_result"]["success"]:
            return f"• {result['execution_result']['output'].strip()}"
        else:
            return f"• Couldn't compute this: {result['execution_result']['error'].strip()}"

    elif label == "unstructured":
        result = unstructured_answer_question(query)
        return f"• {result['answer'].strip()}"

    elif label == "hybrid":
        unstructured_answer = unstructured_answer_question(query)
        structured_answer = structured_answer_question(query, context=unstructured_answer["answer"])

        if structured_answer["execution_result"]["success"]:
            structured_result = structured_answer["execution_result"]["output"].strip()
        else:
            structured_result = f"Couldn't compute this: {structured_answer['execution_result']['error'].strip()}"

        unstructured_result = unstructured_answer["answer"].strip()

        return (
            "• From order data:\n"
            f"  {structured_result}\n\n"
            "• From retailer notes:\n"
            f"  {unstructured_result}"
        )

    else:
        return f"Couldn't classify this question (got label: {label!r})"


class QuestionRequest(BaseModel):
    question: str


app = FastAPI()


@app.post("/ask")
def ask(request: QuestionRequest):
    answer = handle_question(request.question)
    return {"answer": answer}


if __name__ == "__main__":
    # quick standalone test, no server needed - run this file directly
    # with `python api.py` to sanity-check handle_question() before
    # wiring up the actual API with `uvicorn api:app --reload`
    test_questions = [
        "How many orders are currently backordered?",
        "What have retailers said about pricing?",
        "Is Sunbelt School Supply a reliable retailer?",
    ]
    for q in test_questions:
        print("=" * 60)
        print("Question:", q)
        print(handle_question(q))