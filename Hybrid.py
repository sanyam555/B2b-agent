from router import answer_question as router
from unstructured import answer_question as unstructured_answer_question
from structured_query import answer_question as structured_answer_question

query="Is Sunbelt School Supply a reliable retailer?"

answer=router(query)

if answer == "structured":
    result = structured_answer_question(query)
    if result['execution_result']['success']:
        answer = f"• {result['execution_result']['output'].strip()}"
    else:
        answer = f"• Couldn't compute this: {result['execution_result']['error'].strip()}"

elif answer == "unstructured":
    result = unstructured_answer_question(query)
    answer = f"• {result['answer'].strip()}"

elif answer == "hybrid":
    unstructured_answer = unstructured_answer_question(query)
    structured_answer = structured_answer_question(query, context=unstructured_answer['answer'])

    if structured_answer['execution_result']['success']:
        structured_result = structured_answer['execution_result']['output'].strip()
    else:
        structured_result = f"Couldn't compute this: {structured_answer['execution_result']['error'].strip()}"

    unstructured_result = unstructured_answer['answer'].strip()

    answer = (
        "• From order data:\n"
        f"  {structured_result}\n\n"
        "• From retailer notes:\n"
        f"  {unstructured_result}"
    )

    print(answer)
    #print(structured_answer['code'])

    print("next line \n")

    print(structured_answer['execution_result']['output'])

    print("code /n")

    print(structured_answer['code'])