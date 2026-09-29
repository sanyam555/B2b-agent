from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate


PROMPT_TEMPLATE = """
You are a question router.

Your ONLY job is to classify the user's question into:
structured, unstructured, or hybrid.

structured = requires structured data
unstructured = requires unstructured text/documents/notes
hybrid = requires both

Do NOT answer the question.
Do NOT explain
Return exactly ONE word

Structured example:
Question: How many orders are currently backordered?
Answer: structured

Unstructured example:
Question: What have retailers said about pricing?
Answer: unstructured

Hybrid example:
Question: Is Sunbelt School Supply a reliable retailer?
Answer: hybrid


Question:{question}

Answer:
"""
prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["question"])

def answer_question(query):
    final_prompt=prompt.format(question=query)
    llm = OllamaLLM(model="llama3.2")
    answer = llm.invoke(final_prompt)
    answer=answer.strip().lower()
    return answer

if __name__ == "__main__":
    query = "What's the average order value for Frontier Sports over the last year?"
    answer= answer_question(query)
    print(answer)


