from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

PERSIST_DIR = "notes_chroma_db"

PROMPT_TEMPLATE = """
You are answering questions about retailer relationship notes for a B2B
wholesale business.

Use ONLY the notes provided below as context. Do not use outside
knowledge. If the notes don't contain enough information to answer,
say so plainly instead of guessing.

For every claim you make, cite the note it came from using the format
[retailer_name | note_date] immediately after the sentence, like this
example:

Example:
Context:
[Sunbelt School Supply | 2025-03-11] Shipment arrived two weeks late.
[Frontier Sports | 2025-04-02] Requested a discount on bulk orders.

Question: Which retailer had a late shipment?
Answer: Sunbelt School Supply reported a shipment that arrived two
weeks late. [Sunbelt School Supply | 2025-03-11]

Now answer using the real context below.

Context:
{context}

Question: {question}

Answer:
"""

prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])


def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)


def format_context(docs):
    lines = []
    for doc in docs:
        retailer = doc.metadata.get("retailer_name", "unknown retailer")
        date = doc.metadata.get("note_date", "unknown date")
        lines.append(f"[{retailer} | {date}] {doc.page_content}")
    return "\n".join(lines)


def answer_question(query, k=5):
    vectorstore = load_vectorstore()
    docs = vectorstore.similarity_search(query, k=k)

    context = format_context(docs)
    final_prompt = prompt.format(context=context, question=query)

    llm = OllamaLLM(model="llama3.2")
    answer = llm.invoke(final_prompt)

    return {"answer": answer, "retrieved_notes": docs}


if __name__ == "__main__":
    test_questions = [
        "What have retailers said about pricing?",
        "Are there any complaints about shipping delays?",
    ]

    for query in test_questions:
        print("=" * 60)
        print("Question:", query)
        result = answer_question(query)

        print("\nAnswer:\n", result["answer"])
        print("\nRetrieved notes used:")
        for doc in result["retrieved_notes"]:
            print(" -", doc.metadata, "->", doc.page_content)