
import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma




def fetch_page():
    #df1=pd.read_csv('b2b_wholesale_purchase_orders_1m.csv')
    df2=pd.read_csv('retailer_notes.csv')

    df2=df2[['retailer_name','note_date','note_text']]

    documents=[]

    for index,row in df2.iterrows():
        page_content=( 
            f"Note: {row['note_text']}")

        metadata = {"retailer_name": row["retailer_name"], "note_date": row["note_date"]}

        documents.append(Document(page_content=page_content,metadata=metadata))

    return documents

PERSIST_DIR = "notes_chroma_db"
def main():
    doc=fetch_page()

    print("Chunking...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=10)
    chunks = splitter.split_documents(doc)
    print(f"Split into {len(chunks)} chunks")

    print("Embedding + storing in Chroma...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
   
    vector_store=Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=PERSIST_DIR)
    print(f"Done. Vector DB saved to ./{PERSIST_DIR}")

    results=vector_store.similarity_search('Shipping delays',k=3)

    print(results)
    for r in results:
        print(f"[{r.metadata['retailer_name']} | {r.metadata['note_date']}] {r.page_content}")

    print("\n--- Verification: filtered by retailer ---")
    filtered_results = vector_store.similarity_search(
        "shipping delays", k=3, filter={"retailer_name": "Sunbelt School Supply"}
    )
    for r in filtered_results:
        print(f"[{r.metadata['retailer_name']} | {r.metadata['note_date']}] {r.page_content}")


if __name__ == "__main__":
    main()