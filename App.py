import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/ask"

st.title("B2B Retail Intelligence Assistant")
st.write(
    "Ask a question about wholesale purchase orders, retailer notes, "
    "or both — this routes to structured data, unstructured notes, "
    "or a combination of the two."
)

question = st.text_input("Your question:")

if st.button("Ask") and question:
    with st.spinner("Thinking... (structured questions are usually fast, "
                     "hybrid questions can take up to a minute)"):
        try:
            response = requests.post(API_URL, json={"question": question}, timeout=120)
        except requests.exceptions.ConnectionError:
            st.error(
                "Couldn't reach the API. Is it running? "
                "Start it with: uvicorn api:app --reload"
            )
        else:
            if response.status_code == 200:
                answer = response.json()["answer"]
                st.markdown(answer)
            else:
                st.error(f"API returned an error (status {response.status_code}): {response.text}")