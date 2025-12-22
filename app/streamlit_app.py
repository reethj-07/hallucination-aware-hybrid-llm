import streamlit as st
import requests

API_URL = "http://localhost:8000/query"

st.title("Hallucination-Aware RAG Assistant")

query = st.text_input("Ask a question")

rag_enabled = st.toggle("Enable RAG", value=True)

if st.button("Ask"):
    payload = {
        "query": query,
        "use_rag": rag_enabled
    }

    response = requests.post(API_URL, json=payload).json()

    st.subheader("Answer")
    st.write(response["answer"])

    if rag_enabled:
        st.subheader("Retrieved Context")
        st.write(response["context"])

    st.subheader("Grounded")
    st.write(response["grounded"])
