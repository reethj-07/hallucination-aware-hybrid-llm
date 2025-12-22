import streamlit as st
import requests

API_URL = "http://localhost:8000/query"

st.set_page_config(page_title="Hallucination-Aware RAG", layout="centered")

st.title("🧠 Hallucination-Aware RAG System")

query = st.text_input("Ask a question:")
use_rag = st.checkbox("Use RAG", value=True)

if st.button("Submit") and query:
    resp = requests.post(
        API_URL,
        json={"query": query, "use_rag": use_rag},
        timeout=120
    )

    st.subheader("Answer")
    st.write(resp.json()["answer"])
