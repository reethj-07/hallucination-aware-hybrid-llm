import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY")

st.set_page_config(
    page_title="Hallucination-Aware RAG",
    layout="centered"
)

st.title("🧠 Hallucination-Aware RAG System")

query = st.text_input("Ask a question:")
use_rag = st.checkbox("Use RAG", value=True)

if st.button("Submit"):
    if not query.strip():
        st.warning("Please enter a question")
    else:
        with st.spinner("Thinking..."):
            headers = {"x-api-key": API_KEY} if API_KEY else {}
            try:
                response = requests.post(
                    f"{API_URL}/query",
                    json={"query": query, "use_rag": use_rag},
                    headers=headers,
                    timeout=20,
                )
                response.raise_for_status()
                result = response.json()
            except requests.RequestException as exc:
                st.error(f"API error: {exc}")
                st.stop()

        if result.get("error"):
            st.error(f"RAG error: {result['error']}")

        st.markdown("### ✅ Answer")
        st.write(result["answer"])
        st.caption(f"RAG used: {result['used_rag']}")

        if result.get("used_rag"):
            with st.expander("🔍 Retrieved documents"):
                for i, doc in enumerate(result.get("retrieved_documents", []), 1):
                    st.markdown(f"**Document {i}:**\n{doc}")

        if result.get("citations"):
            with st.expander("📌 Citations"):
                for citation in result["citations"]:
                    st.write(
                        f"Span {citation['start']}-{citation['end']} -> Doc {citation['doc_index'] + 1}"
                    )
                    st.caption(citation["snippet"])
