import streamlit as st
from rag.rag_inference import run_rag_pipeline

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
            result = run_rag_pipeline(
                query=query,
                use_rag=use_rag
            )

        if result.get("error"):
            st.error(f"RAG error: {result['error']}")
            st.caption("Falling back to non-RAG response.")

        st.markdown("### ✅ Answer")
        st.write(result["answer"])
        st.caption(f"RAG used: {result['used_rag']}")

        if result["used_rag"]:
            with st.expander("🔍 Retrieved documents"):
                for i, doc in enumerate(result["retrieved_documents"], 1):
                    st.markdown(f"**Document {i}:**\n{doc}")
