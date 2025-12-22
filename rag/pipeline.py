def run_rag_pipeline(query, retriever, generator, use_rag=True):
    context = ""

    if use_rag:
        docs = retriever.retrieve(query)
        context = "\n".join(docs)

    prompt = f"""
Use ONLY the context below to answer the question.
If the answer is not in the context, say "Not found in retrieved documents".

Context:
{context}

Question:
{query}

Answer:
"""

    answer = generator.generate(prompt)

    grounded = "Not found" not in answer

    return {
        "answer": answer,
        "context": context,
        "grounded": grounded
    }
