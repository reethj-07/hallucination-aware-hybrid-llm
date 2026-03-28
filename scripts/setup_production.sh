#!/bin/bash

set -e

echo "🔨 Building production knowledge base..."

# Create directory structure
mkdir -p data/rag_docs
mkdir -p rag/faiss_index

# Generate documents
python scripts/generate_production_docs.py

# Build FAISS index
echo "📚 Indexing documents with FAISS..."
python rag/ingest_docs.py

# Create directory for models
mkdir -p models

echo "✅ Production setup complete!"
echo ""
echo "📊 Knowledge base:"
ls -lh data/rag_docs/ | tail -1
echo "$(ls data/rag_docs | wc -l) documents ready"
echo ""
echo "🗂️  FAISS index:"
ls -lh rag/faiss_index/
