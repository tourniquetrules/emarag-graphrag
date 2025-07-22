#!/bin/bash
# Startup script for GraphRAG Enhanced Emergency Medicine RAG

set -e

echo "🏥 GraphRAG Enhanced Emergency Medicine RAG"
echo "============================================"

# Activate virtual environment
echo "🔄 Activating virtual environment..."
if [ -f "graphrag-env/bin/activate" ]; then
    source graphrag-env/bin/activate
    echo "✅ Virtual environment activated: $VIRTUAL_ENV"
    echo "🐍 Using Python: $(which python)"
else
    echo "❌ Virtual environment not found at graphrag-env/bin/activate"
    echo "   Please create it with: python -m venv graphrag-env"
    exit 1
fi

# Check if GraphRAG indexing has completed
if [ -f "output/entities.parquet" ] && [ -f "output/relationships.parquet" ] && [ -f "output/communities.parquet" ]; then
    echo "✅ GraphRAG indexing completed - all artifacts found"
    echo "🚀 Starting integrated RAG interface..."
    python integrated_rag.py
else
    echo "⚠️  GraphRAG indexing not yet complete"
    echo "📊 Checking indexing status..."
    
    if pgrep -f "graphrag index" > /dev/null; then
        echo "🔄 GraphRAG indexing is still running..."
        echo "   You can monitor progress with: tail -f output/indexing.log"
        echo "   Or start with vector search only: python integrated_rag.py"
        
        read -p "Start interface now with vector search only? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "🚀 Starting interface with vector search..."
            python integrated_rag.py
        else
            echo "ℹ️  Run this script again after indexing completes"
        fi
    else
        echo "❌ GraphRAG indexing not found"
        echo "   Run: graphrag index --root ."
        echo "   Or start with vector search only"
        
        read -p "Start interface with vector search only? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "🚀 Starting interface with vector search..."
            python integrated_rag.py
        fi
    fi
fi
