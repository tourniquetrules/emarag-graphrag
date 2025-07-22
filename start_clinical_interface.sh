#!/bin/bash

set -e

echo "🏥 GraphRAG Enhanced Emergency Medicine RAG - Clinical Edition"
echo "============================================================="

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

# Check for clinical models
echo "🔬 Checking clinical model availability..."

# Test if clinical models are available
CLINICAL_AVAILABLE=$(python3 -c "
try:
    import spacy
    import scispacy
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer
    
    # Test spaCy medical model
    try:
        nlp = spacy.load('en_core_sci_sm')
        print('clinical_spacy_ok')
    except:
        print('clinical_spacy_missing')
    
    # Test Clinical-BERT
    try:
        tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
        print('clinical_bert_ok')
    except:
        print('clinical_bert_missing')
        
except ImportError:
    print('clinical_libs_missing')
")

if [[ "$CLINICAL_AVAILABLE" == *"clinical_libs_missing"* ]]; then
    echo "⚠️  Clinical libraries not installed. Installing now..."
    ./setup_clinical_models.sh
elif [[ "$CLINICAL_AVAILABLE" == *"clinical_spacy_missing"* ]] || [[ "$CLINICAL_AVAILABLE" == *"clinical_bert_missing"* ]]; then
    echo "⚠️  Some clinical models missing. Installing now..."
    ./setup_clinical_models.sh
else
    echo "✅ Clinical models available"
fi

# Check if GraphRAG indexing has completed
if [ -f "output/entities.parquet" ] && [ -f "output/relationships.parquet" ] && [ -f "output/communities.parquet" ]; then
    echo "✅ GraphRAG knowledge graph detected"
    echo "🚀 Starting Clinical-Enhanced RAG Interface..."
    python clinical_rag.py
else
    echo "⚠️  GraphRAG knowledge graph not found"
    echo "   You can still use Clinical Enhanced mode without GraphRAG"
    echo "   Or run 'python setup_graphrag.py' to create the knowledge graph first"
    echo ""
    echo "🚀 Starting Clinical RAG Interface (without GraphRAG)..."
    python clinical_rag.py
fi
