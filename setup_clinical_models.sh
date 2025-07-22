#!/bin/bash

# Clinical Model Setup for GraphRAG Emergency Medicine
echo "🏥 Setting up Clinical AI Models for Emergency Medicine RAG"
echo "=========================================================="

# Activate virtual environment
if [ -f "graphrag-env/bin/activate" ]; then
    source graphrag-env/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "❌ Virtual environment not found. Please create it first."
    exit 1
fi

# Install clinical dependencies
echo "📦 Installing clinical dependencies..."
pip install -r requirements.txt

# Install scispacy medical models
echo "🧠 Installing sciSpaCy medical models..."
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_sm-0.5.1.tar.gz
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_ner_bc5cdr_md-0.5.1.tar.gz

# Download Clinical-BERT and related models (will happen on first use)
echo "🧠 Clinical-BERT models will be downloaded automatically on first use"

# Test the installation
echo "🔬 Testing clinical model installation..."
python3 -c "
import spacy
import scispacy
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel

print('✅ Testing spaCy...')
try:
    nlp = spacy.load('en_core_sci_sm')
    print('✅ Medical spaCy model loaded successfully')
except Exception as e:
    print(f'⚠️  Medical spaCy model issue: {e}')

print('✅ Testing sentence transformers...')
try:
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    print('✅ Sentence transformers working')
except Exception as e:
    print(f'⚠️  Sentence transformers issue: {e}')

print('✅ Testing Clinical-BERT access...')
try:
    tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
    print('✅ Clinical-BERT tokenizer accessible')
except Exception as e:
    print(f'⚠️  Clinical-BERT access issue: {e}')

print('🎉 Clinical model setup test completed!')
"

echo "🎉 Clinical model setup completed!"
echo ""
echo "Next steps:"
echo "1. Run 'python clinical_rag.py' to start the clinical-enhanced interface"
echo "2. Or run 'python setup_graphrag.py --clinical' to setup with clinical models"
