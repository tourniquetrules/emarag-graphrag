#!/bin/bash
# Clinical Models Setup Script for EMARAG-GraphRAG
# This script installs clinical spaCy models and sets up the medical NLP environment

set -e

echo "🏥 Setting up Clinical Models for EMARAG-GraphRAG..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
echo "📋 Python version: $python_version"

# Install clinical spaCy models
echo "📦 Installing clinical spaCy models..."

# Core scientific spaCy model
echo "⬇️  Downloading en_core_sci_sm..."
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_sm-0.5.1.tar.gz

# Biomedical named entity recognition model
echo "⬇️  Downloading en_ner_bc5cdr_md..."
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_ner_bc5cdr_md-0.5.1.tar.gz

# Verify installations
echo "✅ Verifying clinical model installations..."

python3 -c "
import spacy
try:
    nlp = spacy.load('en_core_sci_sm')
    print('✅ en_core_sci_sm loaded successfully')
except Exception as e:
    print(f'❌ Failed to load en_core_sci_sm: {e}')

try:
    nlp = spacy.load('en_ner_bc5cdr_md')
    print('✅ en_ner_bc5cdr_md loaded successfully')
except Exception as e:
    print(f'❌ Failed to load en_ner_bc5cdr_md: {e}')
"

# Test Clinical-BERT model download
echo "🧠 Testing Clinical-BERT model availability..."
python3 -c "
try:
    from transformers import AutoTokenizer, AutoModel
    model_name = 'emilyalsentzer/Bio_ClinicalBERT'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f'✅ Clinical-BERT tokenizer loaded: {model_name}')
    
    # Test model loading (this will download if not cached)
    model = AutoModel.from_pretrained(model_name)
    print(f'✅ Clinical-BERT model loaded: {model_name}')
    
except Exception as e:
    print(f'⚠️  Clinical-BERT setup note: {e}')
    print('🔔 Clinical-BERT will be downloaded automatically on first use')
"

echo ""
echo "🎉 Clinical models setup complete!"
echo ""
echo "📝 Next steps:"
echo "   1. Configure your .env file with API keys"
echo "   2. Add medical literature to the input/ directory"
echo "   3. Run: python setup_graphrag.py"
echo "   4. Launch: python clinical_rag.py"
echo ""
echo "🔗 For issues, visit: https://github.com/tourniquetrules/emarag-graphrag"
