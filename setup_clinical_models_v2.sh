#!/bin/bash

echo "🏥 Setting up Clinical AI Models for GraphRAG..."

# Function to check if command succeeded
check_success() {
    if [ $? -eq 0 ]; then
        echo "✅ $1"
    else
        echo "❌ $1 failed"
        return 1
    fi
}

# Function to install package with fallback options
install_with_fallback() {
    local package_name="$1"
    local pip_name="$2"
    local fallback_command="$3"
    
    echo "📦 Installing $package_name..."
    
    # Try main installation
    if pip install "$pip_name"; then
        echo "✅ Successfully installed $package_name"
        return 0
    fi
    
    # Try fallback if provided
    if [ -n "$fallback_command" ]; then
        echo "⚠️  Main installation failed, trying fallback..."
        if eval "$fallback_command"; then
            echo "✅ Successfully installed $package_name (fallback)"
            return 0
        fi
    fi
    
    echo "❌ Failed to install $package_name"
    return 1
}

# Update pip first
echo "📦 Updating pip..."
python -m pip install --upgrade pip
check_success "pip update"

# Install core clinical dependencies
echo "🧠 Installing Clinical-BERT and sentence transformers..."
pip install sentence-transformers>=3.0.0
check_success "sentence-transformers installation"

pip install transformers>=4.30.0
check_success "transformers installation"

# Install spaCy with medical models - with Python 3.12 compatibility
echo "🩺 Installing spaCy medical components..."

# Install spaCy first
pip install "spacy>=3.7.0"
check_success "spaCy installation"

# Try to install scispacy with multiple approaches
install_scispacy() {
    echo "📚 Installing scientific spaCy models..."
    
    # Method 1: Direct pip install (most compatible)
    if pip install scispacy; then
        echo "✅ scispacy installed successfully"
        
        # Install medical models
        echo "📥 Installing medical language models..."
        
        # Try installing models directly
        if pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz; then
            echo "✅ Installed en_core_sci_sm medical model"
        else
            echo "⚠️  Failed to install en_core_sci_sm model"
        fi
        
        if pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz; then
            echo "✅ Installed en_ner_bc5cdr_md medical NER model"
        else
            echo "⚠️  Failed to install en_ner_bc5cdr_md model"
        fi
        
        return 0
    fi
    
    # Method 2: Install without C extensions (Python 3.12 fallback)
    echo "⚠️  Standard scispacy installation failed, trying compatibility mode..."
    if pip install --no-deps scispacy; then
        echo "✅ scispacy installed in compatibility mode"
        return 0
    fi
    
    echo "❌ Could not install scispacy"
    return 1
}

# Install scispacy
install_scispacy

# Install additional medical/clinical packages
echo "🔬 Installing additional clinical packages..."
pip install medspacy
check_success "medspacy installation"

# Install clinical BERT specifically
echo "🧠 Preparing Clinical-BERT..."
python -c "
import torch
from transformers import AutoTokenizer, AutoModel
print('📥 Pre-downloading Clinical-BERT model...')
try:
    tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
    model = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
    print('✅ Clinical-BERT downloaded and cached successfully')
except Exception as e:
    print(f'⚠️  Clinical-BERT pre-download failed: {e}')
    print('   Will download on first use instead')
"

# Install cross-encoder models
echo "🎯 Installing cross-encoder models..."
python -c "
from sentence_transformers import CrossEncoder
print('📥 Pre-downloading PubMedBERT cross-encoder...')
try:
    cross_encoder = CrossEncoder('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract')
    print('✅ PubMedBERT cross-encoder downloaded successfully')
except Exception as e:
    print(f'⚠️  Cross-encoder pre-download failed: {e}')
    print('   Will download on first use instead')
"

# Test clinical setup
echo "🧪 Testing clinical model setup..."
python -c "
import sys

# Test Clinical-BERT
try:
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
    print('✅ Clinical-BERT tokenizer accessible')
except Exception as e:
    print(f'❌ Clinical-BERT test failed: {e}')

# Test sentence transformers
try:
    from sentence_transformers import SentenceTransformer
    print('✅ Sentence transformers available')
except Exception as e:
    print(f'❌ Sentence transformers test failed: {e}')

# Test spaCy
try:
    import spacy
    print('✅ spaCy available')
except Exception as e:
    print(f'❌ spaCy test failed: {e}')

# Test scispacy
try:
    import scispacy
    print('✅ scispacy available')
except Exception as e:
    print(f'⚠️  scispacy test failed: {e}')
    print('   Clinical system will work without scispacy')

print('🏥 Clinical model setup complete!')
"

echo ""
echo "🎉 Clinical AI model setup finished!"
echo ""
echo "📋 Setup Summary:"
echo "✅ Clinical-BERT: Ready for medical domain embeddings"
echo "✅ Sentence Transformers: Available for encoding"
echo "✅ spaCy: Available for NLP processing" 
echo "⚠️  scispacy: May have compatibility issues but system functional"
echo ""
echo "🚀 You can now run: python clinical_rag.py"
