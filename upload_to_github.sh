#!/bin/bash
# GitHub Upload Helper Script for EMARAG-GraphRAG
# This script helps you upload your repository to GitHub

set -e

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 EMARAG-GraphRAG GitHub Upload Helper${NC}"
echo ""

# Check if we're already in a git repository
if [ -d ".git" ]; then
    echo -e "${YELLOW}⚠️ Git repository already exists. Skipping git init.${NC}"
else
    echo -e "${BLUE}1. Initializing git repository...${NC}"
    git init
    echo -e "${GREEN}✅ Git repository initialized${NC}"
fi

echo ""
echo -e "${BLUE}2. Setting up .gitignore and adding files...${NC}"
git add .
echo -e "${GREEN}✅ Files added to git${NC}"

echo ""
echo -e "${BLUE}3. Creating initial commit...${NC}"
if git diff --cached --quiet; then
    echo -e "${YELLOW}⚠️ No changes to commit${NC}"
else
    git commit -m "Initial commit: EMARAG-GraphRAG Clinical-Enhanced Emergency Medicine RAG

Features:
- Clinical-BERT integration for medical domain embeddings
- GraphRAG knowledge graphs for emergency medicine literature
- Medical spaCy + scispacy for clinical entity recognition
- Multi-provider LLM support (LM Studio, OpenAI, OpenRouter)
- Cross-encoder reranking for clinical relevance
- Gradio web interface for interactive clinical queries

Components:
- clinical_rag.py: Main clinical-enhanced RAG system
- GraphRAG integration with medical literature processing
- Comprehensive documentation and setup scripts
- GitHub Actions workflow for automated testing"

    echo -e "${GREEN}✅ Initial commit created${NC}"
fi

echo ""
echo -e "${BLUE}4. Setting up GitHub remote...${NC}"
if git remote get-url origin >/dev/null 2>&1; then
    echo -e "${YELLOW}⚠️ Remote 'origin' already exists:${NC}"
    git remote get-url origin
else
    echo "Setting up remote for https://github.com/tourniquetrules/emarag-graphrag.git"
    git remote add origin https://github.com/tourniquetrules/emarag-graphrag.git
    echo -e "${GREEN}✅ GitHub remote added${NC}"
fi

echo ""
echo -e "${BLUE}5. Ready to push to GitHub!${NC}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "• Create the repository on GitHub: https://github.com/new"
echo "• Repository name: emarag-graphrag"
echo "• Make it public or private as desired"
echo "• Do NOT initialize with README (we already have one)"
echo ""
echo "Then run:"
echo -e "${GREEN}git branch -M main${NC}"
echo -e "${GREEN}git push -u origin main${NC}"
echo ""
echo -e "${BLUE}🎯 Repository Features Summary:${NC}"
echo "• Clinical-BERT + GraphRAG + Medical spaCy integration"
echo "• Multi-provider LLM support with conversational AI"
echo "• Emergency medicine focus with specialized prompts"
echo "• Comprehensive documentation and setup scripts"
echo "• Automated testing with GitHub Actions"
echo "• Security: .env files properly excluded"
echo ""
echo -e "${GREEN}🎉 Your EMARAG-GraphRAG repository is ready for the world!${NC}"
