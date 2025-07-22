#!/bin/bash
# GitHub Repository Check Script
# Verifies that the repository is ready for upload to GitHub

echo "🔍 Checking EMARAG-GraphRAG repository for GitHub readiness..."
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check functions
check_file() {
    if [ -f "$1" ]; then
        echo -e "${GREEN}✅${NC} $1 exists"
        return 0
    else
        echo -e "${RED}❌${NC} $1 missing"
        return 1
    fi
}

check_directory() {
    if [ -d "$1" ]; then
        echo -e "${GREEN}✅${NC} $1/ directory exists"
        return 0
    else
        echo -e "${YELLOW}⚠️${NC} $1/ directory missing (optional)"
        return 1
    fi
}

check_gitignore_entry() {
    if grep -q "$1" .gitignore 2>/dev/null; then
        echo -e "${GREEN}✅${NC} .gitignore covers $1"
        return 0
    else
        echo -e "${RED}❌${NC} .gitignore missing $1"
        return 1
    fi
}

errors=0

echo -e "${BLUE}📄 Essential Files Check:${NC}"
check_file "README.md" || ((errors++))
check_file "LICENSE" || ((errors++))
check_file "requirements.txt" || ((errors++))
check_file ".gitignore" || ((errors++))
check_file ".env.example" || ((errors++))
check_file "clinical_rag.py" || ((errors++))
echo ""

echo -e "${BLUE}📁 Directory Structure Check:${NC}"
check_directory ".github/workflows"
check_directory "input"
check_directory "prompts"
echo ""

echo -e "${BLUE}🔒 Security Check:${NC}"
if [ -f ".env" ]; then
    echo -e "${RED}❌${NC} .env file found - this should not be committed!"
    ((errors++))
else
    echo -e "${GREEN}✅${NC} .env file not present (good)"
fi

check_gitignore_entry ".env" || ((errors++))
check_gitignore_entry "*.pdf" || ((errors++))
check_gitignore_entry "output/" || ((errors++))
echo ""

echo -e "${BLUE}📝 Documentation Check:${NC}"
check_file "CONTRIBUTING.md"
if [ -f "README.md" ]; then
    if grep -q "Clinical-Enhanced" README.md; then
        echo -e "${GREEN}✅${NC} README.md contains project description"
    else
        echo -e "${YELLOW}⚠️${NC} README.md may need better description"
    fi
fi
echo ""

echo -e "${BLUE}🔧 Configuration Check:${NC}"
check_file "settings.yaml"
check_file "setup.py"
if [ -f "install_clinical_models.sh" ] && [ -x "install_clinical_models.sh" ]; then
    echo -e "${GREEN}✅${NC} install_clinical_models.sh is executable"
else
    echo -e "${YELLOW}⚠️${NC} install_clinical_models.sh not executable or missing"
fi
echo ""

echo -e "${BLUE}🧪 Code Quality Check:${NC}"
python_files=$(find . -name "*.py" -not -path "./.*" -not -path "./*venv*" -not -path "./*env*" -not -path "./venv/*" -not -path "./graphrag-env/*" -not -path "./__pycache__/*")
for file in $python_files; do
    if python3 -m py_compile "$file" 2>/dev/null; then
        echo -e "${GREEN}✅${NC} $file syntax OK"
    else
        echo -e "${RED}❌${NC} $file has syntax errors"
        ((errors++))
    fi
done
echo ""

# Summary
echo -e "${BLUE}📊 Summary:${NC}"
if [ $errors -eq 0 ]; then
    echo -e "${GREEN}🎉 Repository is ready for GitHub!${NC}"
    echo ""
    echo -e "${BLUE}Next steps:${NC}"
    echo "1. Initialize git repository: git init"
    echo "2. Add files: git add ."
    echo "3. Initial commit: git commit -m 'Initial commit: EMARAG-GraphRAG Clinical RAG System'"
    echo "4. Add remote: git remote add origin https://github.com/tourniquetrules/emarag-graphrag.git"
    echo "5. Push to GitHub: git push -u origin main"
    echo ""
    echo -e "${YELLOW}⚠️ Remember to:${NC}"
    echo "• Create .env file locally after cloning (don't commit it)"
    echo "• Add your API keys to the .env file"
    echo "• Install clinical models with ./install_clinical_models.sh"
    echo "• Test the system before publishing"
else
    echo -e "${RED}❌ Found $errors issues that should be fixed before uploading${NC}"
    echo ""
    echo -e "${BLUE}🔧 Recommended fixes:${NC}"
    echo "• Ensure all essential files are present"
    echo "• Fix any syntax errors in Python files"
    echo "• Update .gitignore to cover sensitive files"
    echo "• Remove any .env or sensitive files"
fi

exit $errors
