#!/usr/bin/env python3
"""
Robust environment setup for GraphRAG Enhanced Emergency Medicine RAG
This script ensures all configurations are correct and dependencies are properly set up.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import yaml
from dotenv import load_dotenv

def check_python_version():
    """Ensure Python 3.8+ is being used"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

def setup_environment():
    """Set up the complete environment"""
    print("🏥 GraphRAG Enhanced Emergency Medicine RAG - Environment Setup")
    print("=" * 70)
    
    # Check Python version
    check_python_version()
    
    # Load environment variables
    load_dotenv()
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found in environment")
        print("   Please set it in .env file or environment variables")
        sys.exit(1)
    print("✅ OpenAI API key found")
    
    # Ensure virtual environment exists
    venv_path = Path("graphrag-env")
    if not venv_path.exists():
        print("🔧 Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", "graphrag-env"], check=True)
    
    # Get the correct Python executable
    if sys.platform == "win32":
        python_exe = venv_path / "Scripts" / "python.exe"
        pip_exe = venv_path / "Scripts" / "pip.exe"
    else:
        python_exe = venv_path / "bin" / "python"
        pip_exe = venv_path / "bin" / "pip"
    
    print(f"✅ Using Python: {python_exe}")
    
    # Upgrade pip
    subprocess.run([str(pip_exe), "install", "--upgrade", "pip"], check=True)
    
    # Install requirements
    if Path("requirements.txt").exists():
        print("📦 Installing requirements...")
        subprocess.run([str(pip_exe), "install", "-r", "requirements.txt"], check=True)
        print("✅ Requirements installed")
    
    # Create necessary directories
    directories = ["input", "output", "cache", "prompts", "logs"]
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"✅ Directory created: {dir_name}")
    
    # Copy abstracts from original system if they exist
    original_abstracts = Path("../emarag/abstracts")
    local_input = Path("input")
    
    if original_abstracts.exists():
        print("📄 Copying abstracts from original system...")
        for abstract_file in original_abstracts.glob("*.txt"):
            dest_file = local_input / abstract_file.name
            if not dest_file.exists():
                shutil.copy2(abstract_file, dest_file)
        
        # Count copied files
        txt_files = list(local_input.glob("*.txt"))
        print(f"✅ {len(txt_files)} abstracts available in input directory")
    else:
        print("⚠️  Original abstracts directory not found - will need to populate input/ manually")
    
    # Validate settings.yaml
    validate_settings()
    
    # Create a startup script
    create_startup_script()
    
    print("\n🎉 Environment setup complete!")
    print("\nNext steps:")
    print("1. Run: ./start_system.sh")
    print("2. Or manually: python integrated_rag.py")
    print("3. Access at: http://localhost:7860")

def validate_settings():
    """Validate and fix settings.yaml configuration"""
    settings_file = Path("settings.yaml")
    
    if not settings_file.exists():
        print("❌ settings.yaml not found")
        return
    
    try:
        with open(settings_file, 'r') as f:
            settings = yaml.safe_load(f)
        
        # Check critical settings
        fixes_needed = []
        
        # Check input configuration
        if 'input' in settings:
            input_config = settings['input']
            if input_config.get('file_pattern') == '.*\\.csv$':
                fixes_needed.append("file_pattern should be '.*\\.txt$' not '.*\\.csv$'")
            if input_config.get('file_type') != 'text':
                fixes_needed.append("file_type should be 'text'")
        
        # Check claim_extraction
        if 'claim_extraction' in settings:
            if 'enabled' not in settings['claim_extraction']:
                fixes_needed.append("claim_extraction missing 'enabled' field")
        
        if fixes_needed:
            print("⚠️  Settings validation issues found:")
            for fix in fixes_needed:
                print(f"   - {fix}")
            
            # Auto-fix common issues
            fix_settings(settings_file)
        else:
            print("✅ settings.yaml validation passed")
            
    except Exception as e:
        print(f"❌ Error validating settings.yaml: {e}")

def fix_settings(settings_file):
    """Fix common settings.yaml issues"""
    print("🔧 Auto-fixing settings.yaml...")
    
    # Read the file as text to preserve formatting
    with open(settings_file, 'r') as f:
        content = f.read()
    
    # Apply fixes
    fixes = [
        ('.*\\.csv$', '.*\\.txt$'),
        ('file_type: csv', 'file_type: text'),
        ('.*\\.txt$$', '.*\\.txt$'),  # Fix double dollar signs
    ]
    
    for old, new in fixes:
        if old in content:
            content = content.replace(old, new)
            print(f"   ✅ Fixed: {old} → {new}")
    
    # Ensure claim_extraction has enabled field
    if 'claim_extraction:' in content and 'enabled:' not in content.split('claim_extraction:')[1].split('\n')[0:5]:
        # Add enabled: false after claim_extraction:
        content = content.replace(
            'claim_extraction:\n',
            'claim_extraction:\n  enabled: false\n'
        )
        print("   ✅ Added enabled: false to claim_extraction")
    
    # Write back
    with open(settings_file, 'w') as f:
        f.write(content)
    
    print("✅ settings.yaml fixes applied")

def create_startup_script():
    """Create a reliable startup script"""
    script_content = '''#!/bin/bash
# Reliable startup script for GraphRAG Enhanced Emergency Medicine RAG

set -e

echo "🏥 Starting GraphRAG Enhanced Emergency Medicine RAG"
echo "=" * 50

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d "graphrag-env" ]; then
    echo "❌ Virtual environment not found. Please run setup_environment.py first."
    exit 1
fi

# Activate virtual environment
if [ -f "graphrag-env/bin/activate" ]; then
    source graphrag-env/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "❌ Could not activate virtual environment"
    exit 1
fi

# Check for required files
if [ ! -f "integrated_rag.py" ]; then
    echo "❌ integrated_rag.py not found"
    exit 1
fi

if [ ! -f "settings.yaml" ]; then
    echo "❌ settings.yaml not found"
    exit 1
fi

if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found - ensure OPENAI_API_KEY is set in environment"
fi

# Count input files
INPUT_COUNT=$(find input -name "*.txt" -type f 2>/dev/null | wc -l)
echo "📄 Found $INPUT_COUNT text files in input directory"

if [ $INPUT_COUNT -eq 0 ]; then
    echo "⚠️  No input files found. The system will work but with limited functionality."
    echo "   Consider copying abstracts to the input/ directory."
fi

# Check if ports are available
if lsof -Pi :7860 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "⚠️  Port 7860 is already in use. Attempting to stop existing process..."
    pkill -f "integrated_rag.py" || true
    sleep 2
fi

echo "🚀 Starting integrated RAG interface..."
echo "   Access at: http://localhost:7860"
echo "   Press Ctrl+C to stop"
echo

# Start the application
python integrated_rag.py
'''
    
    script_path = Path("start_system.sh")
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make executable
    os.chmod(script_path, 0o755)
    print("✅ Created start_system.sh startup script")

if __name__ == "__main__":
    setup_environment()
