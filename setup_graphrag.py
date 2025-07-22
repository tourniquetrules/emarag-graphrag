#!/usr/bin/env python3
"""
GraphRAG Enhanced Emergency Medicine RAG System
This script sets up and runs GraphRAG indexing on emergency medicine abstracts
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from dotenv import load_dotenv

def check_environment():
    """Check if required environment variables are set"""
    load_dotenv()
    
    required_vars = ['OPENAI_API_KEY']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("Please update your .env file with the required API keys")
        return False
    
    print("✅ Environment variables configured")
    return True

def setup_directories():
    """Create necessary directories for GraphRAG"""
    directories = ['input', 'output', 'cache', 'prompts']
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Directory created: {directory}")

def install_dependencies():
    """Install required Python packages including clinical models"""
    print("📦 Installing GraphRAG and clinical dependencies...")
    
    try:
        # Install base requirements
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], check=True, capture_output=True, text=True)
        print("✅ Base dependencies installed successfully")
        
        # Install clinical spaCy models
        print("🏥 Installing clinical spaCy models...")
        clinical_models = [
            "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_sm-0.5.1.tar.gz",
            "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_ner_bc5cdr_md-0.5.1.tar.gz"
        ]
        
        for model_url in clinical_models:
            try:
                subprocess.run([
                    sys.executable, "-m", "pip", "install", model_url
                ], check=True, capture_output=True, text=True)
                print(f"✅ Installed clinical model: {model_url.split('/')[-1]}")
            except subprocess.CalledProcessError as e:
                print(f"⚠️  Failed to install clinical model {model_url}: {e}")
        
        # Download Clinical-BERT model (will be done during first use)
        print("🧠 Clinical-BERT will be downloaded on first use")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        print(f"Error output: {e.stderr}")
        return False

def convert_abstracts_to_text():
    """Convert abstracts from original format to text files for GraphRAG"""
    from pathlib import Path
    import shutil
    
    # Check if abstracts exist
    abstracts_source = Path("/home/tourniquetrules/emarag/abstracts")
    input_dir = Path("input")
    
    if not abstracts_source.exists():
        print(f"❌ Source abstracts directory not found: {abstracts_source}")
        return False
    
    # Copy abstracts to input directory
    for abstract_file in abstracts_source.glob("*.txt"):
        shutil.copy2(abstract_file, input_dir)
        print(f"📄 Copied: {abstract_file.name}")
    
    print(f"✅ Copied {len(list(input_dir.glob('*.txt')))} abstracts to input directory")
    return True

def run_graphrag_index():
    """Run GraphRAG indexing process"""
    print("🔍 Starting GraphRAG indexing process...")
    print("This may take several minutes depending on the number of abstracts...")
    
    try:
        # Run GraphRAG indexing using the correct command for GraphRAG 2.4
        result = subprocess.run([
            "graphrag", "index", 
            "--root", "."
        ], capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        
        if result.returncode == 0:
            print("✅ GraphRAG indexing completed successfully!")
            print("📊 Knowledge graph has been created from your emergency medicine abstracts")
            return True
        else:
            print(f"❌ GraphRAG indexing failed")
            print(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ GraphRAG indexing timed out after 1 hour")
        return False
    except Exception as e:
        print(f"❌ Error during GraphRAG indexing: {e}")
        return False

def verify_output():
    """Verify that GraphRAG output was created successfully"""
    output_dir = Path("output")
    
    expected_files = [
        "entities.parquet",
        "relationships.parquet", 
        "communities.parquet",
        "community_reports.parquet",
        "documents.parquet",
        "text_units.parquet"
    ]
    
    missing_files = []
    for file in expected_files:
        if not (output_dir / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"⚠️  Some expected output files are missing: {missing_files}")
        return False
    
    # Also check for LanceDB vector database
    lancedb_dir = output_dir / "lancedb"
    if not lancedb_dir.exists():
        print("⚠️  LanceDB vector database directory missing")
        return False
    
    print("✅ All expected GraphRAG output files were created")
    print("✅ LanceDB vector database is available")
    return True

def main():
    parser = argparse.ArgumentParser(description="Setup GraphRAG for Emergency Medicine RAG")
    parser.add_argument("--skip-install", action="store_true", 
                       help="Skip dependency installation")
    parser.add_argument("--index-only", action="store_true",
                       help="Only run indexing (skip setup steps)")
    parser.add_argument("--clinical", action="store_true",
                       help="Setup with clinical models (Clinical-BERT, spaCy)")
    
    args = parser.parse_args()
    
    print("🏥 GraphRAG Enhanced Emergency Medicine RAG Setup")
    if args.clinical:
        print("🧠 Clinical AI Models Edition")
    print("=" * 50)
    
    if not args.index_only:
        # Environment check
        if not check_environment():
            return 1
        
        # Setup directories
        setup_directories()
        
        # Install dependencies
        if not args.skip_install:
            if not install_dependencies():
                return 1
            
            # Install clinical models if requested
            if args.clinical:
                print("🏥 Setting up clinical models...")
                try:
                    result = subprocess.run([
                        "./setup_clinical_models.sh"
                    ], check=True, capture_output=True, text=True)
                    print("✅ Clinical models setup completed")
                except subprocess.CalledProcessError as e:
                    print(f"⚠️  Clinical models setup had issues: {e}")
        
        # Convert abstracts
        if not convert_abstracts_to_text():
            return 1
    
    # Run GraphRAG indexing
    if not run_graphrag_index():
        return 1
    
    # Verify output
    if not verify_output():
        print("⚠️  Setup completed but some output files may be missing")
        return 0
    
    print("\n🎉 GraphRAG setup completed successfully!")
    if args.clinical:
        print("🧠 Clinical AI models are available!")
        print("\nNext steps:")
        print("1. Run './start_clinical_interface.sh' for clinical-enhanced RAG")
        print("2. Or run 'python clinical_rag.py' directly")
    else:
        print("\nNext steps:")
        print("1. Review the generated knowledge graph in the 'output' directory")
        print("2. Use the query scripts to interact with your GraphRAG system")
        print("3. Integrate GraphRAG queries into your existing emergency medicine RAG")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
