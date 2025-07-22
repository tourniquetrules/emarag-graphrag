#!/usr/bin/env python
"""
Simple GraphRAG indexing script to test functionality
"""
import os
import sys
from pathlib import Path
import asyncio
from graphrag.index import run_pipeline_with_config
from graphrag.config import create_graphrag_config

def main():
    # Set up paths
    root_dir = Path("/home/tourniquetrules/emarag-graphrag")
    input_dir = root_dir / "input" 
    output_dir = root_dir / "output"
    cache_dir = root_dir / "cache"
    
    # Ensure directories exist
    output_dir.mkdir(exist_ok=True)
    cache_dir.mkdir(exist_ok=True)
    
    # Check if we have input files
    txt_files = list(input_dir.glob("*.txt"))
    print(f"Found {len(txt_files)} text files for processing")
    
    if len(txt_files) == 0:
        print("No text files found in input directory")
        return
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("ERROR: OPENAI_API_KEY not found in environment")
        return
    
    print("API key loaded successfully")
    print(f"Processing {len(txt_files)} files...")
    
    # Try to load the configuration
    try:
        config = create_graphrag_config(root_dir=str(root_dir))
        print("Configuration loaded successfully")
        print(f"Config root: {config.root_dir}")
        
        # Test a simple run
        print("Starting indexing process...")
        asyncio.run(run_pipeline_with_config(config))
        print("Indexing completed successfully!")
        
    except Exception as e:
        print(f"Error during indexing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
