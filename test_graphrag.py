#!/usr/bin/env python
"""
Simple GraphRAG test
"""
import os
from pathlib import Path
from dotenv import load_dotenv

def main():
    print("Starting GraphRAG test...")
    
    # Load environment variables
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not openai_api_key:
        print("ERROR: OPENAI_API_KEY not found")
        return
    
    print("API key loaded")
    
    # Check input files
    input_dir = Path("input")
    txt_files = list(input_dir.glob("*.txt"))
    print(f"Found {len(txt_files)} text files")
    
    # Try importing GraphRAG
    try:
        from graphrag.config.load_config import load_config
        print("GraphRAG config module imported")
        
        config = load_config(Path("."))
        print(f"Config created for root: {config.root_dir}")
        
        # Try to run indexing
        from graphrag.cli.index import index_cli
        
        print("Starting pipeline...")
        result = index_cli(
            root_dir=Path("."),
            verbose=True,
            resume=None,
            memprofile=False,
            cache=True,
            reporter="rich",
            config_filepath=None,
            emit=["parquet"],
            dryrun=False,
            init_content_dir=False,
            skip_validation=False
        )
        print("Pipeline completed!")
        print(f"Result: {result}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
