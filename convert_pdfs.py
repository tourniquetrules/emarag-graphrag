#!/usr/bin/env python3
"""
PDF to Text Converter for GraphRAG
Extracts text from PDF abstracts and saves as text files for GraphRAG processing
"""

import os
import sys
from pathlib import Path
import pypdf
import re

def clean_text(text):
    """Clean extracted text for better processing"""
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text)
    
    # Remove page headers/footers common patterns
    text = re.sub(r'EMA.*?EM_RAP', '', text)
    text = re.sub(r'Abstract \d+_', '', text)
    
    # Remove URLs and email patterns
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    
    # Clean up extra spaces again
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file"""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            text = ""
            
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            
            return clean_text(text)
    
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return None

def convert_pdfs_to_text():
    """Convert all PDF abstracts to text files"""
    abstracts_dir = Path("/home/vboxuser/emarag/abstracts")
    input_dir = Path("input")
    
    if not abstracts_dir.exists():
        print(f"❌ Abstracts directory not found: {abstracts_dir}")
        return False
    
    # Get all PDF files
    pdf_files = list(abstracts_dir.glob("*.pdf"))
    
    if not pdf_files:
        print("❌ No PDF files found in abstracts directory")
        return False
    
    print(f"📄 Found {len(pdf_files)} PDF files to convert")
    
    converted_count = 0
    for pdf_file in pdf_files:
        print(f"Processing: {pdf_file.name}")
        
        # Extract text
        text = extract_text_from_pdf(pdf_file)
        
        if text and len(text.strip()) > 100:  # Only save if we got substantial text
            # Create text filename
            text_filename = pdf_file.stem + ".txt"
            text_path = input_dir / text_filename
            
            # Save text file
            try:
                with open(text_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                
                print(f"✅ Converted: {text_filename}")
                converted_count += 1
                
            except Exception as e:
                print(f"❌ Error saving {text_filename}: {e}")
        else:
            print(f"⚠️  No text extracted from {pdf_file.name}")
    
    print(f"\n🎉 Successfully converted {converted_count} PDFs to text files")
    return converted_count > 0

if __name__ == "__main__":
    success = convert_pdfs_to_text()
    sys.exit(0 if success else 1)
