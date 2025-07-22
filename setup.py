#!/usr/bin/env python3
"""
Setup script for EMARAG-GraphRAG: Clinical-Enhanced Emergency Medicine RAG
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = []
try:
    with open('requirements.txt', 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
except FileNotFoundError:
    pass

setup(
    name="emarag-graphrag",
    version="1.0.0",
    author="tourniquetrules",
    description="Clinical-Enhanced Emergency Medicine RAG with GraphRAG integration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tourniquetrules/emarag-graphrag",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "clinical": [
            "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_sm-0.5.1.tar.gz",
            "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_ner_bc5cdr_md-0.5.1.tar.gz",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
        ],
        "viz": [
            "plotly>=5.15.0",
            "pyvis>=0.3.2",
        ]
    },
    entry_points={
        "console_scripts": [
            "emarag-clinical=clinical_rag:main",
            "emarag-setup=setup_graphrag:main",
            "emarag-query=graphrag_query:main",
        ],
    },
    keywords=[
        "emergency medicine", 
        "RAG", 
        "GraphRAG", 
        "clinical AI", 
        "medical NLP", 
        "Clinical-BERT",
        "healthcare AI",
        "medical literature"
    ],
    project_urls={
        "Bug Reports": "https://github.com/tourniquetrules/emarag-graphrag/issues",
        "Source": "https://github.com/tourniquetrules/emarag-graphrag",
        "Documentation": "https://github.com/tourniquetrules/emarag-graphrag#readme",
    },
)
