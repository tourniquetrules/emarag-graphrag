# 🏥 EMARAG-GraphRAG: Clinical-Enhanced Emergency Medicine RAG

A sophisticated clinical-enhanced Retrieval Augmented Generation (RAG) system specifically designed for emergency medicine, combining **Clinical-BERT**, **GraphRAG knowledge graphs**, **medical spaCy models**, and **conversational LLMs** for evidence-based medical analysis.

## ✨ Key Features

- 🧠 **Clinical-BERT Integration**: Medical-domain specialized embeddings (emilyalsentzer/Bio_ClinicalBERT)
- 🕸️ **GraphRAG Knowledge Graphs**: Entity-relationship networks from medical literature
- 🏥 **Medical NLP**: spaCy + scispacy for clinical entity recognition and linking
- 🤖 **Multi-Provider LLM Support**: LM Studio, OpenAI, OpenRouter for conversational responses
- 📊 **Cross-Encoder Reranking**: Clinical relevance scoring for better results
- 🎯 **Emergency Medicine Focus**: Optimized prompts and processing for ED scenarios

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/tourniquetrules/emarag-graphrag.git
cd emarag-graphrag

# Create virtual environment
python -m venv graphrag-env
source graphrag-env/bin/activate  # On Windows: graphrag-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your API keys (at least one required):
# - OPENAI_API_KEY for OpenAI models
# - LM_STUDIO_BASE_URL for local LM Studio
# - OPENROUTER_API_KEY for OpenRouter
```

### 3. Add Medical Literature

```bash
# Place your medical abstracts/papers (.txt format) in the input/ directory
# The system comes with sample emergency medicine abstracts
```

### 4. Initialize GraphRAG Knowledge Graph

```bash
# Run GraphRAG indexing to create knowledge graph
python setup_graphrag.py
```

### 5. Launch Clinical Interface

```bash
# Start the clinical-enhanced interface
python clinical_rag.py

# Or use the convenience script
./start_clinical_interface.sh
```

Visit `http://localhost:7860` to access the web interface.

## 🏗️ Architecture

### Clinical AI Stack
```
┌─────────────────────────────────────────────────┐
│                 Gradio Interface                │
├─────────────────────────────────────────────────┤
│  Multi-Provider LLM Layer                      │
│  ├── LM Studio (Local)                         │
│  ├── OpenAI (GPT-4)                           │
│  └── OpenRouter (DeepSeek, etc.)              │
├─────────────────────────────────────────────────┤
│  Clinical Processing Pipeline                   │
│  ├── Clinical-BERT Embeddings                  │
│  ├── Medical spaCy + scispacy                  │
│  ├── Cross-Encoder Reranking                   │
│  └── UMLS Entity Linking                       │
├─────────────────────────────────────────────────┤
│  Knowledge Systems                             │
│  ├── GraphRAG Knowledge Graph                  │
│  ├── FAISS Vector Search                       │
│  └── Medical Entity Networks                   │
├─────────────────────────────────────────────────┤
│  Data Layer                                    │
│  ├── Emergency Medicine Abstracts              │
│  ├── Clinical Entities (280+)                  │
│  └── Medical Relationships (265+)              │
└─────────────────────────────────────────────────┘
```

### Search Methods

1. **Clinical Enhanced Search**
   - Clinical-BERT embeddings for medical domain understanding
   - Medical entity extraction and linking
   - Cross-encoder reranking for clinical relevance
   - LLM-powered clinical analysis

2. **GraphRAG Search**
   - Knowledge graph entity and relationship extraction
   - Community detection in medical literature
   - Structured medical insights

3. **Hybrid Analysis**
   - Combines vector search with graph-based reasoning
   - Comprehensive clinical recommendations

## � Project Structure

```
emarag-graphrag/
├── clinical_rag.py              # Main clinical-enhanced RAG system
├── graphrag_query.py            # GraphRAG query interface
├── setup_graphrag.py            # GraphRAG indexing and setup
├── convert_pdfs.py              # PDF to text conversion utility
├── input/                       # Medical literature (.txt files)
├── output/                      # GraphRAG knowledge graph artifacts
├── prompts/                     # Medical-specific prompts
├── cache/                       # Processing cache
├── settings.yaml               # GraphRAG configuration
├── requirements.txt            # Python dependencies
├── requirements_clinical.txt   # Clinical AI dependencies
├── .env.example               # Environment configuration template
└── scripts/
    ├── start_clinical_interface.sh
    ├── setup_clinical_models.sh
    └── setup_clinical_models_v2.sh
```

## 🧠 Clinical AI Components

### Clinical-BERT Integration
- **Model**: `emilyalsentzer/Bio_ClinicalBERT`
- **Purpose**: Medical domain-specialized embeddings
- **Features**: Clinical terminology understanding, medical context preservation

### Medical spaCy + scispacy
- **Models**: `en_core_sci_sm`, UMLS entity linking
- **Capabilities**: 
  - Medical entity recognition (conditions, treatments, medications)
  - Clinical abbreviation resolution
  - UMLS concept mapping
  - Medical sentence segmentation

### Cross-Encoder Reranking
- **Purpose**: Clinical relevance scoring
- **Models**: Medical cross-encoders for relevance assessment
- **Benefits**: Improved ranking of clinical evidence

## 🤖 LLM Provider Options

### LM Studio (Recommended for Privacy)
- **Models**: DeepSeek, Llama, medical-specialized models
- **Benefits**: Local processing, privacy, cost-effective
- **Setup**: Configure `LM_STUDIO_BASE_URL` in `.env`

### OpenAI
- **Models**: GPT-4, GPT-3.5-turbo
- **Benefits**: High quality, reliable API
- **Setup**: Add `OPENAI_API_KEY` to `.env`

### OpenRouter
- **Models**: Access to multiple providers (DeepSeek, Claude, etc.)
- **Benefits**: Model diversity, competitive pricing
- **Setup**: Configure `OPENROUTER_API_KEY` in `.env`

## 🏥 Medical Entity Types

The system extracts and analyzes:
- **Medical Conditions**: Diseases, syndromes, disorders
- **Treatments**: Procedures, interventions, protocols
- **Medications**: Drugs, dosages, interactions
- **Symptoms**: Signs, presentations, manifestations
- **Anatomy**: Body systems, organs, structures
- **Pathogens**: Bacteria, viruses, organisms
- **Organizations**: Hospitals, departments, agencies
- **Events**: Medical incidents, procedures, outcomes

## 📊 Search Types

### Vector Search
Best for finding similar content and semantic matches:
```python
# Example: Finding abstracts similar to a query
results = rag_system.vector_search("myocardial infarction treatment", top_k=5)
```

### GraphRAG Search
Best for understanding relationships and context:
```python
# Example: Analyzing medical connections
analysis = graph_rag.analyze_medical_connections("sepsis")
```

### Integrated Search
Combines both methods for comprehensive results:
```python
# Example: Full analysis with recommendations
results = rag_system.integrated_search("chest pain diagnosis")
```

## � Usage Examples

### Clinical Query Examples

1. **Probiotics and Pediatric Fever**
   ```
   Query: "What was the effect of probiotics on fever duration in kids?"
   
   System Response:
   - Clinical-BERT identifies pediatric + probiotic + fever entities
   - GraphRAG finds related studies and relationships
   - LLM generates evidence-based clinical summary
   - Result: "RCT showed median fever reduction from 5 to 3 days..."
   ```

2. **CT Scan Cancer Risk Analysis**
   ```
   Query: "CT cancer risk assessment for emergency patients"
   
   System Output:
   - Medical spaCy extracts imaging + radiation entities
   - Cross-encoder ranks relevant oncology literature
   - Clinical analysis includes risk-benefit considerations
   ```

3. **Sepsis Management Protocols**
   ```
   Query: "Early antibiotic therapy for sepsis in ED"
   
   Clinical Enhancement:
   - UMLS linking for sepsis-related concepts
   - Knowledge graph relationships (sepsis ↔ antibiotics ↔ outcomes)
   - Evidence-based timing recommendations
   ```

### Interface Features

- **Provider Selection**: Choose your preferred LLM provider
- **Clinical Toggle**: Enable/disable LLM-powered responses
- **Search Methods**: Clinical Enhanced vs GraphRAG vs Hybrid
- **Evidence Base**: View supporting literature with relevance scores

## ⚙️ Configuration

### Environment Variables (`.env`)
```bash
# LM Studio (Local AI) - Recommended for privacy
LM_STUDIO_BASE_URL=http://localhost:1234

# OpenAI API (Cloud)
OPENAI_API_KEY=your_openai_api_key_here

# OpenRouter (Multiple providers)
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Optional: GPU configuration
CUDA_VISIBLE_DEVICES=0
```

### GraphRAG Settings (`settings.yaml`)
Key configurations for clinical optimization:

```yaml
entity_extraction:
  entity_types: [medical_condition,treatment,medication,symptom,anatomy,pathogen,organization,person,geo,event]
  
llm:
  model: gpt-4o-mini  # or your preferred model
  max_tokens: 4000
  
embeddings:
  vector_store:
    type: lancedb
    db_uri: "./lancedb"
    
chunk_size: 1024  # Larger chunks for medical context
chunk_overlap: 200
```

### Clinical Model Configuration
The system automatically attempts to load clinical models in order of preference:
1. `emilyalsentzer/Bio_ClinicalBERT` (Primary)
2. `dmis-lab/biobert-base-cased-v1.1` (Backup)
3. `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract` (Alternative)

## 🛠️ Advanced Setup

### Local Clinical Models Setup
```bash
# Install clinical spaCy models
./setup_clinical_models.sh

# Or manually:
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_sm-0.5.1.tar.gz
```

### Custom Medical Literature
1. Convert PDFs to text: `python convert_pdfs.py`
2. Place `.txt` files in `input/` directory
3. Re-run GraphRAG indexing: `python setup_graphrag.py`

### Performance Optimization
```yaml
# In settings.yaml
parallelization:
  num_threads: 8
  concurrent_requests: 10

# For large datasets
chunk_size: 2048
max_gleanings: 3
```

## � Troubleshooting

### Common Issues

**1. Clinical models not loading:**
```bash
# Solution: Install clinical dependencies
pip install -r requirements_clinical.txt
./setup_clinical_models.sh
```

**2. GraphRAG indexing fails:**
```bash
# Check API configuration
python -c "import os; print('OpenAI Key:', bool(os.getenv('OPENAI_API_KEY')))"

# Reduce parallelization if memory issues
# Edit settings.yaml: num_threads: 2, concurrent_requests: 2
```

**3. LLM provider connection issues:**
```bash
# Test LM Studio connection
curl http://localhost:1234/v1/models

# Verify API keys in .env file
cat .env | grep -E "(OPENAI|OPENROUTER)_API_KEY"
```

**4. No GraphRAG results:**
```bash
# Verify knowledge graph files
ls -la output/*.parquet

# Check for entities and relationships
python -c "import pandas as pd; print('Entities:', len(pd.read_parquet('output/entities.parquet')))"
```

### Debug Commands

```bash
# Check system status
python clinical_rag.py --status

# Test clinical models
python -c "from clinical_rag import ClinicalEmbeddingModel; m = ClinicalEmbeddingModel(); print('Status:', m.model is not None)"

# Verify GraphRAG output
ls -la output/artifacts/

# Test search functionality
python test_search.py
```

## 📊 System Requirements

### Minimum Requirements
- **RAM**: 8GB (16GB recommended for large literature sets)
- **Storage**: 5GB for models + data
- **Python**: 3.8+ (3.10+ recommended)

### Recommended Hardware
- **GPU**: NVIDIA GPU with 4GB+ VRAM (for local Clinical-BERT)
- **CPU**: Multi-core for parallel processing
- **RAM**: 16GB+ for optimal performance

## 🤝 Contributing

Contributions welcome! Areas for improvement:

1. **Medical Specialization**
   - Add specialty-specific prompts (cardiology, neurology, etc.)
   - Integrate additional clinical NLP models
   - Expand medical entity types

2. **Performance**
   - Optimize clinical model inference
   - Implement caching strategies
   - Add batch processing capabilities

3. **Evaluation**
   - Medical relevance metrics
   - Clinical accuracy benchmarks
   - User feedback integration

### Development Setup
```bash
# Clone repository
git clone https://github.com/tourniquetrules/emarag-graphrag.git

# Install in development mode
pip install -e .

# Run tests
pytest tests/
```

## 📚 References

- [GraphRAG by Microsoft](https://github.com/microsoft/graphrag)
- [Clinical-BERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT)
- [spaCy + scispacy](https://spacy.io/universe/project/scispacy)
- [Emergency Medicine Guidelines](https://www.acep.org/)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Microsoft GraphRAG team for the knowledge graph framework
- Clinical-BERT authors for medical domain embeddings
- spaCy and scispacy teams for medical NLP tools
- Emergency medicine community for clinical insights

---

**Built with ❤️ for the emergency medicine community**

For questions, issues, or contributions, please visit the [GitHub repository](https://github.com/tourniquetrules/emarag-graphrag).
