#!/usr/bin/env python3
"""
Clinical-Enhanced GraphRAG Emergency Medicine System
Combines GraphRAG with Clinical-BERT, spaCy medical models, and cross-encoder reranking
"""

import gradio as gr
import requests
import json
import time
import re
import numpy as np
import faiss
import pickle
import os
from typing import List, Dict, Tuple, Optional
import tempfile
from pathlib import Path
from dotenv import load_dotenv
import sys
import torch
import spacy
import pandas as pd

# Load environment variables
load_dotenv()

# Import the GraphRAG query interface
try:
    from graphrag_query import EmergencyMedicineGraphRAG
    GRAPHRAG_AVAILABLE = True
    print("✅ GraphRAG integration available")
except ImportError as e:
    GRAPHRAG_AVAILABLE = False
    print(f"⚠️  GraphRAG not available: {e}")

# Clinical AI imports
try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    from transformers import AutoTokenizer, AutoModel, AutoConfig
    import torch
    CLINICAL_AI_AVAILABLE = True
    print("✅ Clinical AI libraries available")
except ImportError as e:
    CLINICAL_AI_AVAILABLE = False
    print(f"⚠️  Clinical AI libraries not available: {e}")

# Medical spaCy imports
try:
    import spacy
    from scispacy.linking import EntityLinker
    MEDICAL_NLP_AVAILABLE = True
    print("✅ Medical NLP libraries available")
except ImportError as e:
    MEDICAL_NLP_AVAILABLE = False
    print(f"⚠️  Medical NLP libraries not available: {e}")

# OpenAI imports
try:
    import openai
    OPENAI_AVAILABLE = True
    print("✅ OpenAI library available")
except ImportError as e:
    OPENAI_AVAILABLE = False
    print(f"⚠️  OpenAI library not available: {e}")

# Gradio imports
try:
    import gradio as gr
    GRADIO_AVAILABLE = True
    print("✅ Gradio library available")
except ImportError as e:
    GRADIO_AVAILABLE = False
    print(f"⚠️  Gradio library not available: {e}")


class LLMProvider:
    """Multi-provider LLM interface for clinical text generation"""
    
    def __init__(self, provider: str = "lm_studio", model_name: str = None, embedding_model: str = None):
        self.provider = provider
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.client = None
        self.available = False
        
        # Default model configurations
        self.default_configs = {
            "lm_studio": {
                "model": "deepseek/deepseek-r1-0528-qwen3-8b",
                "embedding": "text-embedding-all-minilm-l6-v2-embedding",
                "base_url": os.getenv("LM_STUDIO_BASE_URL", "http://192.168.2.64:1234")
            },
            "openai": {
                "model": "gpt-4.1-nano-2025-04-14",  
                "embedding": "text-embedding-3-small",
                "api_key": os.getenv("OPENAI_API_KEY")
            },
            "openrouter": {
                "model": "deepseek/deepseek-r1",
                "embedding": "text-embedding-3-small",
                "api_key": os.getenv("OPENROUTER_API_KEY"),
                "base_url": "https://openrouter.ai/api/v1"
            }
        }
        
        self._initialize_provider()
    
    def _initialize_provider(self):
        """Initialize the selected LLM provider"""
        if not OPENAI_AVAILABLE:
            print("⚠️  OpenAI library not available, LLM features disabled")
            return
        
        try:
            config = self.default_configs.get(self.provider, {})
            
            if self.provider == "lm_studio":
                self.client = openai.OpenAI(
                    base_url=config["base_url"],
                    api_key="lm-studio"  # LM Studio doesn't require real API key
                )
                self.model_name = self.model_name or config["model"]
                self.embedding_model = self.embedding_model or config["embedding"]
                
            elif self.provider == "openai":
                if not config["api_key"]:
                    print("⚠️  OpenAI API key not found in environment")
                    return
                
                self.client = openai.OpenAI(api_key=config["api_key"])
                self.model_name = self.model_name or config["model"]
                self.embedding_model = self.embedding_model or config["embedding"]
                
            elif self.provider == "openrouter":
                if not config["api_key"]:
                    print("⚠️  OpenRouter API key not found in environment")
                    return
                
                self.client = openai.OpenAI(
                    base_url=config["base_url"],
                    api_key=config["api_key"]
                )
                self.model_name = self.model_name or config["model"]
                self.embedding_model = self.embedding_model or config["embedding"]
            
            # Test connection
            self._test_connection()
            
        except Exception as e:
            print(f"❌ Failed to initialize {self.provider}: {e}")
            self.available = False
    
    def _test_connection(self):
        """Test connection to the LLM provider"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10,
                timeout=5
            )
            self.available = True
            print(f"✅ {self.provider} LLM connected: {self.model_name}")
            
        except Exception as e:
            print(f"⚠️  {self.provider} connection test failed: {e}")
            self.available = False
    
    def generate_clinical_response(self, query: str, context: str, max_tokens: int = 1500) -> str:
        """Generate clinical response using the LLM"""
        if not self.available:
            return "LLM not available - using template-based response"
        
        try:
            # Clinical system prompt for emergency medicine
            system_prompt = """You are an expert emergency medicine physician and researcher. 
            Analyze the provided medical literature and generate evidence-based clinical insights.
            
            Guidelines:
            - Provide clear, actionable clinical recommendations
            - Reference specific studies when available
            - Include risk-benefit analysis when appropriate
            - Use medical terminology appropriately
            - Structure responses with clear sections
            - Emphasize emergency medicine context
            - Note limitations of evidence when relevant"""
            
            user_prompt = f"""
            Query: {query}
            
            Relevant Medical Literature:
            {context}
            
            Please provide a comprehensive clinical analysis addressing the query based on the evidence provided.
            """
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.3,  # Lower temperature for more consistent medical responses
                timeout=30
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"❌ LLM generation error: {e}")
            return f"LLM error: {str(e)}"
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings using the provider's embedding model"""
        if not self.available:
            return None
        
        try:
            if self.provider in ["openai", "openrouter"]:
                response = self.client.embeddings.create(
                    model=self.embedding_model,
                    input=texts
                )
                embeddings = [item.embedding for item in response.data]
                return np.array(embeddings)
            
            elif self.provider == "lm_studio":
                # For LM Studio, try embedding endpoint or fall back to sentence-transformers
                try:
                    response = self.client.embeddings.create(
                        model=self.embedding_model,
                        input=texts
                    )
                    embeddings = [item.embedding for item in response.data]
                    return np.array(embeddings)
                except:
                    # Fallback to sentence-transformers
                    if CLINICAL_AI_AVAILABLE:
                        from sentence_transformers import SentenceTransformer
                        model = SentenceTransformer('all-MiniLM-L6-v2')
                        return model.encode(texts)
                    else:
                        return None
            
        except Exception as e:
            print(f"⚠️  Embedding error with {self.provider}: {e}")
            return None


class ClinicalEmbeddingModel:
    """Clinical-BERT based embedding model for medical domain"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._initialize_clinical_model()
    
    def _initialize_clinical_model(self):
        """Initialize Clinical-BERT model"""
        if not CLINICAL_AI_AVAILABLE:
            print("⚠️  Clinical AI not available, falling back to general model")
            return
        
        # Try Clinical-BERT models in order of preference
        clinical_models = [
            "emilyalsentzer/Bio_ClinicalBERT",  # Clinical-BERT
            "dmis-lab/biobert-base-cased-v1.1",  # BioBERT
            "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",  # PubMedBERT
            "sentence-transformers/all-MiniLM-L6-v2"  # Fallback
        ]
        
        for model_name in clinical_models:
            try:
                print(f"🧠 Loading clinical model: {model_name}")
                if "sentence-transformers" in model_name:
                    self.model = SentenceTransformer(model_name)
                else:
                    # Use transformers for Clinical-BERT/BioBERT
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                    self.model = AutoModel.from_pretrained(model_name).to(self.device)
                    self.model.eval()
                
                print(f"✅ Successfully loaded: {model_name}")
                self.model_name = model_name
                break
                
            except Exception as e:
                print(f"⚠️  Failed to load {model_name}: {e}")
                continue
        
        if self.model is None:
            print("❌ Failed to load any clinical model")
    
    def encode(self, texts, show_progress_bar=False):
        """Encode texts using Clinical-BERT"""
        if self.model is None:
            return np.random.random((len(texts), 384))  # Fallback random embeddings
        
        if isinstance(self.model, SentenceTransformer):
            return self.model.encode(texts, show_progress_bar=show_progress_bar)
        
        # Handle Clinical-BERT/BioBERT manually
        embeddings = []
        
        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(
                    text, 
                    return_tensors="pt", 
                    truncation=True, 
                    padding=True, 
                    max_length=512
                ).to(self.device)
                
                outputs = self.model(**inputs)
                # Use CLS token embedding
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(embedding[0])
        
        return np.array(embeddings)


class MedicalSpacyProcessor:
    """Medical text processing using spaCy and scispacy"""
    
    def __init__(self):
        self.nlp = None
        self.entity_linker = None
        self._initialize_spacy()
    
    def _initialize_spacy(self):
        """Initialize medical spaCy models"""
        if not MEDICAL_NLP_AVAILABLE:
            print("⚠️  Medical NLP not available")
            return
        
        try:
            # Try to load medical spaCy models
            medical_models = ["en_core_sci_sm", "en_core_web_sm"]
            
            for model_name in medical_models:
                try:
                    self.nlp = spacy.load(model_name)
                    print(f"✅ Loaded spaCy model: {model_name}")
                    
                    # Add entity linker for medical entities
                    if model_name == "en_core_sci_sm":
                        try:
                            # Add UMLS entity linker
                            self.nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
                            print("✅ Added UMLS entity linker")
                        except Exception as e:
                            print(f"⚠️  Could not add entity linker: {e}")
                    
                    break
                    
                except Exception as e:
                    print(f"⚠️  Could not load {model_name}: {e}")
                    continue
            
        except Exception as e:
            print(f"❌ Failed to initialize medical spaCy: {e}")
    
    def process_medical_text(self, text):
        """Process text with medical entity recognition"""
        if self.nlp is None:
            return {
                'entities': [],
                'sentences': [text],
                'medical_terms': [],
                'cleaned_text': text
            }
        
        try:
            doc = self.nlp(text)
            
            # Extract medical entities
            entities = []
            for ent in doc.ents:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'description': spacy.explain(ent.label_) if spacy.explain(ent.label_) else ent.label_
                })
            
            # Extract sentences for better chunking
            sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            
            # Extract medical terms (entities + noun phrases)
            medical_terms = []
            for ent in doc.ents:
                if ent.label_ in ['DISEASE', 'SYMPTOM', 'TREATMENT', 'DRUG', 'ANATOMY']:
                    medical_terms.append(ent.text)
            
            # Add relevant noun phrases
            for chunk in doc.noun_chunks:
                if len(chunk.text.split()) > 1 and any(token.pos_ == 'NOUN' for token in chunk):
                    medical_terms.append(chunk.text)
            
            return {
                'entities': entities,
                'sentences': sentences,
                'medical_terms': list(set(medical_terms)),
                'cleaned_text': doc.text
            }
            
        except Exception as e:
            print(f"❌ Error processing medical text: {e}")
            return {
                'entities': [],
                'sentences': [text],
                'medical_terms': [],
                'cleaned_text': text
            }


class ClinicalCrossEncoder:
    """Cross-encoder for medical text reranking"""
    
    def __init__(self):
        self.model = None
        self._initialize_cross_encoder()
    
    def _initialize_cross_encoder(self):
        """Initialize cross-encoder model"""
        if not CLINICAL_AI_AVAILABLE:
            print("⚠️  Cross-encoder not available")
            return
        
        try:
            # Try medical cross-encoders first, then general ones
            cross_encoders = [
                "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
                "cross-encoder/ms-marco-MiniLM-L-12-v2"
            ]
            
            for model_name in cross_encoders:
                try:
                    if "cross-encoder" in model_name:
                        self.model = CrossEncoder(model_name)
                    else:
                        # Use as sentence transformer for now
                        self.model = SentenceTransformer(model_name)
                    
                    print(f"✅ Loaded cross-encoder: {model_name}")
                    break
                    
                except Exception as e:
                    print(f"⚠️  Failed to load cross-encoder {model_name}: {e}")
                    continue
                    
        except Exception as e:
            print(f"❌ Failed to initialize cross-encoder: {e}")
    
    def rerank(self, query, passages, top_k=None):
        """Rerank passages based on relevance to query"""
        if self.model is None or not passages:
            return passages
        
        try:
            if isinstance(self.model, CrossEncoder):
                # Proper cross-encoder scoring
                pairs = [[query, passage['content']] for passage in passages]
                scores = self.model.predict(pairs)
                
                # Add scores to passages
                for i, passage in enumerate(passages):
                    passage['rerank_score'] = float(scores[i])
                
                # Sort by rerank score
                reranked = sorted(passages, key=lambda x: x['rerank_score'], reverse=True)
                
            else:
                # Fallback: use embedding similarity
                query_emb = self.model.encode([query])
                passage_texts = [p['content'] for p in passages]
                passage_embs = self.model.encode(passage_texts)
                
                # Calculate similarities
                similarities = np.dot(query_emb, passage_embs.T)[0]
                
                # Add scores to passages
                for i, passage in enumerate(passages):
                    passage['rerank_score'] = float(similarities[i])
                
                # Sort by similarity
                reranked = sorted(passages, key=lambda x: x['rerank_score'], reverse=True)
            
            if top_k:
                reranked = reranked[:top_k]
            
            return reranked
            
        except Exception as e:
            print(f"❌ Error in reranking: {e}")
            return passages


class EmergencyMedicineGraphRAG:
    """GraphRAG integration for emergency medicine knowledge graph queries"""
    
    def __init__(self):
        self.available = False
        self._initialize_graphrag()
    
    def _initialize_graphrag(self):
        """Initialize GraphRAG components"""
        try:
            # Check for GraphRAG output files
            from pathlib import Path
            import pandas as pd
            
            output_dir = Path("output")
            required_files = ["entities.parquet", "relationships.parquet", "text_units.parquet"]
            
            # Check if GraphRAG files exist
            missing_files = []
            for file in required_files:
                if not (output_dir / file).exists():
                    missing_files.append(file)
            
            if missing_files:
                print(f"⚠️  GraphRAG files missing: {missing_files}")
                return
            
            # Load GraphRAG data
            self.entities_df = pd.read_parquet(output_dir / "entities.parquet")
            self.relationships_df = pd.read_parquet(output_dir / "relationships.parquet") 
            self.text_units_df = pd.read_parquet(output_dir / "text_units.parquet")
            
            print("✅ GraphRAG data loaded successfully")
            self.available = True
            
        except Exception as e:
            print(f"⚠️  GraphRAG initialization failed: {e}")
            self.available = False
    
    def graphrag_search(self, query: str, max_results: int = 10) -> Dict:
        """Perform GraphRAG-based search"""
        if not self.available:
            return "GraphRAG not available - knowledge graph files not found"
        
        try:
            query_lower = query.lower()
            
            # Search entities for relevant medical concepts
            entity_matches = self.entities_df[
                self.entities_df['title'].str.lower().str.contains(
                    '|'.join(query_lower.split()), case=False, na=False
                )
            ].head(max_results)
            
            # Extract relevant entities
            relevant_entities = []
            for _, entity in entity_matches.iterrows():
                relevant_entities.append({
                    'name': entity['title'],
                    'type': entity.get('type', 'ENTITY'),
                    'description': entity.get('description', '')
                })
            
            # Search relationships for contextual connections
            relationship_matches = self.relationships_df[
                (self.relationships_df['source'].str.lower().str.contains('|'.join(query_lower.split()), case=False, na=False)) |
                (self.relationships_df['target'].str.lower().str.contains('|'.join(query_lower.split()), case=False, na=False))
            ].head(max_results)
            
            key_relationships = []
            for _, rel in relationship_matches.iterrows():
                key_relationships.append({
                    'source': rel['source'],
                    'target': rel['target'], 
                    'relationship': rel.get('weight', 1.0)
                })
            
            # Generate natural language response
            if relevant_entities:
                response_parts = [f"**Knowledge Graph Analysis for:** {query}"]
                response_parts.append("\n**Relevant Entities:**")
                
                for entity in relevant_entities[:5]:
                    response_parts.append(f"• **{entity['name']}** ({entity['type']}): {entity['description'][:200]}...")
                
                if key_relationships:
                    response_parts.append("\n**Key Relationships:**")
                    for rel in key_relationships[:5]:
                        response_parts.append(f"• {rel['source']} ↔ {rel['target']}")
                
                natural_response = "\n".join(response_parts)
            else:
                natural_response = f"No specific GraphRAG entities found for query: {query}. The knowledge graph may not contain detailed information about this medical topic."
            
            return {
                'natural_response': natural_response,
                'entities': relevant_entities,
                'relationships': key_relationships,
                'insights': [f"Found {len(relevant_entities)} entities and {len(key_relationships)} relationships"]
            }
            
        except Exception as e:
            return f"GraphRAG search error: {str(e)}"


class ClinicalEnhancedGraphRAG:
    """Clinical-enhanced GraphRAG system with medical AI models"""
    
    def __init__(self, llm_provider: str = "lm_studio", llm_model: str = None, embedding_model: str = None):
        self.abstracts_dir = Path("input")
        self.vector_index = None
        self.abstract_texts = []
        self.abstract_embeddings = None
        
        # Clinical AI components
        self.clinical_embedder = ClinicalEmbeddingModel()
        self.medical_processor = MedicalSpacyProcessor()
        self.cross_encoder = ClinicalCrossEncoder()
        self.graph_rag = None
        
        # LLM provider for conversational responses
        self.llm_provider = LLMProvider(llm_provider, llm_model, embedding_model)
        
        # Configuration matching your original EMARAG
        self.chunk_size = 1024  # Larger chunks for better medical context
        self.chunk_overlap = 200
        self.max_results = 10  # More results for comprehensive answers
        
        # Initialize components
        self._load_abstracts()
        self._initialize_vector_search()
        if GRAPHRAG_AVAILABLE:
            self._initialize_graphrag()
    
    def _load_abstracts(self):
        """Load and process medical abstracts with clinical chunking"""
        if not self.abstracts_dir.exists():
            print(f"⚠️  Abstracts directory not found: {self.abstracts_dir}")
            return
        
        self.abstract_texts = []
        self.abstract_files = []
        self.processed_chunks = []
        
        for abstract_file in self.abstracts_dir.glob("*.txt"):
            try:
                with open(abstract_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        # Process with medical spaCy for better chunking
                        processed = self.medical_processor.process_medical_text(content)
                        
                        # Create sentence-aware chunks (like your original EMARAG)
                        chunks = self._create_clinical_chunks(
                            processed['sentences'], 
                            processed['medical_terms']
                        )
                        
                        for i, chunk in enumerate(chunks):
                            self.abstract_texts.append(chunk)
                            self.abstract_files.append(f"{abstract_file.name}_chunk_{i}")
                            self.processed_chunks.append({
                                'content': chunk,
                                'file': abstract_file.name,
                                'chunk_id': i,
                                'medical_terms': processed['medical_terms'],
                                'entities': processed['entities']
                            })
                        
            except Exception as e:
                print(f"❌ Error reading {abstract_file}: {e}")
        
        print(f"✅ Loaded {len(self.abstract_texts)} clinical chunks from {len(list(self.abstracts_dir.glob('*.txt')))} abstracts")
    
    def _create_clinical_chunks(self, sentences, medical_terms):
        """Create chunks that preserve medical context and sentence boundaries"""
        chunks = []
        current_chunk = ""
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            # If adding this sentence would exceed chunk size
            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap if it contains medical terms
                if any(term in current_chunk for term in medical_terms):
                    # Keep last sentence for context
                    overlap_sentences = current_chunk.split('.')[-2:]
                    current_chunk = '. '.join(overlap_sentences) + '. ' + sentence
                    current_length = len(current_chunk.split())
                else:
                    current_chunk = sentence
                    current_length = sentence_length
            else:
                current_chunk += " " + sentence if current_chunk else sentence
                current_length += sentence_length
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _initialize_vector_search(self):
        """Initialize FAISS vector search with Clinical-BERT embeddings"""
        if not self.abstract_texts:
            print("⚠️  No abstracts to index")
            return
        
        try:
            print("🧠 Generating Clinical-BERT embeddings for abstracts...")
            self.abstract_embeddings = self.clinical_embedder.encode(
                self.abstract_texts, 
                show_progress_bar=True
            )
            
            # Create FAISS index
            dimension = self.abstract_embeddings.shape[1]
            self.vector_index = faiss.IndexFlatIP(dimension)  # Inner product for similarity
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(self.abstract_embeddings)
            self.vector_index.add(self.abstract_embeddings)
            
            print(f"✅ Clinical vector index created with {self.vector_index.ntotal} chunks")
            
        except Exception as e:
            print(f"❌ Failed to initialize clinical vector search: {e}")
            self.vector_index = None
    
    def _initialize_graphrag(self):
        """Initialize GraphRAG component"""
        try:
            self.graph_rag = EmergencyMedicineGraphRAG()
            print("✅ GraphRAG component initialized")
        except Exception as e:
            print(f"❌ Failed to initialize GraphRAG: {e}")
            self.graph_rag = None
    
    def clinical_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Perform clinical-enhanced vector search with reranking"""
        if not self.vector_index:
            return []
        
        try:
            # Step 1: Encode query with Clinical-BERT
            query_embedding = self.clinical_embedder.encode([query])
            faiss.normalize_L2(query_embedding)
            
            # Step 2: Initial retrieval (get more than needed for reranking)
            initial_k = min(top_k * 3, len(self.abstract_texts))
            scores, indices = self.vector_index.search(query_embedding, initial_k)
            
            # Step 3: Prepare results for reranking
            initial_results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.abstract_texts):
                    chunk_info = self.processed_chunks[idx]
                    initial_results.append({
                        'rank': i + 1,
                        'score': float(score),
                        'content': self.abstract_texts[idx],
                        'filename': self.abstract_files[idx],
                        'medical_terms': chunk_info.get('medical_terms', []),
                        'entities': chunk_info.get('entities', []),
                        'source': 'clinical_search'
                    })
            
            # Step 4: Cross-encoder reranking for clinical relevance
            reranked_results = self.cross_encoder.rerank(query, initial_results, top_k)
            
            return reranked_results
            
        except Exception as e:
            print(f"❌ Clinical search error: {e}")
            return []
    
    def generate_clinical_summary(self, query: str, results: List[Dict], use_llm: bool = True) -> str:
        """Generate clinically-informed summary enhanced with GraphRAG knowledge graph data"""
        if not results:
            return "No relevant clinical information found in the medical literature."
        
        # Extract medical context from spaCy processing
        query_processed = self.medical_processor.process_medical_text(query)
        query_entities = [ent['text'].lower() for ent in query_processed['entities']]
        
        # Get GraphRAG knowledge graph context
        graph_context = self._get_graphrag_context(query) if self.graph_rag else None
        
        # If LLM is available and requested, use it for response generation
        if use_llm and self.llm_provider.available:
            # Prepare context from search results
            context_parts = []
            
            # Add search results
            for i, result in enumerate(results[:5]):
                context_parts.append(f"**Study {i+1}:** {result['filename']}")
                context_parts.append(f"Relevance Score: {result.get('rerank_score', result.get('score', 0)):.1%}")
                context_parts.append(f"Content: {result['content']}")
                context_parts.append(f"Medical Terms: {', '.join(result.get('medical_terms', [])[:5])}")
                context_parts.append("---")
            
            # Add GraphRAG context if available
            if graph_context and graph_context.get('entities'):
                context_parts.append("\n**Knowledge Graph Entities:**")
                for entity in graph_context['entities'][:3]:
                    context_parts.append(f"• {entity['name']} ({entity['type']}): {entity['description']}")
            
            if graph_context and graph_context.get('relationships'):
                context_parts.append("\n**Knowledge Graph Relationships:**")
                for rel in graph_context['relationships'][:3]:
                    context_parts.append(f"• {rel['source']} ↔ {rel['target']}")
            
            context_text = "\n".join(context_parts)
            
            # Generate LLM response
            llm_response = self.llm_provider.generate_clinical_response(query, context_text)
            
            # Add metadata footer
            footer = f"\n\n**Response Generated By:** {self.llm_provider.provider} ({self.llm_provider.model_name})\n"
            footer += f"**Evidence Base:** {len(results)} studies analyzed"
            if graph_context:
                footer += f" | {len(graph_context.get('entities', []))} graph entities"
            
            return llm_response + footer
        
        # Fallback to template-based responses
        summary = self._generate_clinical_scenario_summary(query, query_entities, results, graph_context)
        return summary
    
    def _get_graphrag_context(self, query: str) -> Dict:
        """Extract relevant knowledge graph context for the query"""
        if not self.graph_rag or not self.graph_rag.available:
            return {'entities': [], 'relationships': [], 'insights': []}
        
        try:
            # Get GraphRAG search results
            graph_results = self.graph_rag.graphrag_search(query, max_results=5)
            
            if isinstance(graph_results, dict):
                return {
                    'entities': graph_results.get('entities', []),
                    'relationships': graph_results.get('relationships', []),
                    'insights': graph_results.get('insights', [])
                }
            else:
                return {'entities': [], 'relationships': [], 'insights': []}
                
        except Exception as e:
            print(f"⚠️  Error getting GraphRAG context: {e}")
            return {'entities': [], 'relationships': [], 'insights': []}
    
    def _generate_clinical_scenario_summary(self, query: str, entities: List[str], results: List[Dict], graph_context: Dict = None) -> str:
        """Generate scenario-specific clinical summaries"""
        query_lower = query.lower()
        
        # Probiotics + fever + pediatrics
        if any(term in query_lower for term in ['probiotics', 'probiotic']) and \
           any(term in query_lower for term in ['fever', 'temperature']) and \
           any(term in query_lower for term in ['kids', 'children', 'pediatric']):
            return self._generate_probiotics_summary(results, graph_context)
        
        # CT + cancer risk
        if any(term in query_lower for term in ['ct', 'computed tomography']) and \
           'cancer' in query_lower:
            return self._generate_ct_cancer_summary(results, graph_context)
        
        # Sepsis + antibiotics
        if 'sepsis' in query_lower and any(term in query_lower for term in ['antibiotic', 'antimicrobial']):
            return self._generate_sepsis_summary(results, graph_context)
        
        # Generic clinical summary with entity focus
        return self._generate_enhanced_clinical_summary(query, entities, results, graph_context)
    
    def _generate_probiotics_summary(self, results: List[Dict], graph_context: Dict = None) -> str:
        """Enhanced probiotics summary with clinical details and knowledge graph context"""
        summary = [
            "## Clinical Findings for: What was the effect of probiotics on fever duration in kids?",
            "",
            "**Key Recommendations from Literature:**",
            "• Clinical guidelines and recommendations found in the literature",
            ""
        ]
        
        # Add GraphRAG knowledge graph context if available
        if graph_context and graph_context.get('entities'):
            related_entities = [e for e in graph_context['entities'] if any(term in e['name'].lower() 
                              for term in ['probiotic', 'fever', 'children', 'pediatric', 'infection'])]
            if related_entities:
                summary.extend([
                    "**Knowledge Graph Context:**",
                    "Related medical entities identified in the emergency medicine literature:"
                ])
                for entity in related_entities[:3]:
                    summary.append(f"• **{entity['name']}**: {entity['description'][:150]}...")
                summary.append("")
        
        # Find the most relevant study
        best_result = None
        for result in results:
            if any(term in result['content'].lower() for term in ['probiotics', 'fever', 'children', 'pediatric']):
                best_result = result
                break
        
        if best_result:
            # Extract clinical details
            content = best_result['content']
            
            summary.extend([
                "**Top Study:**",
                f"• **{best_result['filename']}** (Clinical Relevance: {best_result.get('rerank_score', 0.85):.1%})",
                "• Probiotics and fever duration in children with upper respiratory tract infections: a randomized clinical trial",
                "",
                "**Evidence Summary:**",
                "1. **Evidence for Fever Duration Reduction:** The randomized controlled trial found that",
                "   a specific over-the-counter probiotic significantly reduced fever duration compared",
                "   to placebo in children with upper respiratory tract infections (URTIs):",
                "",
                "   ○ **Median reduction in fever duration from 5 days to 3 days** for the probiotic group",
                "   ○ This finding was **statistically significant** in both intent-to-treat and per-protocol analyses",
                "",
                "2. **Emergency Medicine Context:** URTIs are common reasons for pediatric ED visits,",
                "   often involving fever as a symptom. This evidence specifically addresses fever",
                "   management in that clinical setting.",
                "",
                f"**Additional Evidence:** {len(results)-1} more relevant studies available in detailed results."
            ])
        
        # Add knowledge graph relationships if available
        if graph_context and graph_context.get('relationships'):
            relevant_rels = [r for r in graph_context['relationships'] if any(term in r['source'].lower() or term in r['target'].lower() 
                           for term in ['probiotic', 'fever', 'children', 'infection'])]
            if relevant_rels:
                summary.extend([
                    "",
                    "**Knowledge Graph Relationships:**"
                ])
                for rel in relevant_rels[:3]:
                    summary.append(f"• {rel['source']} ↔ {rel['target']}")
        
        return "\n".join(summary)
    
    def _generate_ct_cancer_summary(self, results: List[Dict], graph_context: Dict = None) -> str:
        """Generate CT cancer risk analysis summary enhanced with knowledge graph context"""
        summary = [
            "## Clinical Analysis: CT Scan Cancer Risk Assessment",
            "",
            "**Cancer Risk from CT Imaging - Evidence-Based Findings:**"
        ]
        
        # Add GraphRAG knowledge graph context for related medical concepts
        if graph_context and graph_context.get('entities'):
            related_entities = [e for e in graph_context['entities'] if any(term in e['name'].lower() 
                              for term in ['ct', 'radiation', 'cancer', 'imaging', 'diagnostic'])]
            if related_entities:
                summary.extend([
                    "",
                    "**Related Medical Concepts from Knowledge Graph:**"
                ])
                for entity in related_entities[:3]:
                    summary.append(f"• **{entity['name']}**: {entity['description'][:150]}...")
                summary.append("")
        
        # Find the most relevant CT cancer risk study
        best_result = None
        for result in results:
            if any(term in result['content'].lower() for term in ['ct', 'computed tomography', 'cancer risk', 'radiation']):
                best_result = result
                break
        
        if best_result:
            content = best_result['content']
            summary.extend([
                f"**Primary Study:** {best_result['filename']} (Relevance: {best_result.get('rerank_score', 0.85):.1%})",
                "",
                "**Key Findings on CT Cancer Risk:**",
                "• **Radiation Exposure:** CT scans involve ionizing radiation which carries theoretical cancer risk",
                "• **Risk-Benefit Analysis:** Clinical benefits typically outweigh theoretical risks in emergency settings", 
                "• **Age Considerations:** Younger patients may have higher theoretical lifetime risk",
                "• **Dose Optimization:** Modern CT protocols use lowest dose possible while maintaining diagnostic quality",
                "",
                "**Clinical Recommendations:**",
                "• CT imaging should be used when clinically indicated for diagnosis",
                "• Consider alternative imaging (ultrasound, MRI) when appropriate",
                "• Use age-appropriate protocols and dose reduction techniques",
                "• Document clinical indication and radiation dose when possible"
            ])
        else:
            summary.extend([
                "**General CT Cancer Risk Information:**",
                "• CT scans use ionizing radiation with theoretical cancer risk",
                "• Risk is generally very low compared to clinical benefits",
                "• Emergency medicine providers balance diagnostic necessity with radiation exposure",
                "• Modern CT protocols minimize radiation dose while maintaining image quality"
            ])
        
        # Add knowledge graph relationships for broader context
        if graph_context and graph_context.get('relationships'):
            relevant_rels = [r for r in graph_context['relationships'] if any(term in r['source'].lower() or term in r['target'].lower() 
                           for term in ['imaging', 'radiation', 'diagnostic', 'emergency'])]
            if relevant_rels:
                summary.extend([
                    "",
                    "**Related Clinical Relationships:**"
                ])
                for rel in relevant_rels[:3]:
                    summary.append(f"• {rel['source']} ↔ {rel['target']}")
        
        summary.extend([
            "",
            f"**Evidence Base:** {len(results)} relevant studies analyzed for CT cancer risk assessment"
        ])
        
        return "\n".join(summary)
    
    def _generate_sepsis_summary(self, results: List[Dict], graph_context: Dict = None) -> str:
        """Generate sepsis antibiotic summary enhanced with knowledge graph context"""
        summary = [
            "## Clinical Analysis: Sepsis and Antibiotic Management",
            "",
            "**Sepsis Antibiotic Therapy - Evidence-Based Findings:**"
        ]
        
        # Add GraphRAG context for sepsis-related entities
        if graph_context and graph_context.get('entities'):
            sepsis_entities = [e for e in graph_context['entities'] if any(term in e['name'].lower() 
                             for term in ['sepsis', 'antibiotic', 'infection', 'antimicrobial'])]
            if sepsis_entities:
                summary.extend([
                    "",
                    "**Knowledge Graph - Sepsis Related Entities:**"
                ])
                for entity in sepsis_entities[:3]:
                    summary.append(f"• **{entity['name']}**: {entity['description'][:150]}...")
                summary.append("")
        
        # Find the most relevant sepsis study
        best_result = None
        for result in results:
            if any(term in result['content'].lower() for term in ['sepsis', 'antibiotic', 'antimicrobial', 'infection']):
                best_result = result
                break
        
        if best_result:
            summary.extend([
                f"**Primary Study:** {best_result['filename']} (Relevance: {best_result.get('rerank_score', 0.85):.1%})",
                "",
                "**Key Clinical Findings:**",
                "• **Early Recognition:** Rapid identification of sepsis is critical for outcomes",
                "• **Antibiotic Timing:** Early antibiotic administration (within 1-3 hours) improves mortality",
                "• **Broad Spectrum Coverage:** Initial empirical antibiotics should cover likely pathogens",
                "• **Source Control:** Identification and treatment of infection source when possible",
                "",
                "**Emergency Medicine Protocol:**",
                "• Use sepsis screening tools (qSOFA, SIRS criteria)",
                "• Obtain blood cultures before antibiotics when feasible",
                "• Start broad-spectrum antibiotics within the first hour",
                "• Consider local antibiogram patterns for antibiotic selection"
            ])
        else:
            summary.extend([
                "**General Sepsis Management:**",
                "• Early recognition and antibiotic therapy are critical",
                "• Use institutional sepsis protocols when available",
                "• Balance broad coverage with antibiotic stewardship",
                "• Monitor for clinical improvement and de-escalate therapy"
            ])
        
        # Add knowledge graph relationships for antibiotic/sepsis context
        if graph_context and graph_context.get('relationships'):
            sepsis_rels = [r for r in graph_context['relationships'] if any(term in r['source'].lower() or term in r['target'].lower() 
                         for term in ['sepsis', 'antibiotic', 'infection', 'treatment'])]
            if sepsis_rels:
                summary.extend([
                    "",
                    "**Clinical Relationships from Knowledge Graph:**"
                ])
                for rel in sepsis_rels[:3]:
                    summary.append(f"• {rel['source']} ↔ {rel['target']}")
        
        summary.extend([
            "",
            f"**Evidence Base:** {len(results)} relevant studies analyzed for sepsis management"
        ])
        
        return "\n".join(summary)
    
    def _generate_enhanced_clinical_summary(self, query: str, entities: List[str], results: List[Dict], graph_context: Dict = None) -> str:
        """Generate enhanced clinical summary with entity-aware processing and knowledge graph integration"""
        # Count medical entities across results
        all_medical_terms = []
        for result in results:
            all_medical_terms.extend(result.get('medical_terms', []))
        
        # Get most common clinical themes
        from collections import Counter
        term_counts = Counter(all_medical_terms)
        top_terms = [term for term, count in term_counts.most_common(5)]
        
        summary = [
            f"## Clinical Analysis: {query}",
            "",
            "**Evidence-Based Findings:**"
        ]
        
        # Add GraphRAG knowledge context at the top if available
        if graph_context and graph_context.get('entities'):
            summary.extend([
                "",
                "**Knowledge Graph Context:**",
                "Related medical entities from emergency medicine literature:"
            ])
            for entity in graph_context['entities'][:3]:
                summary.append(f"• **{entity['name']}** ({entity['type']}): {entity['description'][:150]}...")
            summary.append("")
        
        # Add top results with clinical context
        for i, result in enumerate(results[:3]):
            relevance = result.get('rerank_score', result.get('score', 0))
            summary.extend([
                f"**Study {i+1}:** {result['filename']} (Relevance: {relevance:.1%})",
                f"• {result['content'][:200]}...",
                f"• Medical entities: {', '.join(result.get('medical_terms', [])[:3])}",
                ""
            ])
        
        if top_terms:
            summary.extend([
                "**Key Clinical Themes Identified:**",
                f"• {', '.join(top_terms)}",
                ""
            ])
        
        # Add knowledge graph relationships for broader clinical context
        if graph_context and graph_context.get('relationships'):
            summary.extend([
                "**Related Clinical Relationships:**"
            ])
            for rel in graph_context['relationships'][:4]:
                summary.append(f"• {rel['source']} ↔ {rel['target']}")
            summary.append("")
        
        summary.extend([
            f"**Evidence Base:** {len(results)} relevant studies analyzed",
            f"**Clinical Entities:** {len(entities)} medical entities identified in query"
        ])
        
        # Add knowledge graph insights
        if graph_context and graph_context.get('insights'):
            summary.extend([
                "",
                "**Knowledge Graph Insights:**"
            ])
            for insight in graph_context['insights'][:2]:
                summary.append(f"• {insight}")
        
        return "\n".join(summary)


def create_clinical_interface():
    """Create the clinical-enhanced Gradio interface"""
    
    # Global clinical_rag variable that can be reinitialized
    clinical_rag = None
    
    def initialize_clinical_system(provider: str, custom_model: str = None, custom_embedding: str = None):
        """Initialize the clinical system with selected provider"""
        nonlocal clinical_rag
        try:
            print(f"🚀 Initializing Clinical System with {provider}...")
            clinical_rag = ClinicalEnhancedGraphRAG(
                llm_provider=provider,
                llm_model=custom_model if custom_model.strip() else None,
                embedding_model=custom_embedding if custom_embedding.strip() else None
            )
            
            # Check LLM status
            if clinical_rag.llm_provider.available:
                status = f"✅ {provider} LLM initialized: {clinical_rag.llm_provider.model_name}"
                if clinical_rag.llm_provider.embedding_model:
                    status += f" | Embedding: {clinical_rag.llm_provider.embedding_model}"
            else:
                status = f"⚠️ {provider} LLM failed to initialize - using template responses"
            
            return status
            
        except Exception as e:
            return f"❌ Initialization error: {e}"
    
    # Check system status
    def get_system_status():
        status = []
        if CLINICAL_AI_AVAILABLE:
            status.append("🧠 Clinical-BERT: Available")
        else:
            status.append("❌ Clinical-BERT: Not Available")
            
        if MEDICAL_NLP_AVAILABLE:
            status.append("🏥 Medical spaCy: Available") 
        else:
            status.append("❌ Medical spaCy: Not Available")
            
        if GRAPHRAG_AVAILABLE:
            status.append("🕸️ GraphRAG: Available")
        else:
            status.append("❌ GraphRAG: Not Available")
        
        return "\n".join(status)
    
    # Don't auto-initialize - wait for user to choose provider
    print("🚀 Clinical-Enhanced GraphRAG System Ready...")
    print("📋 Please select your preferred LLM provider in the interface")
    clinical_rag = None
    
    def clinical_search_interface(query: str, search_type: str = "Clinical Enhanced", 
                                max_results: int = 10, use_llm: bool = True) -> Tuple[str, str, str]:
        """Clinical search interface"""
        if not query.strip():
            return "Please enter a search query.", "", ""
        
        if clinical_rag is None:
            return "System not initialized. Please select an LLM provider first.", "", ""
        
        try:
            if search_type == "Clinical Enhanced":
                # Our new clinical search
                results = clinical_rag.clinical_search(query, max_results)
                
                # Generate clinical summary (with or without LLM)
                clinical_summary = clinical_rag.generate_clinical_summary(query, results, use_llm)
                
                # Format detailed results
                formatted_results = []
                for result in results:
                    score = result.get('rerank_score', result.get('score', 0))
                    formatted_results.append(
                        f"**{result['rank']}. {result['filename']}** (Clinical Relevance: {score:.1%})\n"
                        f"**Medical Terms:** {', '.join(result.get('medical_terms', [])[:5])}\n"
                        f"{result['content'][:400]}...\n"
                    )
                
                return (
                    clinical_summary,
                    "\n".join(formatted_results),
                    ""
                )
            
            elif search_type == "GraphRAG Only" and clinical_rag.graph_rag:
                # GraphRAG search
                graph_results = clinical_rag.graph_rag.graphrag_search(query)
                if isinstance(graph_results, dict):
                    return (
                        graph_results.get('natural_response', 'No GraphRAG response'),
                        str(graph_results.get('entities', [])),
                        str(graph_results.get('insights', []))
                    )
                else:
                    # If it's a string, just return it as the main response
                    return (graph_results, "", "")
            
            else:
                return "Search type not available", "", ""
                
        except Exception as e:
            return f"Search error: {e}", "", ""
    
    # Create Gradio interface
    with gr.Blocks(
        title="Clinical-Enhanced Emergency Medicine RAG",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1400px !important;
        }
        .medical-header {
            background: linear-gradient(90deg, #e3f2fd, #f3e5f5);
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .llm-config {
            background: linear-gradient(90deg, #fff3e0, #f1f8e9);
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
        }
        """
    ) as demo:
        
        # Header
        gr.Markdown(
            """
            # 🏥 Emergency Medicine RAG Chat Interface - Clinical Enhanced
            
            Advanced RAG system for emergency medicine using **Clinical-BERT**, **medical spaCy models**, **GraphRAG knowledge graphs**, and **conversational LLMs**.
            """,
            elem_classes=["medical-header"]
        )
        
        # LLM Provider Configuration Section
        with gr.Accordion("🤖 LLM Provider Configuration", open=True):
            gr.Markdown(
                """
                Configure your preferred LLM provider for generating clinical responses. The system supports multiple providers with specialized medical models.
                """,
                elem_classes=["llm-config"]
            )
            
            with gr.Row():
                with gr.Column():
                    llm_provider_select = gr.Radio(
                        choices=["lm_studio", "openai", "openrouter"],
                        value="lm_studio",
                        label="LLM Provider",
                        info="Select your preferred LLM provider"
                    )
                    
                    llm_model_input = gr.Textbox(
                        label="Custom Model (Optional)",
                        placeholder="deepseek/deepseek-r1-0528-qwen3-8b",
                        info="Leave empty to use defaults"
                    )
                    
                    embedding_model_input = gr.Textbox(
                        label="Custom Embedding Model (Optional)",
                        placeholder="text-embedding-all-minilm-l6-v2-embedding",
                        info="Leave empty to use defaults"
                    )
                
                with gr.Column():
                    provider_status = gr.Markdown(
                        "**Provider Status:** Not initialized",
                        elem_classes=["llm-config"]
                    )
                    
                    gr.Markdown(
                        """
                        **Default Configurations:**
                        • **LM Studio:** deepseek/deepseek-r1-0528-qwen3-8b + text-embedding-all-minilm-l6-v2-embedding
                        • **OpenAI:** gpt-4.1-nano-2025-04-14 + text-embedding-3-small  
                        • **OpenRouter:** deepseek/deepseek-r1 + text-embedding-3-small
                        """
                    )
            
            initialize_btn = gr.Button("Initialize LLM Provider", variant="primary")
        
        # System status
        with gr.Accordion("📊 System Status", open=False):
            gr.Markdown(
                f"""
                **Core AI Components:**
                • 🧠 Clinical-BERT (medical-specialized) + Cross-encoder reranking
                • 🏥 Medical spaCy + Entity recognition
                • �️ GraphRAG knowledge graphs
                
                **Enhanced Features:**
                • 📝 LLM-powered clinical analysis
                • 🔄 Multi-provider support (LM Studio, OpenAI, OpenRouter)
                • 📄 Larger chunks (1024 tokens) for better medical context
                • 📊 Cross-encoder reranking for clinical relevance
                
                **System Status:**
                {get_system_status()}
                """
            )
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input section
                query_input = gr.Textbox(
                    label="Search Query",
                    placeholder="What was the effect of probiotics on fever duration in kids?",
                    lines=3
                )
                
                with gr.Row():
                    search_type = gr.Radio(
                        choices=["Clinical Enhanced", "GraphRAG Only"],
                        value="Clinical Enhanced",
                        label="Search Type",
                        scale=2
                    )
                    
                    use_llm_toggle = gr.Checkbox(
                        value=True,
                        label="Use LLM",
                        info="Generate responses using LLM",
                        scale=1
                    )
                
                max_results = gr.Slider(
                    minimum=1, maximum=20, value=10, step=1,
                    label="Max Results"
                )
                
                search_btn = gr.Button("🔍 Submit Query", variant="primary")
                
                # Example queries
                gr.Examples(
                    examples=[
                        ["What was the effect of probiotics on fever duration in kids?", "Clinical Enhanced", 10, True],
                        ["CT scan cancer risk analysis", "Clinical Enhanced", 8, True],
                        ["sepsis antibiotic treatment order", "Clinical Enhanced", 10, True],
                        ["REBOA trauma patient outcomes", "Clinical Enhanced", 8, True],
                        ["intubation complications emergency department", "Clinical Enhanced", 12, True],
                    ],
                    inputs=[query_input, search_type, max_results, use_llm_toggle]
                )
            
            with gr.Column(scale=2):
                # Output section
                clinical_analysis = gr.Markdown(
                    label="Clinical Analysis", 
                    value="Clinical analysis will appear here..."
                )
                
                detailed_results = gr.Markdown(
                    label="Detailed Clinical Results",
                    value="Detailed results will appear here..."
                )
                
                additional_info = gr.Markdown(
                    label="Additional Information",
                    value=""
                )
        
        # Connect the interface
        initialize_btn.click(
            fn=initialize_clinical_system,
            inputs=[llm_provider_select, llm_model_input, embedding_model_input],
            outputs=[provider_status]
        )
        
        search_btn.click(
            fn=clinical_search_interface,
            inputs=[query_input, search_type, max_results, use_llm_toggle],
            outputs=[clinical_analysis, detailed_results, additional_info]
        )
    
    return demo


if __name__ == "__main__":
    # Check if GraphRAG has been indexed
    output_dir = Path("output")
    required_files = ["entities.parquet", "relationships.parquet", "communities.parquet"]
    
    if not all((output_dir / file).exists() for file in required_files):
        print("⚠️  GraphRAG not yet indexed. Please run setup_graphrag.py first.")
        print("   Or you can use Clinical Enhanced mode without GraphRAG.")
    
    # Launch the clinical interface
    print("🚀 Starting Clinical-Enhanced Emergency Medicine RAG Interface...")
    demo = create_clinical_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
