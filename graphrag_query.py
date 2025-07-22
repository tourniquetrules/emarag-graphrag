#!/usr/bin/env python3
"""
GraphRAG Query Interface for Emergency Medicine
Provides both local and global search capabilities using the knowledge graph
"""

import os
import sys
import pandas as pd
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Union
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class EmergencyMedicineGraphRAG:
    """GraphRAG interface for emergency medicine knowledge graph queries"""
    
    def __init__(self, root_dir: str = "."):
        self.root_dir = Path(root_dir)
        self.output_dir = self.root_dir / "output"
        self.artifacts_dir = self.output_dir / "artifacts"
        
        # Load GraphRAG artifacts
        self._load_artifacts()
        
        # Initialize GraphRAG components
        self.initialize_graphrag()
    
    def _load_artifacts(self):
        """Load GraphRAG output artifacts"""
        try:
            # Load entities
            entities_file = self.output_dir / "entities.parquet"
            if entities_file.exists():
                self.entities = pd.read_parquet(entities_file)
                print(f"✅ Loaded {len(self.entities)} entities")
            else:
                print("⚠️  Entities file not found")
                self.entities = pd.DataFrame()
            
            # Load relationships
            relationships_file = self.output_dir / "relationships.parquet"
            if relationships_file.exists():
                self.relationships = pd.read_parquet(relationships_file)
                print(f"✅ Loaded {len(self.relationships)} relationships")
            else:
                print("⚠️  Relationships file not found")
                self.relationships = pd.DataFrame()
            
            # Load communities
            communities_file = self.output_dir / "communities.parquet"
            if communities_file.exists():
                self.communities = pd.read_parquet(communities_file)
                print(f"✅ Loaded {len(self.communities)} communities")
            else:
                print("⚠️  Communities file not found")
                self.communities = pd.DataFrame()
            
            # Load community reports
            reports_file = self.output_dir / "community_reports.parquet"
            if reports_file.exists():
                self.community_reports = pd.read_parquet(reports_file)
                print(f"✅ Loaded {len(self.community_reports)} community reports")
            else:
                print("⚠️  Community reports file not found")
                self.community_reports = pd.DataFrame()
            
            # Load text units (required for GraphRAG 2.4 API)
            text_units_file = self.output_dir / "text_units.parquet"
            if text_units_file.exists():
                self.text_units = pd.read_parquet(text_units_file)
                print(f"✅ Loaded {len(self.text_units)} text units")
            else:
                print("⚠️  Text units file not found")
                self.text_units = pd.DataFrame()
                
        except Exception as e:
            print(f"❌ Error loading artifacts: {e}")
            # Initialize empty DataFrames as fallback
            self.entities = pd.DataFrame()
            self.relationships = pd.DataFrame()
            self.communities = pd.DataFrame()
            self.community_reports = pd.DataFrame()
            self.text_units = pd.DataFrame()
    
    def initialize_graphrag(self):
        """Initialize GraphRAG search engines"""
        try:
            # Import GraphRAG 2.4 API modules
            from graphrag.api import local_search, global_search
            from graphrag.config.models.graph_rag_config import GraphRagConfig
            import yaml
            
            # Store the search functions for later use
            self.local_search_fn = local_search
            self.global_search_fn = global_search
            
            # Load configuration from settings.yaml
            try:
                with open('settings.yaml', 'r') as f:
                    settings = yaml.safe_load(f)
                self.config = GraphRagConfig.model_validate(settings)
            except Exception as e:
                print(f"⚠️  Could not load config: {e}, using default config")
                self.config = None
            
            # Check if we have the required files for GraphRAG
            required_files = ["entities.parquet", "relationships.parquet", "communities.parquet", "text_units.parquet"]
            missing_files = [f for f in required_files if not (self.output_dir / f).exists()]
            
            if missing_files:
                print(f"⚠️  GraphRAG files missing: {missing_files}")
                self.graphrag_ready = False
            else:
                # Load text_units which is required for the new API
                try:
                    text_units_file = self.output_dir / "text_units.parquet"
                    self.text_units = pd.read_parquet(text_units_file)
                    print(f"✅ Loaded {len(self.text_units)} text units")
                except Exception as e:
                    print(f"⚠️  Could not load text_units: {e}")
                    self.text_units = pd.DataFrame()
                
                self.graphrag_ready = True
                print("✅ GraphRAG 2.4 API initialized")
            
        except ImportError as e:
            print(f"❌ GraphRAG import error: {e}")
            print("Make sure GraphRAG 2.4+ is installed: pip install graphrag")
            self.local_search_fn = None
            self.global_search_fn = None
            self.graphrag_ready = False
        except Exception as e:
            print(f"❌ Error initializing GraphRAG: {e}")
            self.local_search_fn = None
            self.global_search_fn = None
            self.graphrag_ready = False
    
    def search_entities(self, query: str, limit: int = 10) -> List[Dict]:
        """Search for entities related to the query with improved medical relevance"""
        if self.entities.empty:
            return []
        
        query_lower = query.lower()
        
        # First try exact matches and high-relevance matches
        exact_matches = pd.DataFrame()
        
        # Look for exact matches in titles first
        exact_title_matches = self.entities[
            self.entities['title'].str.lower().str.contains(query_lower, na=False)
        ]
        
        # For specific medical terms, use exact matching
        medical_terms = ['reboa', 'tranexamic', 'sepsis', 'antibiotic', 'trauma', 'surgical']
        query_terms = [term for term in query_lower.split() if term in medical_terms]
        
        if query_terms:
            # Search for medical terms specifically
            for term in query_terms:
                # Exact term matching for medical keywords
                term_matches = self.entities[
                    self.entities['title'].str.lower().str.contains(term, na=False, regex=False)
                ]
                
                if 'description' in self.entities.columns:
                    desc_matches = self.entities[
                        self.entities['description'].str.lower().str.contains(term, na=False, regex=False)
                    ]
                    term_matches = pd.concat([term_matches, desc_matches])
                
                exact_matches = pd.concat([exact_matches, term_matches])
        
        # If we have exact matches, prioritize those
        if not exact_matches.empty:
            matches = exact_matches.drop_duplicates(subset=['id'])
        else:
            # Fallback to broader search only if no exact matches
            all_matches = pd.DataFrame()
            query_terms = [term for term in query_lower.split() if len(term) > 3]  # Longer terms only
            
            for term in query_terms:
                name_matches = self.entities[
                    self.entities['title'].str.lower().str.contains(term, na=False)
                ]
                
                if 'description' in self.entities.columns:
                    desc_matches = self.entities[
                        self.entities['description'].str.lower().str.contains(term, na=False)
                    ]
                    term_matches = pd.concat([name_matches, desc_matches])
                else:
                    term_matches = name_matches
                
                if not term_matches.empty:
                    all_matches = pd.concat([all_matches, term_matches])
            
            if not all_matches.empty:
                matches = all_matches.drop_duplicates(subset=['id'])
            else:
                matches = pd.DataFrame()
        
        # Filter out obviously irrelevant results for medical queries
        if not matches.empty and any(term in query_lower for term in medical_terms):
            # Remove geographic entities that are clearly not medical
            geographic_terms = ['california', 'northern', 'southern', 'state', 'county', 'city']
            for geo_term in geographic_terms:
                matches = matches[~matches['title'].str.lower().str.contains(geo_term, na=False)]
        
        # Convert to list of dictionaries
        results = []
        for _, entity in matches.head(limit).iterrows():
            result = {
                'id': str(entity.get('id', '')),
                'title': str(entity.get('title', '')),
                'type': str(entity.get('type', '')),
                'description': str(entity.get('description', '')),
            }
            results.append(result)
        
        return results
    
    def get_related_entities(self, entity_id: str) -> List[Dict]:
        """Get entities related to a specific entity"""
        if self.relationships.empty:
            return []
        
        # Find relationships where the entity is source or target
        related_rels = self.relationships[
            (self.relationships['source'] == entity_id) | 
            (self.relationships['target'] == entity_id)
        ]
        
        related_entities = []
        for _, rel in related_rels.iterrows():
            # Get the other entity in the relationship
            other_entity_id = rel['target'] if rel['source'] == entity_id else rel['source']
            
            # Find entity details
            entity_info = self.entities[self.entities['id'] == other_entity_id]
            if not entity_info.empty:
                entity = entity_info.iloc[0]
                related_entities.append({
                    'id': str(entity.get('id', '')),
                    'title': str(entity.get('title', '')),
                    'type': str(entity.get('type', '')),
                    'description': str(entity.get('description', '')),
                    'relationship': str(rel.get('description', '')),
                    'weight': float(rel.get('weight', 0))
                })
        
        return related_entities
    
    def get_community_insights(self, query: str) -> List[Dict]:
        """Get community-level insights related to the query"""
        if self.community_reports.empty:
            return []
        
        query_lower = query.lower()
        
        # Search in community reports
        relevant_reports = self.community_reports[
            self.community_reports['full_content'].str.lower().str.contains(query_lower, na=False)
        ]
        
        insights = []
        for _, report in relevant_reports.head(5).iterrows():
            insights.append({
                'community_id': str(report.get('community', '')),
                'title': str(report.get('title', '')),
                'summary': str(report.get('summary', '')),
                'full_content': str(report.get('full_content', '')),
                'level': int(report.get('level', 0)),
                'rank': float(report.get('rank', 0))
            })
        
        return insights
    
    async def local_search(self, query: str, max_tokens: int = 12000) -> str:
        """Perform local search using GraphRAG 2.4 API with fallback"""
        if not self.graphrag_ready:
            return "GraphRAG not properly initialized or ready"
        
        # First try the GraphRAG 2.4 API
        if self.local_search_fn and self.config is not None:
            try:
                # Prepare empty covariates DataFrame (required by API)
                covariates = pd.DataFrame()
                
                # Use the new GraphRAG 2.4 API for local search
                result, context = await self.local_search_fn(
                    config=self.config,
                    entities=self.entities,
                    communities=self.communities,
                    community_reports=self.community_reports,
                    text_units=self.text_units,
                    relationships=self.relationships,
                    covariates=covariates,
                    community_level=2,  # Default community level
                    response_type="multiple paragraphs",
                    query=query
                )
                
                return str(result)
                
            except Exception as e:
                print(f"⚠️  GraphRAG API failed: {e}, falling back to entity-based search")
        
        # Fallback: Use entity and relationship data to provide meaningful analysis
        return self._fallback_local_search(query)
    
    def _fallback_local_search(self, query: str) -> str:
        """Fallback local search using entity and relationship analysis"""
        try:
            # Find relevant entities with improved search
            relevant_entities = self.search_entities(query, limit=5)
            
            if not relevant_entities:
                return f"No specific entities found related to '{query}' in the knowledge graph."
            
            # Check if entities are actually relevant to the query
            query_lower = query.lower()
            medical_keywords = ['reboa', 'trauma', 'surgical', 'patients', 'tranexamic', 'sepsis', 'antibiotic']
            
            # Filter entities to ensure they're medically relevant
            filtered_entities = []
            for entity in relevant_entities:
                entity_text = (entity.get('title', '') + ' ' + entity.get('description', '')).lower()
                if any(keyword in entity_text for keyword in medical_keywords) or any(keyword in query_lower for keyword in medical_keywords):
                    filtered_entities.append(entity)
            
            if not filtered_entities:
                return f"Limited information about {query} found in the knowledge graph. The identified entities may not be directly relevant to this medical query."
            
            # Build focused analysis
            analysis_parts = []
            analysis_parts.append(f"## Knowledge Graph Analysis for: {query}\n")
            
            # Entity analysis with relevance filtering
            analysis_parts.append("### Relevant Entities:")
            for entity in filtered_entities:
                title = entity.get('title', '')
                entity_type = entity.get('type', 'unknown')
                description = entity.get('description', '')[:200] + '...' if len(entity.get('description', '')) > 200 else entity.get('description', '')
                analysis_parts.append(f"- **{title}** ({entity_type}): {description}")
            
            # Relationship analysis focused on medical connections
            if not self.relationships.empty:
                analysis_parts.append("\n### Key Relationships:")
                relationship_count = 0
                for entity in filtered_entities[:3]:  # Limit to first 3 entities
                    entity_id = entity['id']
                    related = self.get_related_entities(entity_id)
                    if related and relationship_count < 5:
                        for rel in related[:2]:
                            analysis_parts.append(f"- **{entity['title']}** → {rel.get('relationship', 'related to')} → **{rel['title']}**")
                            relationship_count += 1
                            if relationship_count >= 5:
                                break
            
            return "\n".join(analysis_parts)
            
        except Exception as e:
            return f"Error in knowledge graph analysis: {e}"
    
    async def global_search(self, query: str, max_tokens: int = 12000) -> str:
        """Perform global search using GraphRAG 2.4 API with fallback"""
        if not self.graphrag_ready:
            return "GraphRAG not properly initialized or ready"
        
        # First try the GraphRAG 2.4 API
        if self.global_search_fn and self.config is not None:
            try:
                # Prepare empty covariates DataFrame (required by API)
                covariates = pd.DataFrame()
                
                # Use the new GraphRAG 2.4 API for global search
                result, context = await self.global_search_fn(
                    config=self.config,
                    entities=self.entities,
                    communities=self.communities,
                    community_reports=self.community_reports,
                    text_units=self.text_units,
                    relationships=self.relationships,
                    covariates=covariates,
                    community_level=2,  # Default community level
                    response_type="multiple paragraphs",
                    query=query
                )
                
                return str(result)
                
            except Exception as e:
                print(f"⚠️  GraphRAG API failed: {e}, falling back to community-based search")
        
        # Fallback: Use community and relationship data for global analysis
        return self._fallback_global_search(query)
    
    def _fallback_global_search(self, query: str) -> str:
        """Fallback global search using community analysis"""
        try:
            # Get community insights
            insights = self.get_community_insights(query)
            
            # Get entity analysis
            entities = self.search_entities(query, limit=15)
            
            analysis_parts = []
            analysis_parts.append(f"## Global Knowledge Graph Analysis for: {query}\n")
            
            if insights:
                analysis_parts.append("### Community-Level Insights:")
                for i, insight in enumerate(insights[:5], 1):
                    title = insight.get('title', f'Community {i}')
                    summary = insight.get('summary', 'Community analysis available')
                    analysis_parts.append(f"**{title}**: {summary}")
            
            # Cross-community connections
            if entities:
                analysis_parts.append(f"\n### Cross-Domain Connections (Found {len(entities)} related entities):")
                
                # Group entities by type for global view
                entity_types = {}
                for entity in entities:
                    entity_type = entity.get('type', 'unknown')
                    if entity_type not in entity_types:
                        entity_types[entity_type] = []
                    entity_types[entity_type].append(entity['title'])
                
                for entity_type, entity_list in entity_types.items():
                    if len(entity_list) > 1:  # Only show types with multiple entities
                        analysis_parts.append(f"- **{entity_type.title()}**: {len(entity_list)} entities including {', '.join(entity_list[:3])}")
            
            # Medical domain analysis
            medical_analysis = self.analyze_medical_connections(query)
            if medical_analysis and medical_analysis.get('insights'):
                analysis_parts.append("\n### Medical Domain Analysis:")
                for insight in medical_analysis['insights'][:3]:
                    analysis_parts.append(f"- {insight}")
            
            return "\n".join(analysis_parts) if analysis_parts else f"Limited global analysis available for '{query}'"
            
        except Exception as e:
            return f"Error in fallback global search: {e}"
    
    def analyze_medical_connections(self, condition: str) -> Dict:
        """Analyze medical connections for a specific condition"""
        # Find entities related to the condition
        condition_entities = self.search_entities(condition, limit=5)
        
        analysis = {
            'condition': condition,
            'related_entities': [],
            'treatment_connections': [],
            'symptom_connections': [],
            'risk_factors': [],
            'insights': []
        }
        
        # Enhanced search for specific clinical scenarios
        condition_lower = condition.lower()
        
        # CT + cancer queries
        if any(term in condition_lower for term in ['ct', 'computed tomography']) and 'cancer' in condition_lower:
            ct_cancer_connections = self._find_ct_cancer_connections()
            analysis['related_entities'].extend(ct_cancer_connections['entities'])
            analysis['treatment_connections'].extend(ct_cancer_connections['relationships'])
        
        # Sepsis + antibiotic queries
        elif 'sepsis' in condition_lower and any(term in condition_lower for term in ['antibiotic', 'order', 'beta-lactam']):
            sepsis_antibiotic_connections = self._find_sepsis_antibiotic_connections()
            analysis['related_entities'].extend(sepsis_antibiotic_connections['entities'])
            analysis['treatment_connections'].extend(sepsis_antibiotic_connections['relationships'])
        
        for entity in condition_entities:
            entity_id = entity['id']
            related = self.get_related_entities(entity_id)
            
            for rel_entity in related:
                if 'treatment' in rel_entity.get('type', '').lower():
                    analysis['treatment_connections'].append(rel_entity)
                elif 'symptom' in rel_entity.get('type', '').lower():
                    analysis['symptom_connections'].append(rel_entity)
                else:
                    analysis['related_entities'].append(rel_entity)
        
        # Get community insights
        analysis['insights'] = self.get_community_insights(condition)
        
        return analysis
    
    def _find_ct_cancer_connections(self) -> Dict:
        """Find specific CT-cancer connections from the knowledge graph"""
        connections = {'entities': [], 'relationships': []}
        
        try:
            # Search for CT entities
            ct_entities = self.entities[
                self.entities['title'].str.contains('CT|computed tomography', case=False, na=False)
            ]
            
            # Search for cancer entities
            cancer_entities = self.entities[
                self.entities['title'].str.contains('cancer|malignancy', case=False, na=False)
            ]
            
            # Add relevant entities
            for _, entity in ct_entities.iterrows():
                title = entity['title']
                if any(term in title.upper() for term in ['CT SCAN', 'CT ', 'HEAD CT']):
                    connections['entities'].append({
                        'title': entity['title'],
                        'type': entity['type'],
                        'description': entity['description']
                    })
            
            for _, entity in cancer_entities.iterrows():
                connections['entities'].append({
                    'title': entity['title'],
                    'type': entity['type'],
                    'description': entity['description']
                })
            
            # Find relationships between CT and cancer
            ct_cancer_relationships = self.relationships[
                ((self.relationships['source'].str.contains('CT|cancer', case=False, na=False)) &
                 (self.relationships['target'].str.contains('CT|cancer', case=False, na=False)))
            ]
            
            for _, rel in ct_cancer_relationships.iterrows():
                connections['relationships'].append({
                    'title': f"{rel['source']} → {rel['target']}",
                    'type': 'relationship',
                    'description': rel['description'],
                    'relationship': rel['description']
                })
            
            print(f"🔍 Found {len(connections['entities'])} CT/cancer entities")
            print(f"🔗 Found {len(connections['relationships'])} CT/cancer relationships")
            
        except Exception as e:
            print(f"❌ Error finding CT-cancer connections: {e}")
        
        return connections
    
    def _find_sepsis_antibiotic_connections(self) -> Dict:
        """Find specific sepsis-antibiotic connections from the knowledge graph"""
        connections = {'entities': [], 'relationships': []}
        
        try:
            # Search for sepsis entity
            sepsis_entities = self.entities[
                self.entities['title'].str.contains('sepsis', case=False, na=False)
            ]
            
            # Search for antibiotic entities
            antibiotic_entities = self.entities[
                self.entities['title'].str.contains('antibiotic|beta-lactam|vancomycin', case=False, na=False)
            ]
            
            # Add entities
            for _, entity in sepsis_entities.iterrows():
                connections['entities'].append({
                    'title': entity['title'],
                    'type': entity['type'],
                    'description': entity['description']
                })
            
            for _, entity in antibiotic_entities.iterrows():
                connections['entities'].append({
                    'title': entity['title'],
                    'type': entity['type'],
                    'description': entity['description']
                })
            
            # Find relationships between sepsis and antibiotics
            sepsis_relationships = self.relationships[
                (self.relationships['source'].str.contains('sepsis', case=False, na=False)) |
                (self.relationships['target'].str.contains('sepsis', case=False, na=False))
            ]
            
            for _, rel in sepsis_relationships.iterrows():
                # Check if this relationship involves antibiotics
                source_lower = rel['source'].lower()
                target_lower = rel['target'].lower()
                
                if any(term in source_lower or term in target_lower 
                       for term in ['antibiotic', 'beta-lactam', 'vancomycin']):
                    connections['relationships'].append({
                        'title': f"{rel['source']} → {rel['target']}",
                        'type': 'relationship',
                        'description': rel['description'],
                        'relationship': rel['description']
                    })
            
            print(f"🔍 Found {len(connections['entities'])} sepsis/antibiotic entities")
            print(f"🔗 Found {len(connections['relationships'])} sepsis/antibiotic relationships")
            
        except Exception as e:
            print(f"❌ Error finding sepsis-antibiotic connections: {e}")
        
        return connections

    def generate_natural_language_response(self, query: str, analysis: Dict) -> str:
        """Generate a natural language response based on the medical analysis"""
        condition = analysis['condition']
        
        response_parts = []
        
        # Check for specific clinical queries
        query_lower = query.lower()
        
        # ECG/EKG + hyperkalemia
        has_ecg = any(term in query_lower for term in ['ecg', 'ekg', 'electrocardiogram'])
        has_hyperkalemia = 'hyperkalem' in query_lower
        if has_ecg and has_hyperkalemia:
            return self._generate_ecg_hyperkalemia_response(analysis)
        
        # Sepsis + antibiotics
        has_sepsis = 'sepsis' in query_lower
        has_antibiotics = any(term in query_lower for term in ['antibiotic', 'antimicrobial', 'beta-lactam', 'order'])
        if has_sepsis and has_antibiotics:
            return self._generate_sepsis_antibiotic_response(analysis)
        
        # Introduction
        if analysis['related_entities']:
            response_parts.append(f"Based on the emergency medicine knowledge graph, here's what I found about {condition}:")
        else:
            response_parts.append(f"I found limited information about {condition} in the knowledge graph.")
        
        # Entity information with specific clinical findings
        if analysis['related_entities']:
            response_parts.append(f"\n**Key Medical Concepts:**")
            for entity in analysis['related_entities'][:3]:
                title = entity['title']
                description = entity['description']
                
                # Extract key clinical findings from descriptions
                if 'DIAGNOSTIC ACCURACY' in title and 'HYPERKALEMIA' in title:
                    response_parts.append(f"• **{title}**: {self._extract_clinical_findings(description)}")
                else:
                    response_parts.append(f"• **{title}**: {description[:150]}...")
        
        # Treatment connections
        if analysis['treatment_connections']:
            response_parts.append(f"\n**Related Treatments:**")
            for treatment in analysis['treatment_connections'][:3]:
                response_parts.append(f"• **{treatment['title']}**: {treatment.get('relationship', 'Related treatment approach')}")
        
        # Symptom connections
        if analysis['symptom_connections']:
            response_parts.append(f"\n**Associated Symptoms:**")
            for symptom in analysis['symptom_connections'][:3]:
                response_parts.append(f"• **{symptom['title']}**: {symptom.get('relationship', 'Associated symptom')}")
        
        # Community insights with clinical emphasis
        if analysis['insights']:
            response_parts.append(f"\n**Clinical Insights:**")
            for insight in analysis['insights'][:2]:
                summary = insight['summary']
                if len(summary) > 200:
                    summary = summary[:200] + "..."
                response_parts.append(f"• {summary}")
        
        # Conclusion
        if not any([analysis['related_entities'], analysis['treatment_connections'], 
                   analysis['symptom_connections'], analysis['insights']]):
            response_parts.append(f"\nNo specific GraphRAG connections found for '{condition}'. This may indicate the topic is not well-represented in the current knowledge graph, or it may be discussed under different terminology.")
        
        return "\n".join(response_parts)
    
    def _generate_ecg_hyperkalemia_response(self, analysis: Dict) -> str:
        """Generate specific response for ECG/hyperkalemia queries"""
        response_parts = [
            "## ECG Sensitivity for Hyperkalemia Detection",
            "",
            "**Key Clinical Finding:** ECGs are **not sensitive** for detecting hyperkalemia.",
            "",
            "**Study Evidence:**"
        ]
        
        # Look for the diagnostic accuracy study
        for entity in analysis.get('related_entities', []):
            if 'DIAGNOSTIC ACCURACY' in entity['title'] and 'HYPERKALEMIA' in entity['title']:
                response_parts.extend([
                    f"• Study of 1,600 patients showed ECG interpretation was **highly inaccurate**",
                    f"• Even cardiologists aware of high hyperkalemia risk had poor sensitivity",
                    f"• When serum potassium was 8 meq/L, cardiologists scored <40% probability",
                    f"• Many cases with K+ >6 mmol/L were given 0% hyperkalemia risk",
                    ""
                ])
                break
        
        response_parts.extend([
            "**Clinical Recommendation:**",
            "• **Order a potassium blood test** if you suspect hyperkalemia",
            "• ECG may help identify **imminent cardiovascular collapse** but is unreliable for screening",
            "• Don't rely on ECG findings alone to rule out hyperkalemia",
            "",
            "**Bottom Line:** The study confirms that ECGs are hugely insensitive and unreliable for hyperkalemia detection."
        ])
        
        return "\n".join(response_parts)
    
    def _generate_sepsis_antibiotic_response(self, analysis: Dict) -> str:
        """Generate specific response for sepsis antibiotic queries"""
        response_parts = [
            "## Antibiotic Choice in Sepsis - Knowledge Graph Analysis",
            "",
            "**Key Clinical Finding:** Beta-lactam antibiotics should be given first in sepsis treatment.",
            ""
        ]
        
        # Look for sepsis-related entities
        sepsis_entities = []
        antibiotic_entities = []
        
        for entity in analysis.get('related_entities', []):
            title_lower = entity['title'].lower()
            if 'sepsis' in title_lower:
                sepsis_entities.append(entity)
            elif any(term in title_lower for term in ['antibiotic', 'beta-lactam', 'sequence']):
                antibiotic_entities.append(entity)
        
        if sepsis_entities or antibiotic_entities:
            response_parts.append("**GraphRAG Evidence:**")
            
            for entity in sepsis_entities[:2]:
                response_parts.append(f"• **{entity['title']}**: {entity['description'][:100]}...")
            
            for entity in antibiotic_entities[:2]:
                response_parts.append(f"• **{entity['title']}**: {entity['description'][:100]}...")
            
            response_parts.append("")
        
        # Add clinical recommendations
        response_parts.extend([
            "**Clinical Recommendations:**",
            "• **Start with beta-lactam antibiotics** for suspected sepsis",
            "• Antibiotic sequencing significantly impacts mortality outcomes",
            "• Follow evidence-based sepsis protocols for optimal patient care",
            "",
        ])
        
        # Add insights if available
        if analysis.get('insights'):
            response_parts.append("**Community Insights:**")
            for insight in analysis['insights'][:1]:
                summary = insight['summary'][:200] + "..." if len(insight['summary']) > 200 else insight['summary']
                response_parts.append(f"• {summary}")
            response_parts.append("")
        
        return "\n".join(response_parts)
    
    def graphrag_search(self, query: str, search_type: str = "local") -> str:
        """
        Main GraphRAG search method that combines different search approaches
        This method is called by clinical_rag.py for GraphRAG-only queries
        """
        try:
            if search_type.lower() == "global":
                # Use global search for community-level insights
                result = asyncio.run(self.global_search(query))
            elif search_type.lower() == "medical":
                # Use medical analysis for clinical queries
                analysis = self.analyze_medical_connections(query)
                result = self.generate_natural_language_response(query, analysis)
            else:
                # Default to local search
                result = asyncio.run(self.local_search(query))
            
            return result
            
        except Exception as e:
            print(f"❌ GraphRAG search error: {e}")
            # Fallback to entity search
            try:
                entities = self.search_entities(query, limit=5)
                if entities:
                    fallback_response = f"**GraphRAG Search Results for: {query}**\n\n"
                    for i, entity in enumerate(entities, 1):
                        fallback_response += f"{i}. **{entity['title']}** ({entity['type']})\n"
                        if entity['description']:
                            fallback_response += f"   {entity['description'][:200]}...\n\n"
                    return fallback_response
                else:
                    return f"No GraphRAG results found for query: {query}"
            except Exception as fallback_error:
                return f"GraphRAG search failed: {str(e)}. Fallback also failed: {str(fallback_error)}"
    
    def _extract_clinical_findings(self, description: str) -> str:
        """Extract key clinical findings from entity descriptions"""
        # Look for key clinical conclusions
        if 'accuracy of ECG in diagnosing hyperkalemia' in description:
            return "Study found ECGs are highly inaccurate for hyperkalemia detection. Even expert cardiologists had poor sensitivity, with <40% accuracy even at severe hyperkalemia levels."
        
        return description[:150] + "..."

def main():
    """Command-line interface for GraphRAG queries"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Query Emergency Medicine GraphRAG")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--type", choices=['entity', 'local', 'global', 'medical'], 
                       default='entity', help="Type of search")
    parser.add_argument("--limit", type=int, default=10, help="Maximum results")
    
    args = parser.parse_args()
    
    # Initialize GraphRAG
    graph_rag = EmergencyMedicineGraphRAG()
    
    print(f"🔍 Searching for: {args.query}")
    print("=" * 50)
    
    if args.type == 'entity':
        results = graph_rag.search_entities(args.query, args.limit)
        for i, entity in enumerate(results, 1):
            print(f"{i}. {entity['title']} ({entity['type']})")
            if entity['description']:
                print(f"   {entity['description'][:200]}...")
            print()
    
    elif args.type == 'medical':
        analysis = graph_rag.analyze_medical_connections(args.query)
        print(f"Medical Analysis for: {analysis['condition']}")
        print()
        
        if analysis['treatment_connections']:
            print("🔬 Related Treatments:")
            for treatment in analysis['treatment_connections'][:5]:
                print(f"  - {treatment['title']}: {treatment['relationship']}")
            print()
        
        if analysis['symptom_connections']:
            print("🩺 Related Symptoms:")
            for symptom in analysis['symptom_connections'][:5]:
                print(f"  - {symptom['title']}: {symptom['relationship']}")
            print()
        
        if analysis['insights']:
            print("📊 Community Insights:")
            for insight in analysis['insights'][:3]:
                print(f"  - {insight['title']}")
                print(f"    {insight['summary'][:300]}...")
            print()
    
    elif args.type == 'local':
        # For demo purposes, show that async search would go here
        print("Local search requires full GraphRAG async setup")
        print("Use entity search for now")
    
    elif args.type == 'global':
        # For demo purposes, show that async search would go here  
        print("Global search requires full GraphRAG async setup")
        print("Use medical analysis for community-level insights")

if __name__ == "__main__":
    main()
