#!/usr/bin/env python3
"""
GraphRAG Knowledge Graph Visualizer
This script helps visualize and explore the GraphRAG knowledge graph
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

def analyze_knowledge_graph():
    """Analyze the GraphRAG knowledge graph components"""
    output_dir = Path("output")
    
    print("🔍 GraphRAG Knowledge Graph Analysis")
    print("=" * 50)
    
    # Load entities
    try:
        entities_df = pd.read_parquet(output_dir / "entities.parquet")
        print(f"📊 Entities: {len(entities_df)} total")
        print(f"   Top entity types: {entities_df['type'].value_counts().head().to_dict()}")
        
        # Show sample entities
        print("\n🏷️  Sample Entities:")
        for i, row in entities_df.head(10).iterrows():
            print(f"   • {row['title']} ({row['type']}): {row['description'][:100]}...")
            
    except Exception as e:
        print(f"❌ Error loading entities: {e}")
    
    # Load relationships
    try:
        relationships_df = pd.read_parquet(output_dir / "relationships.parquet")
        print(f"\n🔗 Relationships: {len(relationships_df)} total")
        
        # Show sample relationships
        print("\n🔗 Sample Relationships:")
        for i, row in relationships_df.head(10).iterrows():
            print(f"   • {row['source']} → [{row['description']}] → {row['target']}")
            
    except Exception as e:
        print(f"❌ Error loading relationships: {e}")
    
    # Load communities
    try:
        communities_df = pd.read_parquet(output_dir / "communities.parquet")
        print(f"\n🏘️  Communities: {len(communities_df)} total")
        
        # Show community titles
        print("\n🏘️  Community Topics:")
        for i, row in communities_df.head(10).iterrows():
            print(f"   • Community {row['id']}: {row['title']}")
            
    except Exception as e:
        print(f"❌ Error loading communities: {e}")
    
    # Load documents
    try:
        documents_df = pd.read_parquet(output_dir / "documents.parquet")
        print(f"\n📄 Documents: {len(documents_df)} processed")
        
    except Exception as e:
        print(f"❌ Error loading documents: {e}")
    
    # Show stats
    try:
        with open(output_dir / "stats.json", "r") as f:
            stats = json.load(f)
        print(f"\n📈 Processing Stats:")
        for key, value in stats.items():
            print(f"   • {key}: {value}")
            
    except Exception as e:
        print(f"❌ Error loading stats: {e}")

def search_entities(query):
    """Search for entities matching a query"""
    output_dir = Path("output")
    
    try:
        entities_df = pd.read_parquet(output_dir / "entities.parquet")
        
        # Search in title and description
        mask = (
            entities_df['title'].str.contains(query, case=False, na=False) |
            entities_df['description'].str.contains(query, case=False, na=False)
        )
        
        results = entities_df[mask]
        
        print(f"\n🔍 Search Results for '{query}': {len(results)} matches")
        print("-" * 50)
        
        for i, row in results.head(20).iterrows():
            print(f"\n📌 {row['title']} ({row['type']})")
            print(f"   Description: {row['description']}")
            print(f"   Source IDs: {row['text_unit_ids']}")
            
    except Exception as e:
        print(f"❌ Error searching entities: {e}")

def create_network_visualization():
    """Create a simple network visualization of the knowledge graph"""
    try:
        import networkx as nx
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch
        
        output_dir = Path("output")
        
        # Load data
        entities_df = pd.read_parquet(output_dir / "entities.parquet")
        relationships_df = pd.read_parquet(output_dir / "relationships.parquet")
        
        # Create network graph
        G = nx.Graph()
        
        # Add nodes (entities)
        for _, entity in entities_df.head(50).iterrows():  # Limit for visualization
            G.add_node(entity['title'], 
                      type=entity['type'], 
                      description=entity['description'][:100])
        
        # Add edges (relationships)
        for _, rel in relationships_df.head(100).iterrows():  # Limit for visualization
            if rel['source'] in G.nodes and rel['target'] in G.nodes:
                G.add_edge(rel['source'], rel['target'], 
                          description=rel['description'])
        
        # Create visualization
        plt.figure(figsize=(20, 15))
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Draw nodes by type
        node_colors = {'PERSON': 'lightblue', 'ORGANIZATION': 'lightgreen', 
                      'EVENT': 'orange', 'LOCATION': 'pink', 'OTHER': 'lightgray'}
        
        for node_type, color in node_colors.items():
            nodes = [n for n, d in G.nodes(data=True) if d.get('type') == node_type]
            nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=color, 
                                 node_size=1000, alpha=0.8)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, alpha=0.5, width=0.5)
        
        # Draw labels
        labels = {n: n[:20] + "..." if len(n) > 20 else n for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=8)
        
        plt.title("GraphRAG Knowledge Graph - Emergency Medicine Abstracts", 
                 fontsize=16, fontweight='bold')
        plt.axis('off')
        
        # Create legend
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor=color, markersize=10, label=node_type)
                          for node_type, color in node_colors.items()]
        plt.legend(handles=legend_elements, loc='upper left')
        
        # Save the plot
        plt.tight_layout()
        plt.savefig("knowledge_graph_visualization.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Network visualization saved as 'knowledge_graph_visualization.png'")
        
    except ImportError:
        print("❌ NetworkX and matplotlib required for visualization")
        print("   Install with: pip install networkx matplotlib")
    except Exception as e:
        print(f"❌ Error creating visualization: {e}")

def main():
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "analyze":
            analyze_knowledge_graph()
        elif command == "search" and len(sys.argv) > 2:
            query = " ".join(sys.argv[2:])
            search_entities(query)
        elif command == "visualize":
            create_network_visualization()
        else:
            print("Usage:")
            print("  python visualize_graphrag.py analyze")
            print("  python visualize_graphrag.py search <query>")
            print("  python visualize_graphrag.py visualize")
    else:
        # Default: run analysis
        analyze_knowledge_graph()

if __name__ == "__main__":
    main()
