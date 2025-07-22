#!/bin/bash
# Quick status check for GraphRAG

echo "🏥 GraphRAG Status Check"
echo "========================"

# Check if process is running
if pgrep -f "graphrag index" > /dev/null; then
    echo "✅ GraphRAG indexing is running"
    echo "📊 Process details:"
    ps aux | grep "graphrag index" | grep -v grep
    echo
else
    echo "❌ GraphRAG indexing is not running"
fi

# Check output files
echo "📁 Output Status:"
echo "Documents: $([ -f output/documents.parquet ] && echo '✅' || echo '⏳')"
echo "Text Units: $([ -f output/text_units.parquet ] && echo '✅' || echo '⏳')"
echo "Artifacts directory: $([ -d output/artifacts ] && echo '✅' || echo '⏳')"

if [ -d "output/artifacts" ]; then
    echo "Entities: $([ -f output/artifacts/create_final_entities.parquet ] && echo '✅' || echo '⏳')"
    echo "Relationships: $([ -f output/artifacts/create_final_relationships.parquet ] && echo '✅' || echo '⏳')"
    echo "Communities: $([ -f output/artifacts/create_final_communities.parquet ] && echo '✅' || echo '⏳')"
    echo "Reports: $([ -f output/artifacts/create_final_community_reports.parquet ] && echo '✅' || echo '⏳')"
fi

echo
echo "💡 Commands:"
echo "Monitor progress: python monitor_progress.py"
echo "View logs: tail -f output/logs.txt"
echo "Start interface: ./start_interface.sh"
