#!/usr/bin/env python3
"""
GraphRAG Progress Monitor
Monitors the progress of GraphRAG indexing and provides status updates
"""

import os
import time
import subprocess
from pathlib import Path
import json

def check_process():
    """Check if GraphRAG process is running"""
    try:
        result = subprocess.run(['pgrep', '-f', 'graphrag index'], 
                              capture_output=True, text=True)
        return len(result.stdout.strip()) > 0
    except:
        return False

def get_output_status():
    """Check what output files have been generated"""
    output_dir = Path("output")
    cache_dir = Path("cache")
    
    status = {
        'documents': (output_dir / "documents.parquet").exists(),
        'text_units': (output_dir / "text_units.parquet").exists(),
        'entities': (output_dir / "artifacts" / "create_final_entities.parquet").exists(),
        'relationships': (output_dir / "artifacts" / "create_final_relationships.parquet").exists(),
        'communities': (output_dir / "artifacts" / "create_final_communities.parquet").exists(),
        'reports': (output_dir / "artifacts" / "create_final_community_reports.parquet").exists()
    }
    
    return status

def get_log_tail():
    """Get the last few lines of the log file"""
    log_file = Path("output/logs.txt")
    if log_file.exists():
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
                return lines[-5:] if len(lines) >= 5 else lines
        except:
            return []
    return []

def monitor_progress():
    """Monitor GraphRAG progress"""
    print("🔍 GraphRAG Progress Monitor")
    print("=" * 50)
    
    while True:
        is_running = check_process()
        status = get_output_status()
        
        print(f"\n⏰ {time.strftime('%H:%M:%S')} - Status Update:")
        print(f"🔄 Process Running: {'✅ Yes' if is_running else '❌ No'}")
        
        print("\n📊 Output Files:")
        for key, exists in status.items():
            icon = "✅" if exists else "⏳"
            print(f"  {icon} {key}")
        
        # Show recent log entries
        log_lines = get_log_tail()
        if log_lines:
            print("\n📝 Recent Log Entries:")
            for line in log_lines:
                print(f"  {line.strip()}")
        
        # Check if completed
        if not is_running and all(status.values()):
            print("\n🎉 GraphRAG indexing completed successfully!")
            print("You can now run the integrated interface:")
            print("  ./start_interface.sh")
            break
        elif not is_running:
            print("\n⚠️  Process stopped but not all files generated")
            print("Check logs for errors or restart indexing")
            break
        
        print("\n" + "="*50)
        time.sleep(30)  # Check every 30 seconds

if __name__ == "__main__":
    try:
        monitor_progress()
    except KeyboardInterrupt:
        print("\n\n👋 Monitoring stopped by user")
    except Exception as e:
        print(f"\n❌ Monitor error: {e}")
