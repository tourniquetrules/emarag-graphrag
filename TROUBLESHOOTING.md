# 🔧 GraphRAG Enhanced Emergency Medicine RAG - Troubleshooting Guide

## 🚀 Quick Fix Commands

If you encounter any issues, run these commands in order:

```bash
# 1. Validate and auto-fix configuration
python validate_config.py

# 2. Reset environment if needed
python setup_environment.py

# 3. Start with the reliable script
./start_system.sh
```

## 🛠️ Common Issues and Solutions

### 1. "No module named 'graphrag'" or Import Errors

**Problem**: Python packages not installed or virtual environment not activated

**Solutions**:
```bash
# Ensure virtual environment is activated
source graphrag-env/bin/activate  # Linux/Mac
# OR
graphrag-env\Scripts\activate     # Windows

# Reinstall requirements
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. "OPENAI_API_KEY not found" Error

**Problem**: Environment variables not configured

**Solutions**:
```bash
# Check if .env file exists
ls -la .env

# If missing, copy from example
cp .env.example .env

# Edit .env and add your API key
nano .env  # Replace 'your_openai_api_key_here' with actual key
```

### 3. "GraphRAG indexing failed" or Configuration Errors

**Problem**: settings.yaml has incorrect configuration

**Solutions**:
```bash
# Auto-fix common configuration issues
python validate_config.py

# Manual fixes in settings.yaml:
# - Change file_pattern from ".*\.csv$" to ".*\.txt$"
# - Change file_type from "csv" to "text"  
# - Add "enabled: false" under claim_extraction section
```

### 4. "Port 7860 already in use" or Interface Won't Start

**Problem**: Port conflicts or stuck processes

**Solutions**:
```bash
# Kill existing processes
pkill -f "integrated_rag.py"
pkill -f "python.*7860"

# Check what's using the port
lsof -i :7860

# Use alternative port (edit integrated_rag.py line ~801)
# Change server_port=7860 to server_port=7861
```

### 5. "No text files found" or Empty Input Directory

**Problem**: Missing input documents

**Solutions**:
```bash
# Check input directory
ls -la input/

# Copy from original emarag system if available
cp ../emarag/abstracts/*.txt input/

# Or manually add .txt files to input/ directory
```

### 6. Virtual Environment Issues

**Problem**: Virtual environment corrupted or wrong Python version

**Solutions**:
```bash
# Remove and recreate virtual environment
rm -rf graphrag-env
python3 -m venv graphrag-env
source graphrag-env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 7. "Permission denied" or Script Won't Execute

**Problem**: Script permissions not set

**Solutions**:
```bash
# Make scripts executable
chmod +x setup_environment.py
chmod +x validate_config.py
chmod +x start_system.sh
```

### 8. Gradio Interface Not Loading

**Problem**: Interface starts but web page won't load

**Solutions**:
```bash
# Check if process is running
ps aux | grep integrated_rag

# Check port status
ss -tlnp | grep :7860

# Try accessing directly
curl http://localhost:7860

# Check firewall (if on remote server)
ufw allow 7860  # Ubuntu
```

### 9. GraphRAG Indexing Errors

**Problem**: Knowledge graph construction fails

**Solutions**:
```bash
# Run with minimal configuration
# Set claim_extraction.enabled: false in settings.yaml

# Check file formats in input/
file input/*.txt  # Should show "text" files

# Reduce concurrent requests in settings.yaml
# Set concurrent_requests: 5 (instead of 25)
```

### 10. Memory or Performance Issues

**Problem**: System runs out of memory or is slow

**Solutions**:
```bash
# Monitor memory usage
htop

# Reduce batch sizes in settings.yaml:
# embeddings.batch_size: 8 (instead of 16)
# parallelization.num_threads: 10 (instead of 50)

# Use smaller model
# Change model: "gpt-4o-mini" instead of "gpt-4-turbo"
```

## 📊 System Status Checks

### Health Check Script
```bash
#!/bin/bash
echo "🏥 GraphRAG System Health Check"
echo "================================"

# Check Python version
python --version

# Check virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment active: $VIRTUAL_ENV"
else
    echo "❌ Virtual environment not active"
fi

# Check key files
for file in integrated_rag.py settings.yaml .env; do
    if [[ -f "$file" ]]; then
        echo "✅ Found: $file"
    else
        echo "❌ Missing: $file"
    fi
done

# Check input files
input_count=$(find input -name "*.txt" 2>/dev/null | wc -l)
echo "📄 Input files: $input_count"

# Check if service is running
if pgrep -f "integrated_rag.py" > /dev/null; then
    echo "✅ Service is running"
    port_info=$(ss -tlnp | grep :7860)
    if [[ -n "$port_info" ]]; then
        echo "✅ Port 7860 is active"
    else
        echo "❌ Port 7860 not found"
    fi
else
    echo "❌ Service not running"
fi
```

Save this as `health_check.sh` and run with `bash health_check.sh`

## 🆘 Emergency Reset

If all else fails, complete reset:

```bash
# 1. Stop everything
pkill -f integrated_rag
pkill -f python

# 2. Clean up
rm -rf graphrag-env
rm -rf __pycache__
rm -rf cache/*
rm -rf output/*

# 3. Fresh start
python setup_environment.py
python validate_config.py
./start_system.sh
```

## 📞 Getting Help

1. **Run the validator first**: `python validate_config.py`
2. **Check logs**: Look for error messages in the terminal output
3. **System info**: Include output of health check script
4. **Environment**: Mention your OS, Python version, and setup method

## 🔍 Debug Mode

To get more detailed error information:

```python
# Add to integrated_rag.py for debugging
import logging
logging.basicConfig(level=logging.DEBUG)

# Or set environment variable
export DEBUG_MODE=true
```

This will provide more detailed error messages to help identify issues.
