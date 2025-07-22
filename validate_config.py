#!/usr/bin/env python3
"""
Configuration validator for GraphRAG Enhanced Emergency Medicine RAG
Run this script to check and fix common configuration issues.
"""

import os
import yaml
import json
from pathlib import Path
from dotenv import load_dotenv

def validate_environment():
    """Validate environment configuration"""
    print("🔍 Validating Environment Configuration")
    print("-" * 50)
    
    # Load .env file
    load_dotenv()
    
    issues = []
    warnings = []
    
    # Check OpenAI API Key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        issues.append("OPENAI_API_KEY not found in environment or .env file")
    elif api_key == "your_openai_api_key_here":
        issues.append("OPENAI_API_KEY still has placeholder value")
    elif not api_key.startswith("sk-"):
        warnings.append("OPENAI_API_KEY format looks unusual (should start with 'sk-')")
    else:
        print("✅ OpenAI API key found and formatted correctly")
    
    return issues, warnings

def validate_settings_yaml():
    """Validate settings.yaml configuration"""
    print("\n🔍 Validating settings.yaml Configuration")
    print("-" * 50)
    
    settings_file = Path("settings.yaml")
    issues = []
    warnings = []
    
    if not settings_file.exists():
        issues.append("settings.yaml file not found")
        return issues, warnings
    
    try:
        with open(settings_file, 'r') as f:
            settings = yaml.safe_load(f)
        
        # Check input configuration
        if 'input' in settings:
            input_config = settings['input']
            
            # Check file pattern
            file_pattern = input_config.get('file_pattern', '')
            if file_pattern in ['.*\\.csv$', '.*\\.csv$$']:
                issues.append("input.file_pattern is set to CSV but should be '.*\\.txt$' for text files")
            elif file_pattern == '.*\\.txt$$':
                warnings.append("input.file_pattern has double dollar signs, should be single")
            elif file_pattern == '.*\\.txt$':
                print("✅ input.file_pattern correctly set for text files")
            else:
                warnings.append(f"Unusual file_pattern: {file_pattern}")
            
            # Check file type
            file_type = input_config.get('file_type', '')
            if file_type == 'csv':
                issues.append("input.file_type is set to 'csv' but should be 'text'")
            elif file_type == 'text':
                print("✅ input.file_type correctly set to 'text'")
            else:
                warnings.append(f"input.file_type is '{file_type}', expected 'text'")
        else:
            issues.append("'input' section missing from settings.yaml")
        
        # Check claim extraction
        if 'claim_extraction' in settings:
            claim_config = settings['claim_extraction']
            if 'enabled' not in claim_config:
                issues.append("claim_extraction.enabled field is missing")
            elif claim_config['enabled'] is False:
                print("✅ claim_extraction.enabled correctly set to false")
            else:
                warnings.append("claim_extraction.enabled is true (may cause issues)")
        else:
            warnings.append("claim_extraction section missing from settings.yaml")
        
        # Check LLM model configuration
        if 'llm' in settings and 'model' in settings['llm']:
            model = settings['llm']['model']
            if model == 'gpt-4-turbo-preview':
                warnings.append("Using expensive model 'gpt-4-turbo-preview', consider 'gpt-4o-mini' for cost efficiency")
            elif model == 'gpt-4o-mini':
                print("✅ Using cost-effective model 'gpt-4o-mini'")
        
    except yaml.YAMLError as e:
        issues.append(f"Failed to parse settings.yaml: {e}")
    except Exception as e:
        issues.append(f"Error reading settings.yaml: {e}")
    
    return issues, warnings

def validate_directory_structure():
    """Validate directory structure and files"""
    print("\n🔍 Validating Directory Structure")
    print("-" * 50)
    
    issues = []
    warnings = []
    
    # Check required directories
    required_dirs = ["input", "output", "cache", "prompts"]
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            issues.append(f"Required directory '{dir_name}' does not exist")
        elif not dir_path.is_dir():
            issues.append(f"'{dir_name}' exists but is not a directory")
        else:
            print(f"✅ Directory '{dir_name}' exists")
    
    # Check input files
    input_dir = Path("input")
    if input_dir.exists():
        txt_files = list(input_dir.glob("*.txt"))
        if len(txt_files) == 0:
            warnings.append("No .txt files found in input directory")
        else:
            print(f"✅ Found {len(txt_files)} text files in input directory")
    
    # Check for key files
    key_files = ["integrated_rag.py", "settings.yaml", ".env"]
    for file_name in key_files:
        file_path = Path(file_name)
        if not file_path.exists():
            if file_name == ".env":
                warnings.append(f"'{file_name}' not found - copy from .env.example")
            else:
                issues.append(f"Required file '{file_name}' not found")
        else:
            print(f"✅ File '{file_name}' exists")
    
    return issues, warnings

def fix_common_issues():
    """Attempt to fix common configuration issues"""
    print("\n🔧 Attempting to Fix Common Issues")
    print("-" * 50)
    
    fixes_applied = []
    
    # Fix settings.yaml issues
    settings_file = Path("settings.yaml")
    if settings_file.exists():
        try:
            with open(settings_file, 'r') as f:
                content = f.read()
            
            original_content = content
            
            # Fix file pattern
            if '.*\\.csv$' in content:
                content = content.replace('.*\\.csv$', '.*\\.txt$')
                fixes_applied.append("Fixed file_pattern from CSV to TXT")
            
            if '.*\\.txt$$' in content:
                content = content.replace('.*\\.txt$$', '.*\\.txt$')
                fixes_applied.append("Fixed double dollar signs in file_pattern")
            
            if 'file_type: csv' in content:
                content = content.replace('file_type: csv', 'file_type: text')
                fixes_applied.append("Fixed file_type from csv to text")
            
            # Add enabled field to claim_extraction if missing
            if 'claim_extraction:' in content and 'enabled:' not in content.split('claim_extraction:')[1].split('\n\n')[0]:
                content = content.replace(
                    'claim_extraction:\n  ## llm:',
                    'claim_extraction:\n  enabled: false\n  ## llm:'
                )
                fixes_applied.append("Added 'enabled: false' to claim_extraction")
            
            # Write back if changes made
            if content != original_content:
                with open(settings_file, 'w') as f:
                    f.write(content)
            
        except Exception as e:
            print(f"❌ Error fixing settings.yaml: {e}")
    
    # Create missing directories
    required_dirs = ["input", "output", "cache", "prompts"]
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            dir_path.mkdir(exist_ok=True)
            fixes_applied.append(f"Created directory '{dir_name}'")
    
    return fixes_applied

def main():
    """Main validation and fixing routine"""
    print("🏥 GraphRAG Enhanced Emergency Medicine RAG")
    print("📋 Configuration Validator")
    print("=" * 60)
    
    all_issues = []
    all_warnings = []
    
    # Run validations
    env_issues, env_warnings = validate_environment()
    yaml_issues, yaml_warnings = validate_settings_yaml()
    dir_issues, dir_warnings = validate_directory_structure()
    
    all_issues.extend(env_issues)
    all_issues.extend(yaml_issues)
    all_issues.extend(dir_issues)
    
    all_warnings.extend(env_warnings)
    all_warnings.extend(yaml_warnings)
    all_warnings.extend(dir_warnings)
    
    # Apply fixes
    fixes = fix_common_issues()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 VALIDATION SUMMARY")
    print("=" * 60)
    
    if fixes:
        print("🔧 FIXES APPLIED:")
        for fix in fixes:
            print(f"   ✅ {fix}")
        print()
    
    if all_issues:
        print("❌ CRITICAL ISSUES (must be resolved):")
        for issue in all_issues:
            print(f"   • {issue}")
        print()
    
    if all_warnings:
        print("⚠️  WARNINGS (recommended to address):")
        for warning in all_warnings:
            print(f"   • {warning}")
        print()
    
    if not all_issues and not all_warnings:
        print("🎉 All configuration checks passed!")
        print("✅ System should run without configuration issues")
    elif not all_issues:
        print("✅ No critical issues found")
        print("⚠️  Some warnings present but system should still work")
    else:
        print("❌ Critical issues found that need to be resolved")
        print("   Please fix the issues above and run this validator again")
        return 1
    
    print("\n🚀 Ready to run: python integrated_rag.py")
    return 0

if __name__ == "__main__":
    exit(main())
