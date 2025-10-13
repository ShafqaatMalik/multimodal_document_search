#!/usr/bin/env python3
"""
Simple test runner for the multimodal search project.
Run this to test core functionality without external dependencies.
"""

import os
import sys
import subprocess
import traceback
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def run_import_tests():
    """Test that all modules can be imported successfully."""
    print("🔍 Testing module imports...")
    
    modules_to_test = [
        "config",
        "utils", 
        "data_preprocessor",
        "main"
    ]
    
    passed = 0
    failed = 0
    
    for module in modules_to_test:
        try:
            __import__(module)
            print(f"  ✅ {module}")
            passed += 1
        except Exception as e:
            print(f"  ❌ {module}: {str(e)[:100]}")
            failed += 1
    
    return passed, failed

def run_config_tests():
    """Test configuration functionality."""
    print("\n⚙️ Testing configuration...")
    
    try:
        import config
        
        # Test basic config loading
        cfg = config.get_config()
        assert isinstance(cfg, dict)
        assert "clip_model" in cfg
        assert "embedding_dim" in cfg
        
        # Test config values are reasonable
        assert cfg["embedding_dim"] > 0
        assert cfg["text_chunk_size"] > 0
        assert cfg["max_text_length"] > 1000
        assert 0 <= cfg["min_similarity"] <= 1
        
        print("  ✅ Configuration tests passed")
        return 1, 0
        
    except Exception as e:
        print(f"  ❌ Configuration test failed: {e}")
        return 0, 1

def run_utility_tests():
    """Test utility functions."""
    print("\n🛠️ Testing utility functions...")
    
    try:
        import utils
        import numpy as np
        
        passed = 0
        failed = 0
        
        # Test embedding parsing
        test_embedding = "[0.1, 0.2, 0.3]"
        result = utils.parse_embedding(test_embedding)
        assert isinstance(result, np.ndarray)
        assert len(result) == 3
        print("  ✅ Embedding parsing")
        passed += 1
        
        # Test keyword extraction
        keywords = utils.extract_query_keywords("machine learning algorithms")
        assert isinstance(keywords, list)
        print("  ✅ Keyword extraction")
        passed += 1
        
        # Test keyword relevance
        score = utils.calculate_keyword_relevance(
            "This is about machine learning", 
            ["machine", "learning"]
        )
        assert 0 <= score <= 1
        print("  ✅ Keyword relevance")
        passed += 1
        
        # Test empty inputs handle gracefully
        empty_result = utils.get_text_embedding("")
        assert isinstance(empty_result, np.ndarray)
        print("  ✅ Empty input handling")
        passed += 1
        
        return passed, failed
        
    except Exception as e:
        print(f"  ❌ Utility test failed: {e}")
        traceback.print_exc()
        return 0, 1

def run_file_structure_tests():
    """Test that required files and directories exist."""
    print("\n📁 Testing file structure...")
    
    required_files = [
        "main.py",
        "utils.py", 
        "config.py",
        "data_preprocessor.py",
        "requirements.txt",
        "README.md"
    ]
    
    required_dirs = [
        "source_documents",
        "tests"
    ]
    
    passed = 0
    failed = 0
    
    for file in required_files:
        if os.path.exists(file):
            print(f"  ✅ {file}")
            passed += 1
        else:
            print(f"  ❌ Missing: {file}")
            failed += 1
    
    for dir in required_dirs:
        if os.path.isdir(dir):
            print(f"  ✅ {dir}/")
            passed += 1
        else:
            print(f"  ❌ Missing directory: {dir}/")
            failed += 1
    
    return passed, failed

def run_integration_test():
    """Test basic integration without heavy models."""
    print("\n🔗 Testing basic integration...")
    
    try:
        # Test that we can load config and initialize utils
        import config
        cfg = config.get_config()
        
        # Test that data directory structure can be created
        data_dir = cfg["output_data_dir"]
        os.makedirs(data_dir, exist_ok=True)
        
        images_dir = cfg["images_dir"]  
        os.makedirs(images_dir, exist_ok=True)
        
        print(f"  ✅ Directory structure created")
        print(f"  ✅ Config integration working")
        
        return 2, 0
        
    except Exception as e:
        print(f"  ❌ Integration test failed: {e}")
        return 0, 1

def main():
    """Run all tests and report results."""
    print("🚀 Running Multimodal Document Search Tests\n")
    print("=" * 50)
    
    total_passed = 0
    total_failed = 0
    
    # Run all test suites
    test_suites = [
        ("File Structure", run_file_structure_tests),
        ("Module Imports", run_import_tests),
        ("Configuration", run_config_tests),
        ("Utilities", run_utility_tests),
        ("Integration", run_integration_test),
    ]
    
    for suite_name, test_func in test_suites:
        try:
            passed, failed = test_func()
            total_passed += passed
            total_failed += failed
        except Exception as e:
            print(f"\n❌ {suite_name} test suite failed: {e}")
            total_failed += 1
    
    # Summary
    print("\n" + "=" * 50)
    print(f"📊 Test Results Summary")
    print(f"✅ Passed: {total_passed}")
    print(f"❌ Failed: {total_failed}")
    print(f"📈 Success Rate: {total_passed/(total_passed+total_failed)*100:.1f}%")
    
    if total_failed == 0:
        print("\n🎉 All tests passed! Your setup looks good.")
        return 0
    else:
        print(f"\n⚠️ {total_failed} test(s) failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)