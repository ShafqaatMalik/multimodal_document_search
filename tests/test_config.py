"""
Unit tests for configuration management.
"""
import os
import sys
import tempfile
from unittest.mock import patch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from config import get_config, CLIP_MODEL_NAME, EMBEDDING_DIM


class TestConfig:
    """Test configuration management functionality."""
    
    def test_default_config_values(self):
        """Test that default configuration values are correct."""
        config = get_config()
        
        # Test essential config keys exist
        assert "clip_model" in config
        assert "embedding_dim" in config
        assert "source_docs_dir" in config
        assert "output_data_dir" in config
        
        # Test default values
        assert config["clip_model"] == CLIP_MODEL_NAME
        assert config["embedding_dim"] == EMBEDDING_DIM
        assert isinstance(config["text_chunk_size"], int)
        assert isinstance(config["max_text_length"], int)
    
    def test_environment_override(self):
        """Test that environment variables override default config."""
        test_model = "test/model"
        test_dim = 512
        
        with patch.dict(os.environ, {
            'CLIP_MODEL_NAME': test_model,
            'CLIP_EMBEDDING_DIM': str(test_dim)
        }):
            config = get_config()
            assert config["clip_model"] == test_model
            assert config["embedding_dim"] == test_dim
    
    def test_config_types(self):
        """Test that config values have correct types."""
        config = get_config()
        
        assert isinstance(config["clip_model"], str)
        assert isinstance(config["embedding_dim"], int)
        assert isinstance(config["text_chunk_size"], int)
        assert isinstance(config["text_chunk_overlap"], int)
        assert isinstance(config["max_text_length"], int)
        assert isinstance(config["default_top_n"], int)
        assert isinstance(config["min_similarity"], float)
    
    def test_config_ranges(self):
        """Test that config values are within reasonable ranges."""
        config = get_config()
        
        # Embedding dimension should be positive
        assert config["embedding_dim"] > 0
        
        # Chunk size should be reasonable
        assert 50 <= config["text_chunk_size"] <= 2000
        
        # Overlap should be less than chunk size
        assert 0 <= config["text_chunk_overlap"] < config["text_chunk_size"]
        
        # Max text length should be reasonable
        assert 1000 <= config["max_text_length"] <= 100000
        
        # Top N should be reasonable
        assert 1 <= config["default_top_n"] <= 100
        
        # Similarity should be valid probability
        assert 0.0 <= config["min_similarity"] <= 1.0


if __name__ == "__main__":
    pytest.main([__file__])