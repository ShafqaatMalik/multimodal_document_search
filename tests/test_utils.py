"""
Unit tests for utility functions.
"""
import os
import sys
import numpy as np
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (
    parse_embedding, 
    extract_query_keywords, 
    calculate_keyword_relevance,
    get_text_embedding,
    get_image_embedding
)


class TestUtilityFunctions:
    """Test core utility functions."""
    
    def test_parse_embedding_valid_list_string(self):
        """Test parsing valid list string embedding."""
        test_embedding = "[0.1, 0.2, 0.3, 0.4, 0.5]"
        result = parse_embedding(test_embedding)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == 5
        assert np.allclose(result, [0.1, 0.2, 0.3, 0.4, 0.5])
    
    def test_parse_embedding_list_input(self):
        """Test parsing list input embedding."""
        test_embedding = [0.1, 0.2, 0.3]
        result = parse_embedding(test_embedding)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == 3
        assert np.allclose(result, [0.1, 0.2, 0.3])
    
    def test_parse_embedding_numpy_input(self):
        """Test parsing numpy array input."""
        test_embedding = np.array([0.1, 0.2, 0.3])
        result = parse_embedding(test_embedding)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == 3
        assert np.allclose(result, [0.1, 0.2, 0.3])
    
    def test_parse_embedding_invalid_input(self):
        """Test parsing invalid input returns zero vector."""
        result = parse_embedding("invalid_input")
        
        assert isinstance(result, np.ndarray)
        assert len(result) == 768  # Should return CLIP embedding dimension
        assert all(x == 0 for x in result)
    
    def test_parse_embedding_empty_input(self):
        """Test parsing empty input returns zero vector."""
        result = parse_embedding("")
        
        assert isinstance(result, np.ndarray)
        assert len(result) > 0
        assert all(x == 0 for x in result)
    
    def test_extract_query_keywords_basic(self):
        """Test basic keyword extraction."""
        query = "machine learning algorithms"
        keywords = extract_query_keywords(query)
        
        assert isinstance(keywords, list)
        assert "machine" in keywords or "learning" in keywords or "algorithms" in keywords
    
    def test_extract_query_keywords_definition_pattern(self):
        """Test keyword extraction from definition patterns."""
        query = "what is transformer architecture"
        keywords = extract_query_keywords(query)
        
        assert isinstance(keywords, list)
        assert len(keywords) > 0
        # Should extract "transformer architecture" or parts of it
        assert any("transformer" in k or "architecture" in k for k in keywords)
    
    def test_extract_query_keywords_empty(self):
        """Test keyword extraction from empty query."""
        keywords = extract_query_keywords("")
        
        assert isinstance(keywords, list)
        # Empty query should return empty list or very few keywords
        assert len(keywords) <= 1
    
    def test_calculate_keyword_relevance_basic(self):
        """Test basic keyword relevance calculation."""
        text = "This text discusses machine learning and neural networks."
        keywords = ["machine", "learning"]
        score = calculate_keyword_relevance(text, keywords)
        
        assert isinstance(score, (int, float))
        assert 0 <= score <= 1
        assert score > 0  # Should find some relevance
    
    def test_calculate_keyword_relevance_no_match(self):
        """Test keyword relevance with no matching keywords."""
        text = "This text is about cooking recipes."
        keywords = ["machine", "learning"]
        score = calculate_keyword_relevance(text, keywords)
        
        assert isinstance(score, (int, float))
        assert score == 0  # No matching keywords
    
    def test_calculate_keyword_relevance_empty_inputs(self):
        """Test keyword relevance with empty inputs."""
        score1 = calculate_keyword_relevance("", ["keyword"])
        score2 = calculate_keyword_relevance("text", [])
        score3 = calculate_keyword_relevance("", [])
        
        assert all(score == 0 for score in [score1, score2, score3])
    
    @patch('utils.get_clip_model')
    def test_get_text_embedding_valid_input(self, mock_get_model):
        """Test text embedding generation with valid input."""
        # Mock the model and processor
        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_get_model.return_value = (mock_model, mock_processor)
        
        # Mock the model output
        mock_features = MagicMock()
        mock_features.cpu.return_value.numpy.return_value.flatten.return_value = np.array([0.1, 0.2, 0.3])
        mock_model.get_text_features.return_value = mock_features
        
        result = get_text_embedding("test text")
        
        assert isinstance(result, np.ndarray)
        assert len(result) == 3
        mock_processor.assert_called_once()
        mock_model.get_text_features.assert_called_once()
    
    def test_get_text_embedding_empty_input(self):
        """Test text embedding with empty input."""
        result = get_text_embedding("")
        
        assert isinstance(result, np.ndarray)
        assert len(result) > 0
        assert all(x == 0 for x in result)
    
    def test_get_text_embedding_none_input(self):
        """Test text embedding with None input."""
        result = get_text_embedding(None)
        
        assert isinstance(result, np.ndarray)
        assert len(result) > 0
        assert all(x == 0 for x in result)
    
    def test_get_image_embedding_invalid_path(self):
        """Test image embedding with invalid path."""
        result = get_image_embedding("nonexistent_file.jpg")
        
        assert isinstance(result, np.ndarray)
        assert len(result) > 0
        assert all(x == 0 for x in result)
    
    def test_get_image_embedding_empty_path(self):
        """Test image embedding with empty path."""
        result = get_image_embedding("")
        
        assert isinstance(result, np.ndarray)
        assert len(result) > 0
        assert all(x == 0 for x in result)
    
    def test_get_image_embedding_url_path(self):
        """Test image embedding with URL path (should return zero vector)."""
        result = get_image_embedding("http://example.com/image.jpg")
        
        assert isinstance(result, np.ndarray)
        assert len(result) > 0
        assert all(x == 0 for x in result)


if __name__ == "__main__":
    # Simple test runner
    test_class = TestUtilityFunctions()
    test_methods = [method for method in dir(test_class) if method.startswith('test_')]
    
    passed = 0
    failed = 0
    
    for method_name in test_methods:
        try:
            method = getattr(test_class, method_name)
            method()
            print(f"✅ {method_name}")
            passed += 1
        except Exception as e:
            print(f"❌ {method_name}: {e}")
            failed += 1
    
    print(f"\nResults: {passed} passed, {failed} failed")