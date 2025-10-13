# Contributing to Multimodal Document Search

Thank you for your interest in contributing to this project! This document provides guidelines for contributing.

## 🛠 Development Setup

### Prerequisites
- Python 3.8+ (recommended: 3.9 or 3.10)
- Git
- Virtual environment tool (conda, venv, or similar)

### Quick Start

1. **Clone the repository:**
```bash
git clone https://github.com/ShafqaatMalik/multimodal_document_search.git
cd multimodal_document_search
```

2. **Set up Python environment:**
```bash
# Using conda (recommended)
conda create -n multimodal-search python=3.9
conda activate multimodal-search

# Or using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
# Install main dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt
```

4. **Run tests:**
```bash
# Quick test runner
python run_tests.py

# Or full test suite
python tests/test_config.py
python tests/test_utils.py
```

5. **Start the application:**
```bash
streamlit run main.py
```

## 📝 Development Guidelines

### Code Style
- Follow PEP 8 Python style guidelines
- Use type hints where appropriate
- Write descriptive docstrings for functions and classes
- Keep functions focused and modular

### Configuration Management
- All configuration should go through `config.py`
- Use environment variables for deployment-specific settings
- Document new configuration options in README.md

### Testing
- Write unit tests for new functionality
- Test both success and error cases
- Use mocking for external dependencies (CLIP model, file I/O)
- Maintain test coverage above 80%

### Documentation
- Update README.md for user-facing changes
- Update CHANGELOG.md for all changes
- Add inline documentation for complex logic
- Include examples in docstrings

## 🧪 Testing Guidelines

### Running Tests
```bash
# Quick validation
python run_tests.py

# Detailed unit tests
python tests/test_utils.py
python tests/test_config.py

# With coverage (if installed)
pytest --cov=utils tests/
```

### Writing Tests
- Place tests in the `tests/` directory
- Name test files as `test_<module_name>.py`
- Use descriptive test names: `test_<function>_<scenario>`
- Mock external dependencies (CLIP model, file system)

### Test Structure
```python
def test_function_name_scenario():
    """Test description of what this test validates."""
    # Arrange
    input_data = "test input"
    
    # Act
    result = function_to_test(input_data)
    
    # Assert
    assert expected_result == result
```

## 📋 Pull Request Process

1. **Create a feature branch:**
```bash
git checkout -b feature/your-feature-name
```

2. **Make your changes:**
- Write code following the guidelines above
- Add/update tests for your changes
- Update documentation as needed

3. **Test your changes:**
```bash
python run_tests.py
```

4. **Commit your changes:**
```bash
git add .
git commit -m "🎉 Add feature: Your feature description

- Specific change 1
- Specific change 2
- Include any breaking changes"
```

5. **Push and create PR:**
```bash
git push origin feature/your-feature-name
```

6. **Create Pull Request:**
- Provide clear description of changes
- Reference any related issues
- Include testing steps
- Request review from maintainers

## 🏷 Commit Message Guidelines

### Format
```
🎨 Type: Brief description

- Detailed change 1  
- Detailed change 2
- Any breaking changes or migration notes
```

### Types
- 🎉 **feat**: New feature
- 🐛 **fix**: Bug fix
- 📚 **docs**: Documentation changes
- 🎨 **style**: Code style changes (formatting, no logic changes)
- ♻️ **refactor**: Code refactoring
- 🧪 **test**: Adding or updating tests
- 🔧 **chore**: Maintenance tasks

### Examples
```
🎉 feat: Add image similarity search

- Implement CLIP image embedding generation
- Add image upload component to Streamlit UI
- Include image similarity scoring in search results

🐛 fix: Handle empty search queries gracefully

- Add input validation for empty/whitespace queries
- Display helpful message for invalid searches
- Prevent CLIP model errors on empty input
```

## 🚀 Deployment Guidelines

### Environment Variables
Set these for production deployment:
```bash
CLIP_MODEL="openai/clip-vit-large-patch14"
CLIP_EMBEDDING_DIM=768
TEXT_CHUNK_SIZE=512
MIN_SIMILARITY_THRESHOLD=0.1
```

### Performance Considerations
- CLIP model loading takes ~2-3 seconds on first run
- Consider model caching for production deployments
- Large PDF files may require increased memory limits
- Image processing is CPU-intensive

## 📞 Getting Help

- **Issues**: Open GitHub issues for bugs or feature requests
- **Discussions**: Use GitHub discussions for questions
- **Documentation**: Check README.md for setup help

## 🙏 Recognition

Contributors will be recognized in:
- CHANGELOG.md for their specific contributions
- GitHub contributors section
- README.md acknowledgments

Thank you for contributing to making multimodal search better! 🌟