# 🔍 Multimodal Document Search Engine

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.39.0-red.svg)](https://streamlit.io/)
[![CLIP](https://img.shields.io/badge/model-CLIP--ViT--Large-green.svg)](https://huggingface.co/openai/clip-vit-large-patch14)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A powerful multimodal search engine that enables natural language querying across both text content and images within your document collection. Built with OpenAI's CLIP model for state-of-the-art cross-modal understanding.

## ✨ Key Features

- 🔍 **Cross-Modal Search**: Find images using text queries and vice versa
- 📄 **Intelligent Document Processing**: Automatic text extraction and image filtering
- 🎯 **Advanced Search**: Keyword relevance boosting + semantic similarity
- 🎨 **Modern UI**: Clean, responsive interface with real-time search
- ⚡ **Performance Optimized**: Model caching, efficient similarity search
- 🔧 **Configurable**: Easy model and parameter customization
- 📱 **User Friendly**: Intuitive controls with helpful guidance

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- 4GB+ RAM (for CLIP model)
- Git

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/ShafqaatMalik/multimodal_document_search.git
cd multimodal_document_search
```

2. **Create and activate virtual environment**
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Usage

1. **Add your documents**: Place PDF files in the `source_documents/` directory

2. **Process documents** (first time only):
```bash
python data_preprocessor.py
```
This extracts text, images, and generates embeddings (may take several minutes).

3. **Launch the application**:
```bash
streamlit run main.py
```

4. **Start searching**: Open your browser and start querying your documents!

## 📁 Project Structure
```
multimodal_document_search/
├── 📁 source_documents/     # Place your PDF documents here
│   ├── attention.pdf        # Sample: Attention mechanism paper
│   └── example.pdf          # Your documents go here
├── 📁 data/                 # Generated data (auto-created)
│   ├── images/              # Extracted images
│   └── multimodal_data.csv  # Processed embeddings
├── 🐍 main.py               # Streamlit web application
├── 🔧 data_preprocessor.py  # Document processing pipeline
├── 🛠️ utils.py              # Core search and embedding functions
├── ⚙️ config.py             # Configuration management
├── 📋 requirements.txt      # Python dependencies
├── 📝 CHANGELOG.md          # Version history
├── 🚫 .gitignore           # Git ignore rules
├── 🔒 .python-version      # Python version specification
└── 📖 README.md            # This documentation
```

## 🔧 Configuration

The system is highly configurable through `config.py`. Key settings include:

### Model Configuration
```python
# Switch between CLIP model variants
CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"  # Default
# Alternatives: 
# - "openai/clip-vit-base-patch32" (smaller, faster)
# - "openai/clip-vit-large-patch14-336" (higher resolution)
```

### Processing Parameters
```python
TEXT_CHUNK_SIZE = 500           # Words per text chunk
TEXT_CHUNK_OVERLAP = 50         # Overlap between chunks
MIN_IMAGE_WIDTH = 280           # Filter small images
KEYWORD_BOOST_WEIGHT = 0.3      # Search relevance tuning
```

### Environment Variables
Override settings without code changes:
```bash
export CLIP_MODEL_NAME="openai/clip-vit-base-patch32"
export DEFAULT_TOP_N=15
export MAX_TEXT_LENGTH=15000
```

## 🎯 How It Works

### 1. Document Processing
- **Text Extraction**: PyMuPDF extracts text from PDFs page by page
- **Text Chunking**: Long text split into overlapping chunks for better embedding
- **Image Filtering**: Intelligent filtering removes icons, symbols, and duplicates
- **Embedding Generation**: CLIP creates 768-dimensional vectors for text and images

### 2. Search Pipeline
- **Query Processing**: Extract keywords and generate embedding
- **Similarity Calculation**: Cosine similarity between query and document embeddings
- **Relevance Boosting**: Combine semantic similarity (70%) + keyword relevance (30%)
- **Result Ranking**: Sort by combined score and present top matches

### 3. Advanced Features
- **Cross-Modal Search**: Text queries find relevant images and vice versa
- **Smart Filtering**: By document, modality, similarity threshold
- **Performance Optimization**: Model caching, efficient vector operations
- **Error Resilience**: Graceful handling of missing files, corrupt data

## 📊 Performance & Scalability

### Model Performance
- **CLIP ViT-Large**: State-of-the-art multimodal understanding
- **Memory Usage**: ~2GB RAM for model, ~500MB for typical document set
- **Speed**: ~1-2 seconds per query after initial model load

### Scaling Considerations
- **Documents**: Tested with 100+ PDFs, hundreds of pages
- **Images**: Handles thousands of images with intelligent filtering
- **Search Speed**: Sub-second response for datasets up to 10k embeddings

## 🛠️ Development Setup

### For Contributors

1. **Fork and clone**:
```bash
git clone https://github.com/your-username/multimodal_document_search.git
cd multimodal_document_search
```

2. **Install development dependencies**:
```bash
pip install -r requirements.txt
# For development, also install:
pip install pytest black flake8
```

3. **Run tests** (when available):
```bash
python -m pytest tests/
```

### Code Quality
- Follow PEP 8 style guidelines
- Use meaningful variable names and docstrings
- Test changes with sample documents before committing

## 🐛 Troubleshooting

### Common Issues

**"No module named 'torch'"**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**"CLIP model download fails"**
- Check internet connection
- Ensure sufficient disk space (>5GB)
- Try restarting the process

**"Out of memory error"**
- Use smaller CLIP model: set `CLIP_MODEL_NAME="openai/clip-vit-base-patch32"`
- Reduce batch size in processing
- Close other applications to free RAM

**"No results found"**
- Check that data preprocessing completed successfully
- Verify `data/multimodal_data.csv` exists and has content
- Try lowering similarity threshold in advanced options

### Performance Optimization
- **Slow search**: Reduce `DEFAULT_TOP_N` in config
- **High memory usage**: Switch to base CLIP model
- **Long preprocessing**: Filter documents, use smaller images

## 📚 Documentation

### API Reference
- **`utils.py`**: Core embedding and search functions
- **`config.py`**: Configuration management and environment variables
- **`data_preprocessor.py`**: Document processing pipeline
- **`main.py`**: Streamlit web application

### Example Queries
Try these sample searches to get started:
- *"transformer architecture attention mechanism"*
- *"neural network diagram"*
- *"financial performance charts"*
- *"mathematical equations and formulas"*

## 🎥 Demo

A recorded demonstration of the application is available:
[📹 Watch Demo Video](https://drive.google.com/file/d/18rTaSKrvuOD1cN1nj-nqvim1PKla2s7V/view?usp=drive_link)

## 📈 Roadmap

### Upcoming Features
- [ ] Support for additional document formats (Word, PowerPoint)
- [ ] Advanced filtering options (date, author, document type)
- [ ] Batch query processing
- [ ] Integration with cloud storage (S3, Google Drive)
- [ ] Multi-language support
- [ ] Advanced analytics and search insights

### Performance Improvements
- [ ] GPU acceleration support
- [ ] Distributed processing for large document sets
- [ ] Incremental embedding updates
- [ ] Vector database integration (Pinecone, Weaviate)

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. **Fork the repository** and create a feature branch
2. **Make your changes** with proper testing
3. **Follow code style** guidelines (PEP 8, meaningful names)
4. **Update documentation** as needed
5. **Submit a pull request** with clear description

### Development Workflow
```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make changes and commit
git commit -m "Add: your feature description"

# Push and create pull request
git push origin feature/your-feature-name
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI** for the CLIP model and architecture
- **Hugging Face** for model hosting and transformers library
- **Streamlit** for the excellent web framework
- **PyMuPDF** for robust PDF processing capabilities

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/ShafqaatMalik/multimodal_document_search/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ShafqaatMalik/multimodal_document_search/discussions)
- **Email**: [Your contact email if desired]

## ⭐ Star History

If you find this project useful, please consider giving it a star! It helps others discover the project.

---

**Built with ❤️ for the AI and document processing community**
