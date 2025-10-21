# Multimodal Document Search Engine

A multimodal search engine that transforms your document collections into searchable vector embeddings using OpenAI's CLIP model. The system extracts and processes both textual content and images from PDF documents, converting them into high-dimensional vector representations that enable semantic search across modalities. Users can query using natural language to find relevant text passages or images, with the search powered by cosine similarity calculations between query embeddings and document embeddings. This approach enables cross-modal search capabilities where text queries can retrieve relevant images and vice versa, all through unified vector space representation.

## Key Features

- **Cross-Modal Search**: Find images using text queries and vice versa
- **Intelligent Document Processing**: Automatic text extraction and image filtering
- **Advanced Search**: Keyword relevance boosting + semantic similarity
- **Modern UI**: Clean, responsive interface with real-time search
- **Performance Optimized**: Model caching, efficient similarity search
- **Configurable**: Easy model and parameter customization
- **User Friendly**: Intuitive controls with helpful guidance

## Quick Start

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

1. **Use included documents or add your own**: The repository comes with sample PDF files in the `source_documents/` directory. You can add more PDF files there if needed.

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

## Project Structure
```
multimodal_document_search/
├── source_documents/        # PDF documents included in the repository
│   ├── attention.pdf        # Sample: Attention mechanism paper
│   ├── TSLA-Q2-2025-Update.pdf  # Tesla quarterly update sample
│   └── whitepaper_emebddings_vectorstores_v2.pdf  # Embeddings whitepaper
├── data/                    # Generated data (auto-created)
│   ├── images/              # Extracted images
│   └── multimodal_data.csv  # Processed embeddings
├── main.py                  # Streamlit web application
├── data_preprocessor.py     # Document processing pipeline
├── utils.py                 # Core search and embedding functions
├── config.py                # Configuration management
├── requirements.txt         # Python dependencies
└── README.md               # This documentation
```

## Configuration

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

## How It Works

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

## Performance & Scalability

### Model Performance
- **CLIP ViT-Large**: State-of-the-art multimodal understanding
- **Memory Usage**: ~2GB RAM for model, ~500MB for typical document set
- **Speed**: ~1-2 seconds per query after initial model load

### For Contributors

1. **Fork and clone**:
```bash
git clone https://github.com/your-username/multimodal_document_search.git
cd multimodal_document_search
```

2. **Create virtual environment and install dependencies**:
```bash
python -m venv .venv
.venv\Scripts\activate  # On Windows
pip install -r requirements.txt
```

## Troubleshooting

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


## Documentation

### Code Structure
- **`utils.py`**: Core embedding and search functions
- **`config.py`**: Configuration management and environment variables
- **`data_preprocessor.py`**: Document processing pipeline
- **`main.py`**: Streamlit web application

### Example Queries
Try these sample searches with the included documents:
- *"transformer architecture attention mechanism"* (attention.pdf)
- *"vector embeddings and databases"* (whitepaper_emebddings_vectorstores_v2.pdf)
- *"Tesla financial performance"* (TSLA-Q2-2025-Update.pdf)
- *"energy production and storage"* (TSLA-Q2-2025-Update.pdf)

## Demo

A recorded demonstration of the application is available:
[Watch Demo Video](https://drive.google.com/file/d/18rTaSKrvuOD1cN1nj-nqvim1PKla2s7V/view?usp=drive_link)


