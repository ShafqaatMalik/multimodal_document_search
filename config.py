"""
Configuration settings for the Multimodal Document Search Engine.
Centralized configuration management for reproducibility and easy modification.
"""

import os
from typing import Dict, Any

# --- Model Configuration ---
CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"
CLIP_EMBEDDING_DIM = 768

# Alternative models for experimentation
ALTERNATIVE_MODELS = {
    "base": "openai/clip-vit-base-patch32",  # Smaller, faster
    "large": "openai/clip-vit-large-patch14",  # Current default
    "large-336": "openai/clip-vit-large-patch14-336",  # Higher resolution
}

# --- Data Configuration ---
SOURCE_DOCS_DIR = "source_documents"
OUTPUT_DATA_DIR = "data"
IMAGES_DIR = os.path.join(OUTPUT_DATA_DIR, "images")
OUTPUT_CSV_PATH = os.path.join(OUTPUT_DATA_DIR, "multimodal_data.csv")

# --- Processing Configuration ---
# Text processing
TEXT_CHUNK_SIZE = 500
TEXT_CHUNK_OVERLAP = 50
MAX_TEXT_LENGTH = 10000  # Characters

# Image filtering settings
MIN_IMAGE_WIDTH = 280
MIN_IMAGE_HEIGHT = 200
MIN_IMAGE_AREA = 50_000
MIN_LONG_EDGE = 280
SKIP_SQUARE_ICONS = True
SQUARE_TOLERANCE = 0.12
MAX_SQUARE_ICON_SIZE = 350
DEDUPLICATE_IMAGES = True

# --- Search Configuration ---
DEFAULT_TOP_N = 10
DEFAULT_MIN_SIMILARITY = 0.0
KEYWORD_BOOST_WEIGHT = 0.3  # 30% keyword relevance, 70% semantic similarity

# --- UI Configuration ---
APP_TITLE = "Multimodal Document Search"
APP_ICON = "🔍"
DEFAULT_QUERY = "transformer architecture attention mechanism"

# --- Environment Configuration ---
def get_config() -> Dict[str, Any]:
    """
    Get configuration with environment variable overrides.
    Allows for easy configuration changes without code modification.
    """
    return {
        "clip_model": os.getenv("CLIP_MODEL_NAME", CLIP_MODEL_NAME),
        "embedding_dim": int(os.getenv("CLIP_EMBEDDING_DIM", CLIP_EMBEDDING_DIM)),
        "source_docs_dir": os.getenv("SOURCE_DOCS_DIR", SOURCE_DOCS_DIR),
        "output_data_dir": os.getenv("OUTPUT_DATA_DIR", OUTPUT_DATA_DIR),
        "images_dir": os.getenv("IMAGES_DIR", IMAGES_DIR),
        "output_csv": os.getenv("OUTPUT_CSV_PATH", OUTPUT_CSV_PATH),
        "text_chunk_size": int(os.getenv("TEXT_CHUNK_SIZE", TEXT_CHUNK_SIZE)),
        "text_chunk_overlap": int(os.getenv("TEXT_CHUNK_OVERLAP", TEXT_CHUNK_OVERLAP)),
        "max_text_length": int(os.getenv("MAX_TEXT_LENGTH", MAX_TEXT_LENGTH)),
        "default_top_n": int(os.getenv("DEFAULT_TOP_N", DEFAULT_TOP_N)),
        "min_similarity": float(os.getenv("DEFAULT_MIN_SIMILARITY", DEFAULT_MIN_SIMILARITY)),
    }

# --- Version Information ---
PROJECT_VERSION = "1.0.0"
PYTHON_VERSION_REQUIRED = "3.8+"
