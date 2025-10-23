"""
Utility functions for data processing, embedding generation, and search.
Enhanced with robust error handling and performance optimizations.
"""
import pandas as pd
import numpy as np
import torch
import ast
import os
import re  # Added for regex pattern matching
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from logger import get_logger, timeit

# Initialize logger
logger = get_logger(__name__)

# --- Load CLIP Model ---
# Using a singleton pattern to ensure the model is loaded only once for better performance
from config import get_config

config = get_config()
MODEL_NAME = config["clip_model"]
EMBEDDING_DIM = config["embedding_dim"]

model = None
processor = None

def get_clip_model():
    """
    Loads and returns the CLIP model and processor using singleton pattern.
    This prevents reloading the large model multiple times.
    """
    global model, processor
    if model is None:
        logger.info("Loading CLIP model... (This may take a moment on first run)")
        model = CLIPModel.from_pretrained(MODEL_NAME)
        processor = CLIPProcessor.from_pretrained(MODEL_NAME)
        logger.info("CLIP model loaded successfully!")
    return model, processor

# --- Data Loading ---
def load_data(filepath):
    """
    Loads data from a CSV file and converts string embeddings to numpy arrays.
    Enhanced with error handling and support for multiple embedding formats.
    """
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} entries from {filepath}")
        
        if "embedding" in df.columns:
            df["embedding"] = df["embedding"].apply(parse_embedding)
            logger.info("Successfully parsed embeddings")
        else:
            logger.error("No embedding columns found in data file")
            raise ValueError("No embedding columns found in data file")
        
        # Resource usage logging
        text_count = len(df[df["modality"] == "text"])
        image_count = len(df[df["modality"] == "image"])
        total_size = df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
        
        logger.info(f"RESOURCE_USAGE - Total entries: {len(df)}")
        logger.info(f"RESOURCE_USAGE - Text entries: {text_count}")
        logger.info(f"RESOURCE_USAGE - Image entries: {image_count}")
        logger.info(f"RESOURCE_USAGE - Memory usage: {total_size:.2f} MB")
        
        return df
    except FileNotFoundError:
        logger.error(f"File {filepath} not found!")
        logger.info("Please run 'python data_preprocessor.py' first to generate the data file.")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def parse_embedding(embedding_str):
    """
    Parse embedding from string format to numpy array.
    Handles multiple string formats with robust error handling.
    """
    try:
        if isinstance(embedding_str, str):
            # Handle string representation of list
            if embedding_str.startswith('[') and embedding_str.endswith(']'):
                return np.array(ast.literal_eval(embedding_str))
            else:
                # Handle comma-separated values
                clean_str = embedding_str.strip("[]")
                if clean_str:
                    parsed = np.fromstring(clean_str, sep=",")
                    if len(parsed) == 0:
                        logger.warning("Could not parse embedding string, using zero vector")
                        return np.zeros(EMBEDDING_DIM)
                    return parsed
                else:
                    logger.warning("Empty embedding string, using zero vector")
                    return np.zeros(EMBEDDING_DIM)
        elif isinstance(embedding_str, (list, np.ndarray)):
            # Already in proper format
            return np.array(embedding_str)
        else:
            logger.warning(f"Unknown embedding format {type(embedding_str)}, using zero vector")
            return np.zeros(EMBEDDING_DIM)
    except Exception as e:
        logger.warning(f"Error parsing embedding ({str(e)}), using zero vector")
        return np.zeros(EMBEDDING_DIM)

# --- Embedding Generation ---
def get_text_embedding(text):
    """
    Computes a text embedding using the CLIP model.
    Enhanced with input validation and error handling.
    """
    if not text or not isinstance(text, str) or not text.strip():
        logger.warning("Empty or invalid text input, using zero vector")
        return np.zeros(EMBEDDING_DIM)  # CLIP embeddings dimension from config
    
    try:
        model, processor = get_clip_model()
        
        # Truncate very long text to prevent memory issues and improve performance
        max_length = config["max_text_length"]
        if len(text) > max_length:
            text = text[:max_length]
            logger.warning(f"Text truncated to {max_length:,} characters for performance")
            
        inputs = processor(text=text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            text_features = model.get_text_features(**inputs)
        
        return text_features.cpu().numpy().flatten()
    except Exception as e:
        logger.error(f"Error generating text embedding: {str(e)}")
        return np.zeros(EMBEDDING_DIM)

def get_image_embedding(image_path):
    """
    Computes an image embedding using the CLIP model.
    Enhanced with robust error handling for various image formats and missing files.
    """
    if not image_path or not isinstance(image_path, str):
        logger.warning("Invalid image path")
        return np.zeros(EMBEDDING_DIM)
        
    # Handle placeholder URLs (common in documents with referenced images)
    if (image_path.startswith(('http://', 'https://')) or 
        image_path.startswith('placeholder_') or
        not image_path.strip()):
        logger.warning(f"Placeholder or URL detected: {image_path[:50]}...")
        return np.zeros(EMBEDDING_DIM)
    
    try:
        model, processor = get_clip_model()
        
        # Check if file exists
        if not os.path.exists(image_path):
            logger.warning(f"Image file not found: {image_path}")
            return np.zeros(EMBEDDING_DIM)
            
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
        
        return image_features.cpu().numpy().flatten()
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {str(e)}")
        return np.zeros(EMBEDDING_DIM)

# --- Improved Search and Similarity ---

def extract_query_keywords(query):
    """
    Extract important keywords from the query for boosting and filtering.
    Returns a list of keywords and their variations.
    """
    # Lowercase for case-insensitive matching
    query = query.lower()
    
    # Extract keywords based on common patterns
    keywords = []
    
    # Check for definition/explanation patterns
    definition_patterns = [
        r'explain\s+(?:the\s+)?(?:concept\s+of\s+)?([a-z\-]+(?:\s+[a-z\-]+){0,3})',
        r'what\s+is\s+(?:a\s+)?([a-z\-]+(?:\s+[a-z\-]+){0,3})',
        r'define\s+(?:the\s+)?([a-z\-]+(?:\s+[a-z\-]+){0,3})',
        r'meaning\s+of\s+([a-z\-]+(?:\s+[a-z\-]+){0,3})',
    ]
    
    for pattern in definition_patterns:
        matches = re.findall(pattern, query)
        if matches:
            for match in matches:
                keywords.append(match)
                # Add variations (e.g., "self-attention" -> "attention", "self attention")
                if "-" in match:
                    keywords.append(match.replace("-", " "))
                if " " in match:
                    keywords.extend(match.split())
    
    # If no definition patterns matched, extract any noun phrases
    if not keywords:
        # Simple heuristic: split on spaces and take words with 4+ chars
        words = [word for word in query.split() if len(word) >= 4]
        words = [w for w in words if w not in ["what", "when", "where", "explain", "concept", "define"]]
        keywords.extend(words)
    
    # Remove duplicates while preserving order
    seen = set()
    return [k for k in keywords if not (k in seen or seen.add(k))]

def calculate_keyword_relevance(text, keywords):
    """
    Calculate a relevance score based on keyword presence and positioning.
    Returns a score between 0 and 1.
    """
    if not text or not keywords:
        return 0
    
    text_lower = text.lower()
    
    # Initialize score
    score = 0
    max_score = len(keywords)  # Maximum possible score
    
    # Check for keywords
    for keyword in keywords:
        if keyword in text_lower:
            # Base score for presence
            keyword_score = 1
            
            # Boost score if keyword appears in the first sentence (likely a definition)
            first_sentence = text_lower.split('.')[0]
            if keyword in first_sentence:
                keyword_score += 0.5
                
            # Boost for definition patterns
            definition_patterns = [
                f"{keyword} is", 
                f"{keyword} refers to", 
                f"{keyword} means", 
                f"concept of {keyword}",
                f"definition of {keyword}"
            ]
            for pattern in definition_patterns:
                if pattern in text_lower:
                    keyword_score += 1
                    break
            
            # Count occurrences (with diminishing returns)
            occurrences = text_lower.count(keyword)
            if occurrences > 1:
                keyword_score += min(occurrences / 10, 0.5)  # Cap at +0.5
                
            score += keyword_score
    
    # Normalize score (0-1)
    normalized_score = min(score / (max_score * 2.5), 1.0)  # 2.5 is max possible per keyword
    return normalized_score

def find_similar_texts(query_embedding, text_df, top_n=5, query_text=None):
    """
    Finds the most similar text chunks to a query embedding.
    Enhanced with keyword relevance boosting for better semantic search.
    """
    if text_df.empty:
        logger.warning("No text data available for search")
        return []
        
    try:
        keywords = []
        if query_text and isinstance(query_text, str):
            keywords = extract_query_keywords(query_text)
            logger.debug(f"Extracted keywords: {keywords}")
        
        # Calculate vector similarity
        text_embeddings = np.vstack(text_df["embedding"].values)
        cosine_scores = cosine_similarity([query_embedding], text_embeddings)[0]
        
        # Apply keyword boosting if keywords were extracted
        if keywords:
            # Calculate keyword relevance for each text chunk
            relevance_scores = [calculate_keyword_relevance(text, keywords) 
                              for text in text_df["content"].values]
            
            # Combine scores (weighted average: 70% cosine similarity, 30% keyword relevance)
            combined_scores = 0.7 * cosine_scores + 0.3 * np.array(relevance_scores)
            logger.debug(f"Applied keyword boosting with {len(keywords)} keywords")
        else:
            combined_scores = cosine_scores
            logger.debug("No keywords extracted, using pure semantic similarity")

        # Get the top N results
        top_indices = combined_scores.argsort()[-top_n:][::-1]

        results = []
        for i in top_indices:
            content = text_df.iloc[i]["content"]
            
            # Skip empty or placeholder content
            if isinstance(content, str) and (
                content.startswith('placeholder_') or 
                content.strip() == "" or
                content.lower() == "n/a"
            ):
                content = "[Placeholder text content]"
                logger.debug(f"Replaced empty/placeholder content at index {i}")
                
            results.append(
                {
                    "modality": "text",
                    "content": content,
                    "source_doc": text_df.iloc[i]["source_doc"],
                    "page": text_df.iloc[i]["page"],
                    "similarity": float(combined_scores[i]),
                }
            )
        
        logger.info(f"Found {len(results)} similar text results")
        logger.debug(f"Top similarity score: {results[0]['similarity'] if results else 'N/A'}")
        return results
    except Exception as e:
        logger.error(f"Error in text search: {str(e)}")
        return []

def find_similar_images(query_embedding, image_df, top_n=5):
    """
    Finds the most similar images to a query embedding.
    Enhanced with error handling and support for placeholder images.
    """
    if image_df.empty:
        logger.warning("No image data available for search")
        return []
        
    try:
        # Calculate vector similarity
        image_embeddings = np.vstack(image_df["embedding"].values)
        similarities = cosine_similarity([query_embedding], image_embeddings)[0]

        # Get the top N results
        top_indices = similarities.argsort()[-top_n:][::-1]

        results = []
        placeholder_count = 0
        for i in top_indices:
            content = image_df.iloc[i]["content"]
            
            # Handle placeholder URLs or invalid paths
            is_placeholder = (isinstance(content, str) and (
                content.startswith(('http://', 'https://')) or 
                content.startswith('placeholder_') or
                not content.strip()
            ))
            
            if is_placeholder:
                placeholder_count += 1
            
            results.append(
                {
                    "modality": "image",
                    "content": content,
                    "is_placeholder": is_placeholder,
                    "source_doc": image_df.iloc[i]["source_doc"],
                    "page": image_df.iloc[i]["page"],
                    "caption": image_df.iloc[i].get("caption", "N/A"),
                    "similarity": similarities[i],
                }
            )
        
        logger.info(f"Found {len(results)} similar image results ({placeholder_count} placeholders)")
        if results:
            logger.debug(f"Top image similarity score: {results[0]['similarity']}")
        return results
    except Exception as e:
        logger.error(f"Error in image search: {str(e)}")
        return []
