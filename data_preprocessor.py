"""
Data preprocessing script to:
1. Extract text and images from source documents (e.g., PDFs).
2. Chunk text and save it.
3. Save image filepaths.
4. Compute embeddings for both text and images using CLIP.
5. Store everything in a structured CSV file.

NOTE: This example uses PyMuPDF (fitz) for PDF processing.
"""
import os
import hashlib
import fitz  # PyMuPDF
import pandas as pd
from tqdm import tqdm
from logger import get_logger

# Initialize logger
logger = get_logger(__name__)

from utils import get_text_embedding, get_image_embedding

# --- Configuration ---
from config import get_config, TEXT_CHUNK_SIZE, TEXT_CHUNK_OVERLAP
from config import (
    SOURCE_DOCS_DIR, OUTPUT_DATA_DIR, IMAGES_DIR, OUTPUT_CSV_PATH,
    MIN_IMAGE_WIDTH, MIN_IMAGE_HEIGHT, MIN_IMAGE_AREA, MIN_LONG_EDGE,
    SKIP_SQUARE_ICONS, SQUARE_TOLERANCE, MAX_SQUARE_ICON_SIZE, DEDUPLICATE_IMAGES
)

# Create output directories if they don't exist
os.makedirs(IMAGES_DIR, exist_ok=True)

def chunk_text(text, chunk_size=TEXT_CHUNK_SIZE, overlap=TEXT_CHUNK_OVERLAP):
    """Splits text into overlapping chunks using configuration values."""
    tokens = text.split()
    chunks = []
    for i in range(0, len(tokens), chunk_size - overlap):
        chunks.append(" ".join(tokens[i : i + chunk_size]))
    return chunks

def process_documents():
    """
    Main function to process all documents in the source directory.
    """
    all_data = []
    doc_files = [f for f in os.listdir(SOURCE_DOCS_DIR) if f.endswith(".pdf")]
    
    logger.info(f"Starting document processing for {len(doc_files)} PDF files")
    logger.info(f"Output directory: {OUTPUT_DATA_DIR}")
    logger.info(f"Images directory: {IMAGES_DIR}")

    for doc_name in doc_files:
        doc_path = os.path.join(SOURCE_DOCS_DIR, doc_name)
        logger.info(f"Processing document: {doc_name}...")
        doc = fitz.open(doc_path)
        
        seen_hashes = set() if DEDUPLICATE_IMAGES else None
        kept, skip_small, skip_square, skip_dup = 0, 0, 0, 0
        logger.debug(f"Document has {len(doc)} pages")

        # --- Process Text ---
        text_chunk_count = 0
        for page_num, page in enumerate(tqdm(doc, desc=f"Pages in {doc_name}")):
            text = page.get_text()
            text_chunks = chunk_text(text)
            
            logger.debug(f"Page {page_num+1}: {len(text_chunks)} text chunks")
            
            for chunk in text_chunks:
                if not chunk.strip():
                    continue
                embedding = get_text_embedding(chunk)
                all_data.append({
                    "modality": "text",
                    "content": chunk,
                    "embedding": list(embedding),
                    "source_doc": doc_name,
                    "page": page_num + 1,
                    "caption": ""
                })
                text_chunk_count += 1
        
            # --- Process Images ---
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                width = base_image.get("width", 0)
                height = base_image.get("height", 0)
                
                # Filter 1: Size-based (removes tiny icons/symbols)
                if (width < MIN_IMAGE_WIDTH or 
                    height < MIN_IMAGE_HEIGHT or
                    width * height < MIN_IMAGE_AREA or
                    max(width, height) < MIN_LONG_EDGE):
                    skip_small += 1
                    continue
                
                # Filter 2: Square icon detection (T, checkboxes, bullets, etc.)
                if SKIP_SQUARE_ICONS and max(width, height) <= MAX_SQUARE_ICON_SIZE:
                    aspect_deviation = abs(width - height) / max(width, height)
                    if aspect_deviation <= SQUARE_TOLERANCE:
                        skip_square += 1
                        continue
                
                # Filter 3: Deduplication (repeated logos/watermarks)
                if DEDUPLICATE_IMAGES:
                    img_hash = hashlib.md5(image_bytes).hexdigest()
                    if img_hash in seen_hashes:
                        skip_dup += 1
                        continue
                    seen_hashes.add(img_hash)
                
                image_filename = f"{os.path.splitext(doc_name)[0]}_p{page_num + 1}_{img_index}.{image_ext}"
                image_filepath = os.path.join(IMAGES_DIR, image_filename)

                with open(image_filepath, "wb") as f:
                    f.write(image_bytes)

                embedding = get_image_embedding(image_filepath)
                if embedding is not None:
                    all_data.append({
                        "modality": "image",
                        "content": image_filepath, # Store the path
                        "embedding": list(embedding),
                        "source_doc": doc_name,
                        "page": page_num + 1,
                        "caption": f"Image {img_index + 1} from page {page_num + 1}" # Placeholder caption
                    })
                    kept += 1
        logger.info(f"  ✓ Kept {kept} images | Filtered: {skip_small} small, {skip_square} square-icons, {skip_dup} duplicates")
        
        # Add a log message for the completed document
        logger.info(f"Completed processing document: {doc_name}")

    # --- Save to CSV ---
    df = pd.DataFrame(all_data)
    df.to_csv(OUTPUT_CSV_PATH, index=False)
    
    text_count = len(df[df["modality"] == "text"])
    image_count = len(df[df["modality"] == "image"])
    logger.info(f"\nPreprocessing complete. Data saved to {OUTPUT_CSV_PATH}")
    logger.info(f"Summary: {text_count} text chunks, {image_count} images from {len(doc_files)} documents")

if __name__ == "__main__":
    process_documents()
