"""
Main Streamlit application for the Multimodal Search Engine.
Enhanced with robust error handling, performance optimizations, and better UX.

This script creates a user interface where users can:
- Input a search query with validation
- Select search filters (modality, source document)
- View combined and sorted search results from text and images
- Get helpful feedback and error messages
"""

import streamlit as st
import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import (
    load_data,
    get_text_embedding,
    get_image_embedding,
    find_similar_texts,
    find_similar_images,
    get_clip_model,
)
from logger import get_logger, timeit
import time

# Initialize logger
logger = get_logger(__name__)

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Multimodal Search Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Enhanced Styling ---
st.markdown("""
<style>
    div[data-testid="stMarkdownContainer"] h1, .stMarkdown h1 {
        color: #1e3a8a !important;
        font-weight: 700 !important;
        text-align: center !important;
        margin-bottom: 1rem !important;
    }
    
    div[data-testid="stMarkdownContainer"] h2,
    div[data-testid="stMarkdownContainer"] h3,
    div[data-testid="stMarkdownContainer"] h4,
    .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
        color: #1e40af !important;
        font-weight: 600 !important;
        margin-top: 1.5rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    div[data-testid="stSidebar"] h1,
    div[data-testid="stSidebar"] h2,
    div[data-testid="stSidebar"] h3,
    .sidebar .stMarkdown h1,
    .sidebar .stMarkdown h2,
    .sidebar .stMarkdown h3,
    div[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] h2 {
        color: #1e40af !important;
        font-weight: 600 !important;
    }
    
    div[data-testid="stMarkdownContainer"] p, .stMarkdown p {
        color: #374151 !important;
        font-size: 20px !important;
        line-height: 1.7 !important;
        font-weight: 400 !important;
    }
    
    div[data-testid="stAlert"] {
        background-color: #eff6ff !important;
        border-left: 4px solid #1e40af !important;
        border-radius: 6px !important;
    }
    
    div[data-testid="stExpander"] summary {
        color: #1e40af !important;
        font-weight: 600 !important;
        font-size: 20px !important;
    }
    
    div[data-testid="stButton"] button {
        background: linear-gradient(135deg, #93c5fd 0%, #60a5fa 40%, #3b82f6 100%) !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 4px rgba(59,130,246,0.25), 0 4px 10px -2px rgba(59,130,246,0.3) !important;
        transition: filter 0.2s ease, box-shadow 0.2s ease, transform 0.15s ease !important;
    }
    
    div[data-testid="stButton"] button:hover:not(:disabled) {
        filter: brightness(1.07) !important;
        box-shadow: 0 3px 6px rgba(59,130,246,0.35), 0 6px 14px -2px rgba(59,130,246,0.4) !important;
    }
    
    div[data-testid="stButton"] button:active:not(:disabled) {
        transform: translateY(1px) !important;
        box-shadow: 0 1px 3px rgba(59,130,246,0.35) !important;
    }
    
    div[data-testid="stButton"] button:focus-visible {
        outline: 2px solid #bfdbfe !important;
        outline-offset: 2px !important;
    }
    
    .stSelectbox label, .stTextInput label {
        color: #1e40af !important;
        font-weight: 600 !important;
        font-size: 18px !important;
    }
    
    div[data-testid="stMetric"] {
        background-color: #f8fafc !important;
        padding: 1rem !important;
        border-radius: 8px !important;
        border: 1px solid #e2e8f0 !important;
    }
    
    div[data-testid="stMetric"] label {
        color: #1e40af !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Performance Optimization: Cache Data Loading ---
@st.cache_data(show_spinner="Loading search data...")
def load_search_data():
    """
    Loads and returns the preprocessed data from the CSV file.
    Uses Streamlit caching for better performance on subsequent loads.
    """
    try:
        logger.info("Loading search data from CSV file...")
        data = load_data("data/multimodal_data.csv")
        logger.info(f"Successfully loaded data with {len(data)} entries")
        return data
    except FileNotFoundError:
        logger.error("Data file not found! Please run data_preprocessor.py first")
        st.error(
            "❌ **Data file not found!** Please run `python data_preprocessor.py` first to generate the required data file."
        )
        st.info("💡 The data preprocessing step extracts text and images from your documents and creates embeddings.")
        st.code("python data_preprocessor.py", language="bash")
        return None
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        st.error(f"❌ **Error loading data:** {str(e)}")
        st.info("💡 Try regenerating the data file by running the preprocessor again.")
        return None

# --- Performance Optimization: Cache Model Loading ---
@st.cache_resource(show_spinner="Loading CLIP model... (This may take a moment on first run)")
def load_clip_model_cached():
    """
    Cache the CLIP model loading to avoid reloading the 1.7GB model on every session.
    This provides ~80% faster app restarts after the first load.
    """
    logger.info("Loading CLIP model (this will be cached for subsequent runs)...")
    try:
        model, processor = get_clip_model()
        logger.info(f"Successfully loaded CLIP model: {model.__class__.__name__}")
        return model, processor
    except Exception as e:
        logger.error(f"Error loading CLIP model: {str(e)}")
        st.error(f"❌ **Error loading CLIP model:** {str(e)}")
        return None, None

# Initialize model at startup for better performance
logger.info("Initializing application...")
model, processor = load_clip_model_cached()

# Load data with error handling
df = load_search_data()

if df is not None:
    # Main title
    st.markdown(
        """
        <div style='text-align: center; margin-bottom: 0; margin-top: -2.3rem;'>
            <h1 style='color: #1e3a8a; font-weight: 700; font-size: 2.85rem; margin: 0 0 -0.55rem 0; line-height: 1.05;'>
                Multimodal Document Search
            </h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style='text-align: center; margin-top: -1.05rem; margin-bottom: 0.75rem;'>
            <p style='font-size: 20px; color: #4b5563; line-height: 1.28; font-weight: 400; margin: 0;'>
                Search through text and images from your documents using natural language.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Display data statistics
    with st.expander("📊 Dataset Information", expanded=False):
        text_count = len(df[df["modality"] == "text"])
        image_count = len(df[df["modality"] == "image"])
        doc_count = df["source_doc"].nunique()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📝 Text Entries", text_count)
        with col2:
            st.metric("🖼️ Image Entries", image_count)
        with col3:
            st.metric("📄 Documents", doc_count)

    # Search Input and Filters
    with st.sidebar:
        st.markdown("""
        <div style='margin-bottom: 1.5rem;'>
            <h2 style='color: #1e40af; font-weight: 600; font-size: 1.5rem; margin: 0;'>
                🔍 Search Configuration
            </h2>
        </div>
        """, unsafe_allow_html=True)
        
        query = st.text_input(
            "",
            value="transformer architecture attention mechanism",
            help="Describe what you're looking for in natural language",
            label_visibility="collapsed"
        )

        st.markdown("""
        <p style='font-size: 19px; color: #374151; font-weight: 500; margin-bottom: 0.5rem;'>
            Select search scope:
        </p>
        """, unsafe_allow_html=True)
        
        modality = st.selectbox(
            "",
            ["Text and Images", "Text Only", "Images Only"],
            help="Choose whether to search through text, images, or both",
            label_visibility="collapsed"
        )

        try:
            source_docs = ["All"] + sorted(df["source_doc"].unique().tolist())
        except Exception as e:
            st.error(f"Error loading documents: {e}")
            source_docs = ["All"]
            
        st.markdown("""
        <p style='font-size: 19px; color: #374151; font-weight: 500; margin-bottom: 0.5rem;'>
            Filter by document:
        </p>
        """, unsafe_allow_html=True)
        
        selected_doc = st.selectbox(
            "",
            source_docs,
            help="Limit search to a specific document or search all",
            label_visibility="collapsed"
        )
        
        with st.expander("⚙️ Advanced Options"):
            max_results = st.slider(
                "Maximum results per category:",
                min_value=3,
                max_value=20,
                value=10,
                help="Number of top results to show for text and images"
            )
            
            min_similarity = st.slider(
                "Minimum similarity threshold:",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.05,
                help="Filter out results below this similarity score"
            )
        
        search_button = st.button(
            "🔍 Search",
            type="primary",
            use_container_width=True,
            help="Click to start searching"
        )

    # Search Logic
    if search_button and query.strip():
        logger.info(f"Starting search with query: '{query}'")
        logger.info(f"Search parameters: modality='{modality}', doc='{selected_doc}', max_results={max_results}, min_similarity={min_similarity}")
        
        # Start timing the entire query
        query_start_time = time.time()
        
        with st.spinner("🔍 Searching through documents..."):
            if selected_doc != "All":
                search_df = df[df["source_doc"] == selected_doc].copy()
                logger.info(f"Filtered to document '{selected_doc}': {len(search_df)} entries")
            else:
                search_df = df.copy()
                logger.info(f"Searching across all documents: {len(search_df)} entries")

            text_df = search_df[search_df["modality"] == "text"].copy()
            image_df = search_df[search_df["modality"] == "image"].copy()
            logger.info(f"Search scope: {len(text_df)} text entries, {len(image_df)} image entries")

            text_results = []
            image_results = []

            # Generate query embedding with timing
            with st.spinner("🧠 Generating query embedding..."):
                embedding_start = time.time()
                try:
                    logger.info("Generating query embedding...")
                    query_embedding = get_text_embedding(query)
                    embedding_time = time.time() - embedding_start
                    logger.info(f"EMBEDDING_GENERATION - Query embedding generated in {embedding_time:.4f}s")
                except Exception as e:
                    logger.error(f"Error generating query embedding: {str(e)}")
                    st.error(f"❌ Error generating query embedding: {str(e)}")
                    st.stop()

            # Text retrieval with timing
            if "Text" in modality and not text_df.empty:
                logger.info(f"Starting text retrieval with {len(text_df)} entries...")
                text_search_start = time.time()
                try:
                    text_results = find_similar_texts(
                        query_embedding, text_df, top_n=max_results, query_text=query
                    )
                    text_results = [r for r in text_results if r["similarity"] >= min_similarity]
                    text_search_time = time.time() - text_search_start
                    logger.info(f"TEXT_RETRIEVAL - Completed with {len(text_results)} results in {text_search_time:.4f}s")
                except Exception as e:
                    logger.error(f"Error during text retrieval: {str(e)}")
                    st.error(f"❌ Error during text search: {str(e)}")

            # Image retrieval with timing
            if "Image" in modality and not image_df.empty:
                logger.info(f"Starting image retrieval with {len(image_df)} entries...")
                image_search_start = time.time()
                try:
                    image_results = find_similar_images(
                        query_embedding, image_df, top_n=max_results
                    )
                    image_results = [r for r in image_results if r["similarity"] >= min_similarity]
                    image_search_time = time.time() - image_search_start
                    logger.info(f"IMAGE_RETRIEVAL - Completed with {len(image_results)} results in {image_search_time:.4f}s")
                except Exception as e:
                    logger.error(f"Error during image retrieval: {str(e)}")
                    st.error(f"❌ Error during image search: {str(e)}")

            combined_results = text_results + image_results
            combined_results.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Calculate total query time
        total_query_time = time.time() - query_start_time
        
        logger.info(f"Combined results: {len(combined_results)} total entries")
        logger.info(f"QUERY_TIMING - Total query time: {total_query_time:.4f}s")
        logger.info("Search completed")

        st.markdown("""
        <div style='margin-top: 2rem; margin-bottom: 1rem;'>
            <h2 style='color: #1e40af; font-weight: 600; font-size: 1.8rem; margin: 0;'>
                🎯 Search Results
            </h2>
        </div>
        """, unsafe_allow_html=True)

        if not combined_results:
            st.warning("🔍 No results found. Try:")
            st.markdown("""
            - Using different keywords
            - Reducing the similarity threshold
            - Changing the search scope
            - Selecting 'All' documents
            """)
        else:
            text_count = sum(1 for r in combined_results if r["modality"] == "text")
            image_count = sum(1 for r in combined_results if r["modality"] == "image")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Results", len(combined_results))
            with col2:
                st.metric("📝 Text Results", text_count)
            with col3:
                st.metric("🖼️ Image Results", image_count)
                
            st.markdown("---")
            
            for idx, result in enumerate(combined_results, 1):
                with st.container():
                    col1, col2 = st.columns([1, 4])
                    
                    with col1:
                        if result["modality"] == "image":
                            st.markdown("""
                            <div style='background-color: #f0f9ff; padding: 1rem; border-radius: 8px; border-left: 4px solid #1e40af; text-align: center;'>
                                <p style='font-size: 17px; color: #1e40af; font-weight: 600; line-height: 1.7; margin: 0;'>
                                    🖼️ Image Result
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            if result.get("is_placeholder", False):
                                st.warning("🖼️ Placeholder Image")
                                st.caption("Referenced image not available")
                            else:
                                try:
                                    if os.path.exists(result["content"]):
                                        logger.debug(f"Loading image from path: {result['content']}")
                                        st.image(
                                            result["content"],
                                            caption=f"Page {result['page']}",
                                            use_column_width=True,
                                        )
                                    else:
                                        logger.warning(f"Image file not found: {result['content']}")
                                        st.error("🖼️ Image file not found")
                                except Exception as e:
                                    logger.error(f"Error loading image: {str(e)}", exc_info=True)
                                    st.error(f"🖼️ Error loading image: {str(e)[:50]}...")
                        else:
                            st.info("📝 Text Result")
                    
                    with col2:
                        similarity_color = "🟢" if result["similarity"] > 0.7 else "🟡" if result["similarity"] > 0.4 else "🔴"
                        st.markdown(f"""
                        <h3 style='color: #1e40af; font-weight: 600; margin-bottom: 0.5rem;'>
                            {similarity_color}
                        </h3>
                        """, unsafe_allow_html=True)
                        
                        col_sim, col_doc, col_page = st.columns(3)
                        with col_sim:
                            st.metric("Similarity", f"{result['similarity']:.3f}")
                        with col_doc:
                            st.markdown(f"""
                            <p style='font-size: 19px; color: #374151; font-weight: 500;'>
                                <strong style='color: #1e40af;'>Source:</strong> {result['source_doc']}
                            </p>
                            """, unsafe_allow_html=True)
                        with col_page:
                            st.markdown(f"""
                            <p style='font-size: 19px; color: #374151; font-weight: 500;'>
                                <strong style='color: #1e40af;'>Page:</strong> {result['page']}
                            </p>
                            """, unsafe_allow_html=True)
                        
                        if result["modality"] == "text":
                            content = result["content"]
                            if len(content) > 1000:
                                st.markdown(f"""
                                <div style='background-color: #f8fafc; padding: 1rem; border-radius: 8px; border-left: 4px solid #1e40af;'>
                                    <p style='font-size: 17px; color: #374151; line-height: 1.7; margin: 0;'>
                                        {content[:1000]}...
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)
                                with st.expander("Show full text"):
                                    st.markdown(f"""
                                    <div style='background-color: #f8fafc; padding: 1rem; border-radius: 8px;'>
                                        <p style='font-size: 17px; color: #374151; line-height: 1.7; margin: 0;'>
                                            {content}
                                        </p>
                                    </div>
                                    """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                <div style='background-color: #f8fafc; padding: 1rem; border-radius: 8px; border-left: 4px solid #1e40af;'>
                                    <p style='font-size: 17px; color: #374151; line-height: 1.7; margin: 0;'>
                                        {content}
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            caption = result.get('caption', 'N/A')
                            if caption != 'N/A':
                                st.markdown(f"""
                                <div style='background-color: #f0f9ff; padding: 1rem; border-radius: 8px; border-left: 4px solid #1e40af;'>
                                    <p style='font-size: 17px; color: #374151; line-height: 1.7; margin: 0;'>
                                        <strong style='color: #1e40af;'>Caption:</strong> {caption}
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                <div style='background-color: #f9fafb; padding: 1rem; border-radius: 8px; border-left: 4px solid #9ca3af;'>
                                    <p style='font-size: 17px; color: #6b7280; line-height: 1.7; margin: 0; font-style: italic;'>
                                        No caption available
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)
                    
                    st.markdown("---")

    elif search_button and not query.strip():
        logger.warning("Search attempted with empty query")
        st.warning("⚠️ Please enter a search query to begin searching.")
    
    # Help section when no search has been performed
    if not search_button:
        logger.info("Initial page load - displaying help information")
        st.info("� Configure your search in the sidebar and click **Search** to begin!")
        
        with st.expander("💡 How to use this search engine", expanded=True):
            st.markdown("""
            <div style='font-size: 16px; line-height: 1.6;'>
            
            <h3 style='color: #1e40af; font-weight: 600; margin-bottom: 1rem;'>Getting Started:</h3>
            <ol style='font-size: 16px; line-height: 1.7; color: #374151; margin-left: 1.5rem;'>
                <li style='margin-bottom: 0.5rem;'><strong>Enter a search query</strong> - Describe what you're looking for in natural language</li>
                <li style='margin-bottom: 0.5rem;'><strong>Choose search scope</strong> - Search text, images, or both</li>
                <li style='margin-bottom: 0.5rem;'><strong>Filter by document</strong> - Optionally limit to specific documents</li>
                <li style='margin-bottom: 0.5rem;'><strong>Adjust settings</strong> - Use advanced options for fine-tuning</li>
                <li style='margin-bottom: 0.5rem;'><strong>Click Search</strong> - View results ranked by semantic similarity</li>
            </ol>

            <h3 style='color: #1e40af; font-weight: 600; margin-top: 2rem; margin-bottom: 1rem;'>Tips for Better Results:</h3>
            <ul style='font-size: 16px; line-height: 1.7; color: #374151; margin-left: 1.5rem;'>
                <li style='margin-bottom: 0.5rem;'>Use descriptive, specific terms</li>
                <li style='margin-bottom: 0.5rem;'>Try both technical and common language</li>
                <li style='margin-bottom: 0.5rem;'>Experiment with different similarity thresholds</li>
                <li style='margin-bottom: 0.5rem;'>Use document filters for focused searches</li>
            </ul>
            
            </div>
            """, unsafe_allow_html=True)

else:
    # Show error state when data couldn't be loaded
    st.error("❌ Cannot start the application without data. Please check the error messages above.")
