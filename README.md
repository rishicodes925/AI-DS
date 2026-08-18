# Import required libraries
import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch
import os

# Set page title and configuration
st.set_page_config(
    page_title="Session Summary Search",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to improve the appearance
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .result-box {
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    h1, h2, h3 {
        color: #1E88E5;
    }
</style>
""", unsafe_allow_html=True)

# Application title
st.title("Session Summary Search Tool")
st.markdown("Enter keywords to find the most relevant session summaries.")

# Function to load data
@st.cache_data
def load_data():
    """Load the dataset and cache it for performance"""
    try:
        df = pd.read_excel("Session-Summary-for-E6-project.xlsx")
        # Drop the SerialNo column if it exists
        if 'SerialNo' in df.columns:
            df = df.drop('SerialNo', axis=1)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Function to load BERT model and cache it
@st.cache_resource
def load_model():
    """Load the BERT model and cache it"""
    try:
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to get document embeddings
@st.cache_data
def get_document_embeddings(_model, documents):
    """Create and cache document embeddings"""
    try:
        return _model.encode(documents, convert_to_tensor=True)
    except Exception as e:
        st.error(f"Error creating embeddings: {e}")
        return None

# Function to search for relevant documents
def search_documents(query, model, doc_embeddings, documents, top_k=3):
    """Search for documents relevant to the query keywords"""
    try:
        # Encode the query
        query_embedding = model.encode(query, convert_to_tensor=True)
        
        # Calculate cosine similarity between query and all documents
        cos_scores = util.cos_sim(query_embedding, doc_embeddings)[0]
        
        # Get top-k results
        top_results = torch.topk(cos_scores, k=min(top_k, len(documents)))
        
        results = []
        for score, idx in zip(top_results[0], top_results[1]):
            results.append({
                "score": score.item(),
                "summary": documents[idx],
                "index": idx.item()
            })
            
        return results
    except Exception as e:
        st.error(f"Error during search: {e}")
        return []

# Main application logic
def main():
    # Load data
    df = load_data()
    if df is None:
        st.warning("Please make sure your Excel file 'Session-Summary-for-E6-project.xlsx' is in the current directory.")
        return
    
    # Load BERT model
    model = load_model()
    if model is None:
        st.warning("Failed to load the BERT model. Please check your internet connection.")
        return
    
    # Extract documents
    documents = df['Session_Summary'].astype(str).tolist()
    
    # Get document embeddings (cached)
    doc_embeddings = get_document_embeddings(model, documents)
    if doc_embeddings is None:
        return
    
    # Sidebar for input
    st.sidebar.header("Search Options")
    
    # Input for keywords
    keywords = st.sidebar.text_area(
        "Enter keywords (separated by spaces or commas):",
        height=100,
        help="Enter keywords that describe your topic of interest"
    )
    
    # Search button
    search_button = st.sidebar.button("Search")
    
    # Process search when button is clicked
    if search_button and keywords:
        # Clean and prepare keywords
        keywords = keywords.replace(',', ' ')
        
        st.subheader("Search Results")
        with st.spinner("Searching for relevant session summaries..."):
            # Get search results
            results = search_documents(keywords, model, doc_embeddings, documents)
            
            if results:
                for i, result in enumerate(results):
                    similarity = result["score"] * 100  # Convert to percentage
                    
                    # Display result in a box
                    st.markdown(f"""
                    <div class="result-box">
                        <h3>Result {i+1} - Relevance: {similarity:.2f}%</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display the full summary
                    st.text_area(
                        f"Summary {i+1}",
                        value=result["summary"],
                        height=200,
                        key=f"summary_{i}"
                    )
                    
                    st.markdown("---")
            else:
                st.info("No relevant results found. Try different keywords.")
    elif search_button:
        st.warning("Please enter some keywords to search.")
    else:
        st.info("Enter keywords in the sidebar and click 'Search' to find relevant session summaries.")

# Run the app
if __name__ == "__main__":
    main()
