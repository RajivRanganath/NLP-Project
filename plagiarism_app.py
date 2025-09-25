import streamlit as st
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, util
# --- Page Setup ---
st.set_page_config(page_title="Plagiarism Checker with Web Corpus", page_icon="📚", layout="wide")

# =============================================================================
# Core Functions
# =============================================================================

@st.cache_resource
def load_model():
    """Load the SentenceTransformer model once."""
    return SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_pdf(file_obj):
    """Extract text from an uploaded PDF file."""
    text = ""
    try:
        reader = PdfReader(file_obj)
        for page in reader.pages:
            text += (page.extract_text() or "") + "\n"
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
    return text

def chunk_text_into_sentences(text):
    """Split text into sentences and filter for meaningful length."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.split()) > 7]

@st.cache_data(ttl=3600) # Cache scraped data for 1 hour
def scrape_text_from_url(url):
    """Scrape visible text from a given URL."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()
        return ' '.join(soup.stripped_strings)
    except requests.exceptions.RequestException as e:
        st.warning(f"Could not fetch {url}. Reason: {e}")
        return ""

def check_plagiarism_semantically(user_text, reference_texts, model, threshold):
    """Checks for plagiarism sentence by sentence using a semantic model."""
    user_sentences = chunk_text_into_sentences(user_text)
    if not user_sentences or not reference_texts:
        return []

    # Combine all reference texts into one and then chunk
    combined_reference_text = " ".join(reference_texts)
    reference_sentences = chunk_text_into_sentences(combined_reference_text)
    if not reference_sentences:
        return []

    # Encode all sentences
    user_embeddings = model.encode(user_sentences, convert_to_tensor=True)
    reference_embeddings = model.encode(reference_sentences, convert_to_tensor=True)

    # Compute cosine similarities
    cosine_scores = util.pytorch_cos_sim(user_embeddings, reference_embeddings)

    matches = []
    for i in range(len(user_sentences)):
        max_score_for_sentence = cosine_scores[i].max()
        if max_score_for_sentence > threshold:
            matches.append({
                "sentence": user_sentences[i],
                "score": max_score_for_sentence.item()
            })
    return matches

# =============================================================================
# Streamlit User Interface
# =============================================================================

st.title("📚 Plagiarism Checker with Dynamic Web Corpus")
st.markdown("Build your reference library by scraping web pages, then check your document for plagiarism.")

model = load_model()

# --- Session State to store the library ---
if 'reference_library' not in st.session_state:
    st.session_state.reference_library = []

# --- Sidebar for Corpus Building ---
with st.sidebar:
    st.header("Build Your Reference Library")
    st.markdown("Enter URLs (one per line) to scrape and add to your reference corpus.")
    
    urls_input = st.text_area("Enter URLs:", height=150, value="https://en.wikipedia.org/wiki/Artificial_intelligence\nhttps://en.wikipedia.org/wiki/Machine_learning")
    
    if st.button("Scrape and Build Corpus"):
        urls = [url.strip() for url in urls_input.split('\n') if url.strip()]
        if not urls:
            st.warning("Please enter at least one URL.")
        else:
            with st.spinner("Scraping URLs and building library..."):
                scraped_texts = []
                progress_bar = st.progress(0)
                for i, url in enumerate(urls):
                    text = scrape_text_from_url(url)
                    if text:
                        scraped_texts.append(text)
                    progress_bar.progress((i + 1) / len(urls))
                
                st.session_state.reference_library = scraped_texts
                st.success(f"Successfully built a library from {len(scraped_texts)} URLs.")
    
    if st.session_state.reference_library:
        st.info(f"Your reference library currently contains {len(st.session_state.reference_library)} documents.")

# --- Main Page for Plagiarism Checking ---
st.header("Check Your Document")
uploaded_file = st.file_uploader("Upload a PDF file to check", type=['pdf'])
similarity_threshold = st.slider("Similarity Threshold (higher is stricter)", 0.60, 1.0, 0.75, 0.05)

if uploaded_file is not None:
    if not st.session_state.reference_library:
        st.error("Your reference library is empty. Please scrape some URLs using the sidebar first.")
    else:
        if st.button("Analyze Document"):
            with st.spinner("Extracting text from your document..."):
                user_document_text = extract_text_from_pdf(uploaded_file)
            
            if not user_document_text.strip():
                st.warning("Could not extract any text from the document.")
            else:
                st.success("Text extracted. Now comparing against your reference library...")
                
                with st.spinner("Analyzing for plagiarism..."):
                    matches = check_plagiarism_semantically(
                        user_document_text, 
                        st.session_state.reference_library, 
                        model, 
                        threshold=similarity_threshold
                    )

                st.header("Analysis Report")
                num_sentences = len(chunk_text_into_sentences(user_document_text))
                plagiarism_ratio = len(matches) / num_sentences if num_sentences > 0 else 0
                st.metric("Plagiarized Sentences Found", f"{len(matches)} ({plagiarism_ratio:.1%})")

                if matches:
                    st.error(f"Found {len(matches)} sentences highly similar to your reference library:")
                    for match in sorted(matches, key=lambda x: x['score'], reverse=True):
                        st.markdown(f"> **Similarity: {match['score']:.2f}** - *'{match['sentence']}'*")
                        st.write("---")
                else:

                    st.success("✅ No significant plagiarism detected.")
