import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import os
import traceback
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, util
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer

# =============================================================================
# FINAL FIX: Setup Function for NLTK Data
# This function runs first to ensure NLTK data is available on the server.
# =============================================================================
def setup_nltk():
    """
    Downloads the necessary NLTK data packages if they are not already present.
    """
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        st.info("Downloading NLTK 'punkt_tab' data...")
        nltk.download('punkt_tab')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        st.info("Downloading NLTK 'stopwords' data...")
        nltk.download('stopwords')

# --- Run the setup function once at the start of the script ---
setup_nltk()

# =============================================================================
# Engine 1: TF-IDF Local Checker - Class Definitions
# =============================================================================
class PlagiarismDetector:
    def __init__(self, threshold=0.65):
        self.threshold = threshold
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.reference_texts = []
        self.reference_metadata = []
    def preprocess_text(self, text):
        text = re.sub(r'\s+', ' ', text.strip()); text = re.sub(r'[^\w\s.!?]', '', text); return text
    def extract_text_from_pdf(self, file):
        try:
            reader = PdfReader(file); text = ""
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                if page_text.strip(): text += f"\n--- Page {i+1} ---\n" + page_text
            return self.preprocess_text(text)
        except Exception as e: raise Exception(f"Error reading PDF: {str(e)}")
    def get_sentences(self, text):
        sentences = sent_tokenize(text)
        return [s.strip() for s in sentences if len(s.strip()) > 20 and not s.startswith('--- Page')]
    def get_semantic_fingerprint(self, text):
        words = word_tokenize(text.lower())
        filtered_words = [self.stemmer.stem(w) for w in words if w.isalpha() and w not in self.stop_words]
        return ' '.join(sorted(set(filtered_words)))
    def load_reference_texts(self, folder="reference_pdfs"):
        self.reference_texts = []; self.reference_metadata = []
        if not os.path.exists(folder): return []
        for filename in os.listdir(folder):
            if filename.endswith((".pdf", ".txt")):
                try:
                    path = os.path.join(folder, filename)
                    if filename.endswith(".pdf"):
                        with open(path, 'rb') as f: text = self.extract_text_from_pdf(f)
                    else:
                        with open(path, 'r', encoding='utf-8') as f: text = self.preprocess_text(f.read())
                    self.reference_texts.append(text)
                    self.reference_metadata.append({'filename': filename, 'word_count': len(text.split()), 'char_count': len(text)})
                except Exception as e: print(f"Warning: Could not load {filename}: {str(e)}")
        return self.reference_texts
    def advanced_similarity_check(self, text1, text2):
        try:
            vec = TfidfVectorizer(stop_words='english', ngram_range=(1,3), max_features=10000); mat = vec.fit_transform([text1, text2]); cos_sim = cosine_similarity(mat[0:1], mat[1:2])[0][0]
            fp1, fp2 = self.get_semantic_fingerprint(text1), self.get_semantic_fingerprint(text2)
            if fp1 and fp2: fp_mat = TfidfVectorizer().fit_transform([fp1, fp2]); sem_sim = cosine_similarity(fp_mat[0:1], fp_mat[1:2])[0][0]
            else: sem_sim = 0.0
            w1, w2 = set(text1.lower().split()), set(text2.lower().split()); jac_sim = len(w1 & w2) / len(w1 | w2) if w1 or w2 else 0.0
            return min(0.5 * cos_sim + 0.3 * sem_sim + 0.2 * jac_sim, 1.0)
        except: return 0.0
    def plagiarism_check(self, uploaded_text):
        if not self.reference_texts: self.load_reference_texts()
        if not self.reference_texts: return [], {"error": "No reference documents found"}
        uploaded_sentences = self.get_sentences(uploaded_text)
        results, max_sim = [], 0.0; flagged = 0
        for i, sentence in enumerate(uploaded_sentences):
            if len(sentence.strip()) < 10: continue
            best_match = {'similarity':0.0, 'source':None, 'source_sentence':None}
            for idx, ref_text in enumerate(self.reference_texts):
                ref_sentences = self.get_sentences(ref_text)
                for ref_sentence in ref_sentences[:200]:
                    sim = self.advanced_similarity_check(sentence, ref_sentence)
                    if sim > best_match['similarity']: best_match.update({'similarity': sim, 'source': self.reference_metadata[idx]['filename'], 'source_sentence': ref_sentence[:100]+"..."})
            is_plag = best_match['similarity'] >= self.threshold;
            if is_plag: flagged +=1
            max_sim = max(max_sim, best_match['similarity'])
            results.append({'sentence': sentence, 'similarity': best_match['similarity'], 'is_plagiarized': is_plag, 'source': best_match['source'], 'source_sentence': best_match['source_sentence'], 'sentence_index': i+1})
        stats = {'total_sentences': len(uploaded_sentences), 'flagged_sentences': flagged, 'plagiarism_percentage': flagged / len(uploaded_sentences) *100 if uploaded_sentences else 0, 'max_similarity': max_sim, 'avg_similarity': np.mean([r['similarity'] for r in results]) if results else 0, 'reference_count': len(self.reference_texts), 'word_count': len(uploaded_text.split())}
        return results, stats

class TextParaphraser:
    def __init__(self): self.stemmer = PorterStemmer()
    def sentence_restructure_paraphrase(self, text):
        sentences = sent_tokenize(text); paraphrased = []
        for s in sentences:
            words = word_tokenize(s)
            if len(words) > 5 and ',' in s: 
                parts = s.split(',')
                if len(parts) >=2: paraphrased.append(f"{parts[1].strip()}, {parts[0].strip()}"); continue
            replacements = {'however':'nevertheless','therefore':'consequently','because':'since', 'although':'while','important':'significant','show':'demonstrate', 'find':'discover','use':'utilize','help':'assist','big':'large','small':'minor'}
            mod = s.lower()
            for old,new in replacements.items(): mod = mod.replace(old,new)
            paraphrased.append(mod.capitalize())
        return ' '.join(paraphrased)

# =============================================================================
# Engine 2: Semantic Web Checker - Core Functions
# =============================================================================

@st.cache_resource
def load_semantic_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def semantic_extract_text_from_pdf(file_obj):
    text = ""; reader = PdfReader(file_obj)
    for page in reader.pages: text += (page.extract_text() or "") + "\n"
    return text

def semantic_chunk_text_into_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.split()) > 7]

@st.cache_data(ttl=3600)
def scrape_text_from_url(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}; response = requests.get(url, headers=headers, timeout=10); response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        for element in soup(["script", "style", "nav", "footer", "header"]): element.decompose()
        return ' '.join(soup.stripped_strings)
    except: return ""

def semantic_check_plagiarism(user_text, reference_texts, model, threshold):
    user_sentences = semantic_chunk_text_into_sentences(user_text)
    if not user_sentences or not reference_texts: return []
    combined_ref_text = " ".join(reference_texts); ref_sentences = semantic_chunk_text_into_sentences(combined_ref_text)
    if not ref_sentences: return []
    user_embeds = model.encode(user_sentences, convert_to_tensor=True); ref_embeds = model.encode(ref_sentences, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(user_embeds, ref_embeds)
    matches = []
    for i in range(len(user_sentences)):
        max_score = scores[i].max()
        if max_score > threshold: matches.append({"sentence": user_sentences[i], "score": max_score.item()})
    return matches

# =============================================================================
# Streamlit User Interface
# =============================================================================
st.set_page_config(page_title="Plagiarism & Paraphraser", layout="wide")
st.title("🎓 Universal Plagiarism & Paraphraser Dashboard")

# --- Engine Selection ---
engine_choice = st.sidebar.radio(
    "Select Plagiarism Engine:",
    ('TF-IDF Local Checker', 'Semantic Web Checker'),
    help="TF-IDF is faster and works offline with local files. Semantic is more powerful and checks against scraped web pages."
)

st.sidebar.markdown("---")

if engine_choice == 'TF-IDF Local Checker':
    # --- UI for TF-IDF Local Checker ---
    st.sidebar.header("🔍 Debug Info")
    st.sidebar.write(f"Current directory: {os.getcwd()}")
    @st.cache_resource
    def get_detector():
        detector = PlagiarismDetector(threshold=0.65); detector.load_reference_texts(); return detector, len(detector.reference_texts)
    @st.cache_resource
    def get_paraphraser(): return TextParaphraser()
    detector, ref_count = get_detector(); paraphraser = get_paraphraser()
    st.sidebar.write(f"Reference documents found: {ref_count}")
    st.sidebar.subheader("⚙️ Processing Options")
    max_sentences = st.sidebar.slider("Max sentences to analyze", 10, 100, 50); chunk_size = st.sidebar.slider("Text chunk size (chars)", 1000, 5000, 2000)
    if "plag_stats" not in st.session_state: st.session_state.plag_stats = None
    if "plag_results" not in st.session_state: st.session_state.plag_results = None
    tab1, tab2, tab3 = st.tabs(["📄 PDF Plagiarism", "✍️ Paraphrase", "📊 Analytics"])
    with tab1:
        st.header("Upload PDF or TXT to Check Plagiarism (TF-IDF Method)")
        st.info(f"Using local file corpus: {ref_count} documents found in 'reference_pdfs' folder.")
        uploaded_file = st.file_uploader("Choose a PDF or TXT", type=['pdf', 'txt'], key="tfidf_uploader")
        if uploaded_file and detector:
            if uploaded_file.size > 10*1024*1024: st.error("❌ Max file size: 10 MB")
            elif st.button("🔍 Start Analysis", type="primary", key="tfidf_button"):
                progress_bar = st.progress(0); status_text = st.empty()
                try:
                    status_text.text("Step 1/4: Extracting text…"); progress_bar.progress(25)
                    if uploaded_file.name.lower().endswith(".pdf"): text = detector.extract_text_from_pdf(uploaded_file)
                    else: uploaded_file.seek(0); text = str(uploaded_file.read(), "utf-8")
                    status_text.text("Step 2/4: Processing text…"); progress_bar.progress(50)
                    if len(text) > chunk_size: text = text[:chunk_size]; st.warning(f"⚠️ Text truncated.")
                    with st.expander("📄 Text Preview"): st.text_area("Preview", text[:1000], height=200, key="tfidf_preview")
                    status_text.text("Step 3/4: Checking plagiarism…"); progress_bar.progress(75)
                    original_sentences = detector.get_sentences(text)
                    if len(original_sentences) > max_sentences: st.warning(f"Analyzing first {max_sentences} sentences."); text_for_check = '. '.join(original_sentences[:max_sentences])
                    else: text_for_check = text
                    results, stats = detector.plagiarism_check(text_for_check)
                    st.session_state.plag_stats = stats; st.session_state.plag_results = results
                    status_text.text("Step 4/4: Displaying results…"); progress_bar.progress(100); st.success("✅ Analysis completed!")
                    progress_bar.empty(); status_text.empty()
                    st.subheader("📊 Results Summary")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Plagiarism %", f"{stats['plagiarism_percentage']:.1f}%"); c2.metric("Max Similarity", f"{stats['max_similarity']*100:.1f}%"); c3.metric("Flagged/Total", f"{stats['flagged_sentences']}/{stats['total_sentences']}"); c4.metric("Avg Similarity", f"{stats['avg_similarity']*100:.1f}%")
                    if stats['flagged_sentences'] > 0:
                        st.subheader("🚨 Flagged Sentences")
                        flagged = [r for r in results if r['is_plagiarized']]
                        df = pd.DataFrame([{'Sentence #': r['sentence_index'], 'Similarity %': f"{r['similarity']*100:.1f}%", 'Source': r['source'], 'Sentence': r['sentence'][:100] + "…"} for r in flagged])
                        st.dataframe(df, use_container_width=True)
                    else: st.success("🎉 No significant plagiarism detected!"); st.balloons()
                except Exception as e: st.error(f"❌ Error: {str(e)}"); st.code(traceback.format_exc())
    with tab2:
        st.header("Text Paraphrasing Tool")
        input_method = st.radio("Choose input method:", ["Type/Paste Text", "Upload Text File"], key="para_radio")
        input_text = ""
        if input_method == "Type/Paste Text":
            input_text = st.text_area("Enter text to paraphrase", height=200, key="para_text_area")
        else:
            up_txt = st.file_uploader("Upload text file", type=['txt'], key="para_uploader")
            if up_txt:
                input_text = str(up_txt.read(), "utf-8")
                st.text_area("Uploaded preview", input_text[:500], height=150, key="para_preview")
        if st.button("🔄 Paraphrase Text", type="primary", key="para_button") and input_text:
            if len(input_text.split()) > 2000: st.error("❌ Max 2000 words.")
            else:
                with st.spinner("Paraphrasing…"):
                    out = paraphraser.sentence_restructure_paraphrase(input_text); out = '. '.join(s.strip().capitalize() for s in out.split('. '))
                    col1, col2 = st.columns(2)
                    with col1: st.text_area("Original", input_text, height=300, key="para_orig")
                    with col2: st.text_area("Paraphrased", out, height=300, key="para_para")
                    st.download_button("📥 Download", out, file_name=f"paraphrased_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with tab3:
        st.header("📊 Analytics Dashboard")
        if not st.session_state.plag_stats or not st.session_state.plag_results: st.info("🔎 No analysis data yet. Run a plagiarism check in Tab 1.")
        else:
            stats = st.session_state.plag_stats; results = st.session_state.plag_results
            st.subheader("Overall Plagiarism"); donut = px.pie(values=[stats['plagiarism_percentage'], 100 - stats['plagiarism_percentage']], names=["Plagiarized", "Unique"], hole=0.6, color_discrete_sequence=["#e74c3c", "#2ecc71"]); donut.update_traces(textinfo="percent+label"); st.plotly_chart(donut, use_container_width=True)
            st.subheader("Per-Sentence Similarity"); df_sent = pd.DataFrame({"Sentence #": [r['sentence_index'] for r in results], "Similarity": [r['similarity'] for r in results]}); bar = px.bar(df_sent, x="Sentence #", y="Similarity", color="Similarity", color_continuous_scale="Reds"); bar.update_layout(yaxis=dict(tickformat=".0%")); st.plotly_chart(bar, use_container_width=True)
            flagged = [r for r in results if r['is_plagiarized']]
            if flagged:
                st.subheader("Flagged Sentences (interactive)"); df_flag = pd.DataFrame([{"Sentence": r["sentence"],"Similarity %": round(r["similarity"]*100,1),"Source": r["source"] or "N/A"} for r in flagged]); st.dataframe(df_flag.style.background_gradient(subset=["Similarity %"], cmap="Reds"), use_container_width=True)

else: # This block is for the 'Semantic Web Checker'
    # --- UI for Semantic Web Checker ---
    model = load_semantic_model()
    if 'reference_library' not in st.session_state: st.session_state.reference_library = []
    with st.sidebar:
        st.header("Build Your Web Reference Library")
        st.markdown("Enter URLs (one per line) to scrape.")
        urls_input = st.text_area("Enter URLs:", height=150, value="https://en.wikipedia.org/wiki/Artificial_intelligence\nhttps://en.wikipedia.org/wiki/Machine_learning", key="semantic_urls")
        if st.button("Scrape and Build Corpus", key="semantic_scrape_button"):
            urls = [url.strip() for url in urls_input.split('\n') if url.strip()]
            if not urls: st.warning("Please enter at least one URL.")
            else:
                with st.spinner("Scraping URLs..."):
                    scraped_texts = [scrape_text_from_url(url) for url in urls if scrape_text_from_url(url)]
                    st.session_state.reference_library = scraped_texts
                    st.success(f"Built library from {len(scraped_texts)} URLs.")
        if st.session_state.reference_library: st.info(f"Reference library contains {len(st.session_state.reference_library)} documents.")
    st.header("Check Document Against Web Corpus (Semantic Method)")
    uploaded_file = st.file_uploader("Upload a PDF file to check", type=['pdf'], key="semantic_uploader")
    similarity_threshold = st.sidebar.slider("Similarity Threshold", 0.60, 1.0, 0.75, 0.05, key="semantic_slider")
    if uploaded_file:
        if not st.session_state.reference_library: st.error("Reference library is empty. Please scrape URLs first.")
        elif st.button("Analyze Document", key="semantic_button"):
            with st.spinner("Analyzing document..."):
                user_document_text = semantic_extract_text_from_pdf(uploaded_file)
                if not user_document_text.strip(): st.warning("Could not extract text from document.")
                else:
                    matches = semantic_check_plagiarism(user_document_text, st.session_state.reference_library, model, threshold=similarity_threshold)
                    st.header("Analysis Report")
                    num_sents = len(semantic_chunk_text_into_sentences(user_document_text))
                    plag_ratio = len(matches) / num_sents if num_sents > 0 else 0
                    st.metric("Plagiarized Sentences Found", f"{len(matches)} ({plag_ratio:.1%})")
                    if matches:
                        st.error("Found highly similar sentences:")
                        for match in sorted(matches, key=lambda x: x['score'], reverse=True): st.markdown(f"> **Similarity: {match['score']:.2f}** - *'{match['sentence']}'*"); st.write("---")
                    else: st.success("✅ No significant plagiarism detected.")

# Footer
st.markdown("---")
st.caption(f"Dashboard updated: {datetime.now().strftime('%Y%m%d_%H%M%S')}")
