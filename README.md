# Universal Plagiarism & Paraphraser Dashboard

A powerful Streamlit-based web application that detects plagiarism and paraphrases text. It features two core engines:
1. **TF-IDF Local Checker:** Compares uploaded documents against a local corpus of PDFs using TF-IDF and Jaccard similarity.
2. **Semantic Web Checker:** Scrapes live URLs to build a custom web corpus and uses a SentenceTransformer (`all-MiniLM-L6-v2`) to detect semantic similarities in meaning.

## How to Run Locally

1. Install the required dependencies:
   ```bash
   pip install streamlit pandas plotly sentence-transformers beautifulsoup4 scikit-learn PyPDF2 nltk faiss-cpu
   ```
2. Run the Streamlit application:
   ```bash
   streamlit run "plagiarism_app (1).py"
   ```

## Live Application
🔗 [View the Live App Here](https://nlp-project-dkpqfn4ivbr3mfjjkb9h6m.streamlit.app/) 
