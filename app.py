import re
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

DATA_PATH = r"C:\Projects\IR System\medquad.csv"

STOPWORDS = set(ENGLISH_STOP_WORDS)

# =========================
# TEXT UTILS
# =========================
def clean_text(x: str) -> str:
    x = str(x).lower()
    x = re.sub(r"\s+", " ", x).strip()
    x = re.sub(r"[^a-z0-9\s\.\,\?\!\-\%/]", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x

def tokenize(text: str):
    # Regex tokenizer (no NLTK)
    text = clean_text(text)
    toks = re.findall(r"[a-z0-9]+", text)
    return [t for t in toks if t not in STOPWORDS and len(t) > 2]

# =========================
# DATA LOADER
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)

    required = {"question", "answer", "source", "focus_area"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}. Required: {required}")

    df = df.copy()

    # Replace blanks -> NA, then drop missing core fields
    df.replace(r"^\s*$", pd.NA, regex=True, inplace=True)
    df = df.dropna(subset=["question", "answer"])

    # Fill metadata if missing
    df["source"] = df["source"].fillna("unknown")
    df["focus_area"] = df["focus_area"].fillna("general")

    # Ensure string dtype
    for c in required:
        df[c] = df[c].astype(str)

    # Cleaning fields
    df["question_clean"] = df["question"].apply(clean_text)
    df["answer_clean"] = df["answer"].apply(clean_text)

    # Retrieval docs
    df["doc_q"] = df["question_clean"]
    df["doc_qa"] = df["question_clean"] + " " + df["answer_clean"]

    # Remove duplicates (good IR practice)
    df = df.drop_duplicates(subset=["question", "answer"]).reset_index(drop=True)

    return df

# =========================
# MODELS
# =========================
class TFIDF:
    def __init__(self):
        self.vec = TfidfVectorizer(stop_words="english", max_features=50000, ngram_range=(1, 2))
        self.mat = None

    def fit(self, docs):
        self.mat = self.vec.fit_transform(docs)

    def search(self, query, top_k=10):
        qv = self.vec.transform([clean_text(query)])
        scores = cosine_similarity(qv, self.mat).flatten()
        idx = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in idx]

class BM25:
    def __init__(self):
        self.model = None
        self.tdocs = None

    def fit(self, docs):
        self.tdocs = [tokenize(d) for d in docs]
        self.model = BM25Okapi(self.tdocs)

    def search(self, query, top_k=10):
        q = tokenize(query)
        scores = np.array(self.model.get_scores(q), dtype=float)
        idx = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in idx]

    def all_scores(self, query):
        q = tokenize(query)
        return np.array(self.model.get_scores(q), dtype=float)

class SBERT:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
        self.emb = None

    def fit(self, docs):
        self.emb = self.model.encode(
            docs,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        ).astype(np.float32)

    def search(self, query, top_k=10):
        q = self.model.encode([clean_text(query)], convert_to_numpy=True, normalize_embeddings=True)[0].astype(np.float32)
        scores = np.dot(self.emb, q)
        idx = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in idx]

    def all_scores(self, query):
        q = self.model.encode([clean_text(query)], convert_to_numpy=True, normalize_embeddings=True)[0].astype(np.float32)
        return np.dot(self.emb, q)

def minmax_norm(x):
    x = np.array(x, dtype=float)
    if x.max() == x.min():
        return np.ones_like(x)
    return (x - x.min()) / (x.max() - x.min())

class HYBRID:
    def __init__(self, bm25, sbert, alpha=0.4):
        self.bm25 = bm25
        self.sbert = sbert
        self.alpha = alpha

    def search(self, query, top_k=10):
        b = minmax_norm(self.bm25.all_scores(query))
        s = minmax_norm(self.sbert.all_scores(query))
        fused = self.alpha * b + (1 - self.alpha) * s
        idx = np.argsort(fused)[::-1][:top_k]
        return [(int(i), float(fused[i])) for i in idx]

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="MediSearch", page_icon="🩺", layout="wide")
st.title("🩺 MediSearch: Health Information Retrieval (CSC-785)")
st.caption("MedQuAD-based IR system. Not medical advice.")

# Load data
df = load_data()

with st.sidebar:
    st.header("Settings")
    model_choice = st.selectbox("Retrieval Model", ["HYBRID", "BM25", "SBERT", "TF-IDF"])
    top_k = st.slider("Top K", 3, 20, 10)
    alpha = st.slider("Hybrid alpha (BM25 weight)", 0.0, 1.0, 0.4, 0.05)

@st.cache_resource
def build_models():
    tfidf = TFIDF()
    tfidf.fit(df["doc_qa"].tolist())

    bm25 = BM25()
    bm25.fit(df["doc_q"].tolist())

    sbert = SBERT("all-MiniLM-L6-v2")
    sbert.fit(df["doc_q"].tolist())

    hybrid = HYBRID(bm25, sbert, alpha=0.4)
    return tfidf, bm25, sbert, hybrid

tfidf, bm25, sbert, hybrid = build_models()
hybrid.alpha = alpha

query = st.text_input("Enter your medical question:")

if st.button("Search") and query.strip():
    if model_choice == "TF-IDF":
        results = tfidf.search(query, top_k)
    elif model_choice == "BM25":
        results = bm25.search(query, top_k)
    elif model_choice == "SBERT":
        results = sbert.search(query, top_k)
    else:
        results = hybrid.search(query, top_k)

    st.subheader("Top Results")
    for r, (i, score) in enumerate(results, start=1):
        row = df.iloc[i]
        with st.expander(f"{r}. score={score:.4f} | {row['question']}", expanded=(r == 1)):
            st.markdown("**Answer:**")
            st.write(row["answer"])
            st.write("**Source:**", row["source"])
            st.write("**Focus Area:**", row["focus_area"])
