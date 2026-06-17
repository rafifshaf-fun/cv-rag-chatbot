"""
Central configuration for the RAG CV Chatbot.
All tunable constants live here so nothing is buried in pipeline code.
"""

import os
from dotenv import load_dotenv

# ── Environment setup (runs once on first import) ──────────────────────
load_dotenv()
try:
    import streamlit as st
    for key, value in st.secrets.items():
        os.environ[key] = str(value)
except Exception:
    pass

os.environ["LANGCHAIN_TRACING_V2"] = "false"

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_PATH = os.path.join(BASE_DIR, "faiss_index")
CV_FILES = [
    os.path.join(BASE_DIR, "data", "my_cv.md"),
    os.path.join(BASE_DIR, "data", "about_me.md"),
]

# ── Models ─────────────────────────────────────────────────────────────
PRIMARY_MODEL = "gemini-3-flash-preview"
FALLBACK_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# ── Chunking ───────────────────────────────────────────────────────────
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
CHUNK_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# ── Retrieval ──────────────────────────────────────────────────────────
RETRIEVER_K = 8
RETRIEVER_FETCH_K = 25
RETRIEVER_LAMBDA_MULT = 0.6

# ── LLM ────────────────────────────────────────────────────────────────
LLM_TEMPERATURE = 0.3
