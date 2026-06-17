"""
FAISS vector-store — build from scratch or load a cached index.
"""

import os

from langchain_community.vectorstores import FAISS

from config import FAISS_PATH
from src.embeddings import get_embeddings
from src.ingestion import load_documents, split_documents


def build_vectorstore(force: bool = False):
    """
    (Re)build the FAISS index from source documents.

    Args:
        force: If True, rebuild even if a cached index exists.
    """
    embeddings = get_embeddings()
    docs = load_documents()
    chunks = split_documents(docs)

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(FAISS_PATH)
    print(f"✅ Vector store built with {len(chunks)} chunks.")
    return vectorstore


def load_vectorstore():
    """Load cached FAISS index, falling back to a fresh build."""
    embeddings = get_embeddings()
    if os.path.exists(FAISS_PATH):
        return FAISS.load_local(
            FAISS_PATH,
            embeddings,
            allow_dangerous_deserialization=True,
        )
    print("⚠️  No cached index found. Building from scratch...")
    return build_vectorstore()
