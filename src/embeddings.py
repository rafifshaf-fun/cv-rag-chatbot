"""
Embedding model — thin wrapper so you can swap providers in one place.
"""

from langchain_community.embeddings import FastEmbedEmbeddings
from config import EMBEDDING_MODEL_NAME


def get_embeddings() -> FastEmbedEmbeddings:
    """Return the ONNX-based FastEmbed instance (no API key needed)."""
    return FastEmbedEmbeddings(model_name=EMBEDDING_MODEL_NAME)
