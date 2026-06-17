"""
Document loading & chunking.
Update CV_FILES in config.py to add new data sources.
"""

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CV_FILES, CHUNK_SIZE, CHUNK_OVERLAP, CHUNK_SEPARATORS


def load_documents(filepaths: list[str] | None = None):
    """Load plain-text / markdown files and return LangChain Documents."""
    paths = filepaths or CV_FILES
    docs = []
    for filepath in paths:
        loader = TextLoader(filepath, encoding="utf-8")
        docs.extend(loader.load())
    return docs


def split_documents(docs):
    """Split documents into overlapping chunks for retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=CHUNK_SEPARATORS,
    )
    return splitter.split_documents(docs)
