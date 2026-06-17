"""
Retriever — MMR-based retrieval configured from central settings.
"""

from config import RETRIEVER_K, RETRIEVER_FETCH_K, RETRIEVER_LAMBDA_MULT


def get_retriever(vectorstore):
    """
    Wrap a FAISS vectorstore in an MMR retriever.

    MMR balances relevance with diversity so you get
    different sections of the CV rather than near-duplicate chunks.
    """
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": RETRIEVER_K,
            "fetch_k": RETRIEVER_FETCH_K,
            "lambda_mult": RETRIEVER_LAMBDA_MULT,
        },
    )
