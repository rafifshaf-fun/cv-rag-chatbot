from rag_pipeline import build_vectorstore, get_embeddings
from langchain_community.vectorstores import FAISS

# Force rebuild
vs = build_vectorstore()

# Test retrieval directly
results = vs.similarity_search("projects built computer vision stock", k=6)
print(f"\n🔍 Found {len(results)} chunks:\n")
for i, doc in enumerate(results):
    print(f"--- Chunk {i+1} ---")
    print(doc.page_content[:200])
    print()