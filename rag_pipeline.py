"""
Backward-compatibility wrapper — re-exports from the modular src/ package.

New code should import directly from src.*, e.g.:
    from src.chain import get_chain
    from src.vectorstore import build_vectorstore, load_vectorstore
    from src.embeddings import get_embeddings
"""

# Environment & config (loads .env, st.secrets, disables LangSmith tracing)
import config  # noqa: F401 — side-effect import is intentional

from src.chain import get_chain  # noqa: F401, E402
from src.vectorstore import build_vectorstore, load_vectorstore  # noqa: F401, E402
from src.embeddings import get_embeddings  # noqa: F401, E402