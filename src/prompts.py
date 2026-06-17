"""
Prompt templates used by the RAG pipeline.
Edit these in one place to change the bot's behaviour.
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ── Prompt that turns a chatty question into a standalone query ─────────
CONTEXTUALIZE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given the chat history and the latest user question, "
            "rephrase it as a standalone question. Do NOT answer it. "
            "Just return the rephrased question as plain text.",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# ── Prompt that answers questions using retrieved context ──────────────
QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an AI assistant representing Rafif Shafwan's CV and personal profile. "
            "Use the context below to answer questions about his skills, experience, education, "
            "projects, and background. Be specific — quote project names, tools, and details "
            "from the context whenever possible. "
            "Only say you don't know if the context genuinely contains no relevant information.\n\n"
            "Context:\n{context}",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
