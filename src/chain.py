"""
RAG chain — wires ingestion, retrieval, prompts, and LLM into one pipeline.
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory

from src.vectorstore import load_vectorstore
from src.retrieval import get_retriever
from src.llm import get_llm_with_fallback, get_session_history
from src.prompts import CONTEXTUALIZE_PROMPT, QA_PROMPT


def get_chain():
    """
    Build and return the full conversational RAG chain.

    Returns a RunnableWithMessageHistory that:
      1. Rephrases follow-up questions against chat history
      2. Retrieves relevant CV chunks via MMR
      3. Answers using the retrieved context
    """
    vectorstore = load_vectorstore()
    retriever = get_retriever(vectorstore)
    llm = get_llm_with_fallback()

    contextualize_chain = CONTEXTUALIZE_PROMPT | llm | StrOutputParser()
    answer_chain = QA_PROMPT | llm | StrOutputParser()

    # ── Core RAG logic ─────────────────────────────────────────────────
    def rag_response(input_dict: dict) -> dict:
        chat_history = input_dict.get("chat_history", [])
        user_input = input_dict["input"]

        # Rephrase if there's prior conversation, else use the raw question
        if chat_history:
            standalone_question = contextualize_chain.invoke(
                {"input": user_input, "chat_history": chat_history}
            )
        else:
            standalone_question = user_input

        docs = retriever.invoke(standalone_question)
        context_text = "\n\n".join(doc.page_content for doc in docs)

        print(f"\n[DEBUG] Question: {user_input}")
        print(f"[DEBUG] Context preview:\n{context_text[:600]}\n")

        answer = answer_chain.invoke(
            {"input": user_input, "chat_history": chat_history, "context": context_text}
        )

        return {"answer": answer, "context": docs}

    # ── Wrap with message history ──────────────────────────────────────
    return RunnableWithMessageHistory(
        RunnableLambda(rag_response),
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
