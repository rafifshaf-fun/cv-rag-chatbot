import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable

load_dotenv()
try:
    for key, value in st.secrets.items():
        os.environ[key] = str(value)
except Exception:
    pass

os.environ["LANGCHAIN_TRACING_V2"] = "false"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_PATH = os.path.join(BASE_DIR, "faiss_index")
CV_FILES = [
    os.path.join(BASE_DIR, "data", "my_cv.md"),
    os.path.join(BASE_DIR, "data", "about_me.md"),
]

PRIMARY_MODEL = "gemini-3-flash-preview"
FALLBACK_MODEL = "gemini-2.5-flash"

store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def get_embeddings():
    return FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")


def build_vectorstore():
    docs = []
    for filepath in CV_FILES:
        loader = TextLoader(filepath, encoding="utf-8")
        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(docs)

    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(FAISS_PATH)
    print(f"Vector store built with {len(chunks)} chunks.")
    return vectorstore


def load_vectorstore():
    embeddings = get_embeddings()
    if os.path.exists(FAISS_PATH):
        return FAISS.load_local(
            FAISS_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
    return build_vectorstore()


def make_llm(model_name: str) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.3
    )


def invoke_with_fallback(chain, inputs: dict) -> str:
    try:
        return chain.invoke(inputs)
    except (ResourceExhausted, ServiceUnavailable) as e:
        print(f"[WARN] {PRIMARY_MODEL} rate limited: {e}. Falling back to {FALLBACK_MODEL}.")
        raise


def get_chain():
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 8, "fetch_k": 25, "lambda_mult": 0.6}
    )

    primary_llm = make_llm(PRIMARY_MODEL)
    fallback_llm = make_llm(FALLBACK_MODEL)

    # Both chains share the same prompts
    contextualize_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Given the chat history and the latest user question, "
            "rephrase it as a standalone question. Do NOT answer it. "
            "Just return the rephrased question as plain text."
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    qa_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an AI assistant representing Rafif Shafwan's CV and personal profile. "
            "Use the context below to answer questions about his skills, experience, education, "
            "projects, and background. Be specific — quote project names, tools, and details "
            "from the context whenever possible. "
            "Only say you don't know if the context genuinely contains no relevant information.\n\n"
            "Context:\n{context}"
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    # LangChain's built-in .with_fallbacks() handles automatic fallback
    llm = primary_llm.with_fallbacks(
        [fallback_llm],
        exceptions_to_handle=(ResourceExhausted, ServiceUnavailable)
    )

    contextualize_chain = contextualize_prompt | llm | StrOutputParser()
    answer_chain = qa_prompt | llm | StrOutputParser()

    def rag_response(input_dict: dict) -> dict:
        chat_history = input_dict.get("chat_history", [])
        user_input = input_dict["input"]

        if chat_history:
            standalone_question = contextualize_chain.invoke({
                "input": user_input,
                "chat_history": chat_history
            })
        else:
            standalone_question = user_input

        docs = retriever.invoke(standalone_question)
        context_text = "\n\n".join(doc.page_content for doc in docs)

        print(f"\n[DEBUG] Question: {user_input}")
        print(f"[DEBUG] Context preview:\n{context_text[:600]}\n")

        answer = answer_chain.invoke({
            "input": user_input,
            "chat_history": chat_history,
            "context": context_text
        })

        return {"answer": answer, "context": docs}

    conversational_chain = RunnableWithMessageHistory(
        RunnableLambda(rag_response),
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    return conversational_chain