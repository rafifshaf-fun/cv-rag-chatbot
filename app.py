import streamlit as st
from src.chain import get_chain

# ── Page config ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Rafif's AI CV Assistant",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Hide Streamlit chrome ──────────────────────────────────────────────
st.markdown(
    """
    <style>
        #MainMenu {visibility: hidden;}
        .stDeployButton {display: none;}
        header {visibility: visible !important;}
        .block-container {padding-top: 2rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h2 style='text-align: center;'>Rafif Shafwan</h2>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("assets/profpic.jpeg", width=150)

    st.markdown(
        "<p style='text-align: center;'>"
        "<a href='https://rafifshaf-fun.github.io'>Portfolio</a> · "
        "<a href='https://linkedin.com/in/rafif-shafwan'>LinkedIn</a> · "
        "<a href='https://github.com/rafifshaf-fun'>GitHub</a>"
        "</p>",
        unsafe_allow_html=True,
    )

    st.divider()

    try:
        with open("data/cv-rafif-shafwan-general-en.pdf", "rb") as pdf:
            st.download_button(
                label="📄 Download CV",
                data=pdf.read(),
                file_name="Rafif_Shafwan_CV.pdf",
                mime="application/octet-stream",
                use_container_width=True,
            )
    except FileNotFoundError:
        st.info("📄 CV PDF not found — place it in `data/` to enable downloads.")

# ── Chat header ────────────────────────────────────────────────────────
st.title("💬 Ask me anything")
st.caption("About Rafif's experience, skills, and projects")

# ── Chat history ───────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm Rafif's AI assistant. What would you like to know?"}
    ]

for msg in st.session_state.messages:
    avatar = "👤" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# ── Handle input ───────────────────────────────────────────────────────
if prompt := st.chat_input("Ask about Rafif's background…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        chain = get_chain()
        config = {"configurable": {"session_id": "cv_chat_session"}}
        chain_input = {"input": prompt}

        try:
            def stream():
                for chunk in chain.stream(chain_input, config=config):
                    if hasattr(chunk, "content"):
                        yield chunk.content
                    elif isinstance(chunk, dict):
                        yield chunk.get("answer") or chunk.get("output") or str(chunk)
                    else:
                        yield str(chunk)

            response = st.write_stream(stream())

        except Exception:
            with st.spinner("Thinking…"):
                raw = chain.invoke(chain_input, config=config)
                if isinstance(raw, dict):
                    response = raw.get("answer") or raw.get("output") or str(raw)
                elif hasattr(raw, "content"):
                    response = raw.content
                else:
                    response = str(raw)
                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})