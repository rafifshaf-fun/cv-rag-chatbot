import streamlit as st

# Import your existing RAG pipeline
from rag_pipeline import get_chain

# 1. Page Configuration (Must be the first Streamlit command)
st.set_page_config(
    page_title="Rafif's AI CV Assistant",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)

# 2. Custom CSS to fix the sidebar toggle but hide Streamlit's default menus
st.markdown("""
    <style>
        /* Hide the default Streamlit main menu and deploy button */
        #MainMenu {visibility: hidden;}
        .stDeployButton {display: none;}
        
        /* Ensure the header (which contains the sidebar toggle) remains visible */
        header {visibility: visible !important;}
        
        /* Optional: Adjust top padding to make it look cleaner */
        .block-container {padding-top: 2rem;}
    </style>
""", unsafe_allow_html=True)

# 3. Enhanced Sidebar
with st.sidebar:
    st.markdown("<h1 style='text-align: center;'>👨‍💻 About Me</h1>", unsafe_allow_html=True)
    
    # Centered avatar using st.columns
    col1, col2, col3 = st.columns([1,2,1])  # middle column wider
    with col2:
        st.image("assets/profpic.jpeg", width=150)

    st.markdown("<p style='text-align: center;'>Hi! I'm Rafif. Ask this AI anything about my experience, skills, or projects.</p>", unsafe_allow_html=True)
    st.divider()

    # Core competencies highlight
    st.markdown("<h4 style='text-align: left;'>🎯 Core Competencies</h4>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: left;'>- Machine Learning<br>- Computer Vision<br>- Data Analytics</p>", unsafe_allow_html=True)

    # Download CV Button
    try:
        with open("data/Rafif-Shafwan-CV.pdf", "rb") as pdf_file:
            pdf_byte = pdf_file.read()
            st.download_button(
                label="📄 Download My Resume",
                data=pdf_byte,
                file_name="Rafif_CV.pdf",
                mime="application/octet-stream"
            )
    except FileNotFoundError:
        st.warning("⚠️ Please place 'Rafif-Shafwan-CV.pdf' in the 'data' folder to enable downloads.")

    st.markdown("<p style='text-align: center;'><a href='https://linkedin.com/rafif-shafwan'>LinkedIn</a> | <a href='https://github.com/rafifshaf-fun'>GitHub</a></p>", unsafe_allow_html=True)

    st.divider()

    with st.expander("🛠️ How this app is built"):
        st.markdown("""
        - **UI:** Streamlit  
        - **LLM:** Google Gemini (`langchain_google_genai`)  
        - **Architecture:** Retrieval-Augmented Generation (RAG)  
        - **Memory:** LangChain Message History
        """)

# 4. Chat Interface Setup
st.title("💬 Chat with my CV")
st.caption("Powered by LangChain & Google Gemini")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm Rafif's AI assistant. How can I help you learn more about his background?"}
    ]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    avatar = "👤" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# 5. Handle User Input
if prompt := st.chat_input("Ask about my skills, experience, or projects..."):
    
    # Display user message in chat message container
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # Generate and display assistant response
    with st.chat_message("assistant", avatar="🤖"):
        # Load the RAG chain
        chain = get_chain()
        
        # Define the config with a session_id required by LangChain memory
        config = {"configurable": {"session_id": "cv_chat_session"}}
        
        # Format the input as a dictionary, which is standard for memory chains
        chain_input = {"input": prompt}
        
        try:
            # STREAMING RESPONSE
            def generate_response():
                # Apply chain_input and config to stream()
                for chunk in chain.stream(chain_input, config=config): 
                    # Safely handle different chunk formats from LangChain memory
                    if hasattr(chunk, "content"):
                        yield chunk.content
                    elif isinstance(chunk, dict) and "answer" in chunk:
                        yield chunk["answer"]
                    elif isinstance(chunk, dict) and "output" in chunk:
                        yield chunk["output"]
                    else:
                        yield str(chunk)

            response = st.write_stream(generate_response())
            
        except Exception as e:
            # FALLBACK: Standard invoke is used if streaming fails
            with st.spinner("Thinking..."):
                # Apply chain_input and config to invoke()
                raw_response = chain.invoke(chain_input, config=config) 
                
                # Extract text based on common LangChain memory output formats
                if isinstance(raw_response, dict) and "answer" in raw_response:
                    response = raw_response["answer"]
                elif isinstance(raw_response, dict) and "output" in raw_response:
                    response = raw_response["output"]
                elif hasattr(raw_response, "content"):
                    response = raw_response.content
                else:
                    response = str(raw_response)
                    
                st.markdown(response)
                
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})