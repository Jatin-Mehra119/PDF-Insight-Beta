import os
import tempfile
import streamlit as st
from streamlit_chat import message
from preprocessing import Model

# Home Page Setup 
st.set_page_config(
    page_title="PDF Insight Pro", 
    page_icon="📄", 
    layout="centered",
)

# Custom CSS for a more polished look
st.markdown("""
    <style>
        .main { 
            background-color: #f5f5f5;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
        }
        .stTextInput input {
            border-radius: 8px;
            padding: 10px;
        }
        .stFileUploader input {
            border-radius: 8px;
        }
        .stMarkdown h1 {
            color: #4CAF50;
        }
    </style>
""", unsafe_allow_html=True)

# Custom title and header
st.title("📄 PDF Insight Pro")
st.subheader("Empower Your Documents with AI-Driven Insights")

def display_messages():
    """
    Displays the chat messages in the Streamlit UI.
    """
    st.subheader("🗨️ Conversation")
    st.markdown("---")
    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        message(msg, is_user=is_user, key=str(i))
    st.session_state["process_input_spinner"] = st.empty()

def process_user_input():
    """
    Processes the user input by generating a response from the assistant.
    """
    if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:
        user_input = st.session_state["user_input"].strip()
        with st.session_state["process_input_spinner"], st.spinner("Analyzing..."):
            agent_response = st.session_state["assistant"].get_response(
                user_input,
                st.session_state["temperature"],
                st.session_state["max_tokens"],
                st.session_state["model"]
            )

        st.session_state["messages"].append((user_input, True))
        st.session_state["messages"].append((agent_response, False))
        st.session_state["user_input"] = ""

def process_file():
    """
    Processes the uploaded PDF file and appends its content to the context.
    """
    for file in st.session_state["file_uploader"]:
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(file.getbuffer())
            file_path = tf.name

        with st.session_state["process_file_spinner"], st.spinner(f"Processing {file.name}..."):
            try:
                st.session_state["assistant"].add_to_context(file_path)
            except Exception as e:
                st.error(f"Failed to process file {file.name}: {str(e)}")
        os.remove(file_path)

def main_page():
    """
    Main function to set up the Streamlit UI and handle user interactions.
    """
    # Initialize session state variables
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    if "assistant" not in st.session_state:
        st.session_state["assistant"] = Model()

    if "user_input" not in st.session_state:
        st.session_state["user_input"] = ""

    if "temperature" not in st.session_state:
        st.session_state["temperature"] = 0.5

    if "max_tokens" not in st.session_state:
        st.session_state["max_tokens"] = 550

    if "model" not in st.session_state:
        st.session_state["model"] = "llama-3.1-8b-instant"

    # File uploader
    st.subheader("📤 Upload Your PDF Documents")
    st.file_uploader(
        "Choose PDF files to analyze",
        type=["pdf"],
        key="file_uploader",
        on_change=process_file,
        accept_multiple_files=True,
    )

    st.session_state["process_file_spinner"] = st.empty()

    # Document management section
    if st.session_state["assistant"].contexts:
        st.subheader("🗂️ Manage Uploaded Documents")
        for i, context in enumerate(st.session_state["assistant"].contexts):
            st.text_area(f"Document {i+1} Context", context[:500] + "..." if len(context) > 500 else context, height=100)
            if st.button(f"Remove Document {i+1}"):
                st.session_state["assistant"].remove_from_context(i)

    # Model settings
    with st.expander("⚙️ Customize AI Settings", expanded=True):
        st.slider("Sampling Temperature", min_value=0.0, max_value=1.0, step=0.1, key="temperature", help="Higher values make output more random.")
        st.slider("Max Tokens", min_value=50, max_value=1000, step=50, key="max_tokens", help="Limits the length of the response.")
        st.selectbox("Choose AI Model", ["llama-3.1-8b-instant", "llama3-70b-8192", "gemma-7b-it"], key="model")

    # Display messages and input box
    display_messages()
    st.text_input("Type your query and hit Enter", key="user_input", on_change=process_user_input, placeholder="Ask something about your documents...")
    # Developer info and bug report
    st.subheader("🐞 Bug Report")
    st.markdown("""
        If you encounter any bugs or issues while using the app, please send a bug report to the developer. You can include a screenshot (optional) to help identify the problem.\n
    """)
    st.subheader("💡 Suggestions")
    st.markdown("""
        Suggestions to improve the app's UI and user interface are also welcome. Feel free to reach out to the developer with your suggestions.\n
    """)
    st.subheader("👨‍💻 Developer Info")
    st.markdown("""
        **Developer**: Jatin Mehra\n
        **Email**: jatinmehra119@gmail.com\n
        **Mobile**: 9910364780\n
    """)
if __name__ == "__main__":
    main_page()