import streamlit as st
import os
from langchain_openai import OpenAI
from langchain_community.llms import HuggingFaceEndpoint
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set Streamlit page configuration
st.set_page_config(page_title="LangChain Q&A", page_icon="🤖", layout="wide")

# Sidebar: Control Panel
st.sidebar.header("⚙️ Control Panel")

# User-provided OpenAI API Key
if "openai_api_key" not in st.session_state:
    st.session_state["openai_api_key"] = os.getenv("OPENAI_API_KEY", "")

openai_key = st.sidebar.text_input("🔑 Enter OpenAI API Key:", 
                                   value=st.session_state["openai_api_key"], type="password")

if openai_key:
    st.session_state["openai_api_key"] = openai_key

# User-provided Hugging Face API Key
if "huggingface_api_key" not in st.session_state:
    st.session_state["huggingface_api_key"] = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")

huggingface_key = st.sidebar.text_input("🔑 Enter Hugging Face API Key:", 
                                        value=st.session_state["huggingface_api_key"], type="password")

if huggingface_key:
    st.session_state["huggingface_api_key"] = huggingface_key

# Model selection: OpenAI & Hugging Face models
model_options = {
    "OpenAI": ["gpt-3.5-turbo-instruct"],
    "Hugging Face": [
        "mistralai/Mistral-7B-Instruct-v0.2",
        "tiiuae/falcon-7b-instruct",
        "bigscience/bloomz-7b1"
    ]
}

# Select provider (OpenAI or Hugging Face)
provider = st.sidebar.radio("🌍 Select Model Provider:", ["OpenAI", "Hugging Face"])

# Select Model
selected_model = st.sidebar.selectbox("🤖 Select Model:", model_options[provider])

# Save selection in session state
st.session_state["selected_model"] = selected_model
st.session_state["provider"] = provider

# Function to generate response
def load_answer(question):
    try:
        if not question.strip():
            return "⚠️ Please enter a valid question."

        if st.session_state["provider"] == "OpenAI":
            if not st.session_state["openai_api_key"]:
                return "❌ Error: OpenAI API key is missing!"
            llm = OpenAI(
                model_name=st.session_state["selected_model"], 
                temperature=0, 
                openai_api_key=st.session_state["openai_api_key"]
            )
            return llm.invoke(question)

        elif st.session_state["provider"] == "Hugging Face":
            if not st.session_state["huggingface_api_key"]:
                return "❌ Error: Hugging Face API key is missing!"
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.session_state["huggingface_api_key"]
            llm = HuggingFaceEndpoint(repo_id=st.session_state["selected_model"])
            return llm.invoke(question)

    except Exception as e:
        return f"❌ Error: {str(e)}"

# Main UI
st.title("🧠 LangChain Q&A Chatbot")

# Input box
user_input = st.text_input("Ask a question:", key="input")

# Generate answer when button is clicked
if st.button("Get Answer"):
    if user_input:
        with st.spinner("Thinking... 💭"):
            response = load_answer(user_input)
        st.subheader("Answer:")
        st.write(response)
    else:
        st.warning("⚠️ Please enter a question first.")
