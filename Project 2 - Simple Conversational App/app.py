import streamlit as st
import os
from langchain_openai import ChatOpenAI
from langchain_community.llms import HuggingFaceHub
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

# From here down is all the StreamLit UI
st.set_page_config(page_title="LangChain Demo", page_icon=":robot:", layout="wide")

st.header("Hey, I'm your Chat GPT")

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
    "OpenAI": ["gpt-3.5-turbo"],
    "Hugging Face": [
        "tiiuae/falcon-7b-instruct",
        "databricks/dolly-v2-3b",
        "bigscience/bloomz-560m",
        "google/flan-t5-base",
        "TheBloke/Mistral-7B-Instruct-v0.1-GGUF"
    ]
}

# Select provider (OpenAI or Hugging Face)
provider = st.sidebar.radio("🌍 Select Model Provider:", ["OpenAI", "Hugging Face"])

# Select Model
selected_model = st.sidebar.selectbox("🤖 Select Model:", model_options[provider])

# Save selection in session state
st.session_state["selected_model"] = selected_model
st.session_state["provider"] = provider


if "sessionMessages" not in st.session_state:
    st.session_state.sessionMessages = [
        SystemMessage(content="You are a helpful assistant.")
    ]

# Optionally track cached provider/model.
if "cached_provider" not in st.session_state:
    st.session_state["cached_provider"] = None
if "cached_model" not in st.session_state:
    st.session_state["cached_model"] = None
if "model_obj" not in st.session_state:
    st.session_state["model_obj"] = None

def initialize_chat_model():
    try:
        # If we already have a model cached and it matches the current provider/model, reuse it.
        if (st.session_state["model_obj"] is not None
            and st.session_state["cached_provider"] == provider
            and st.session_state["cached_model"] == selected_model):
            return st.session_state["model_obj"]

        # Otherwise, create and cache.
        if provider == "OpenAI":
            if not st.session_state["openai_api_key"]:
                st.error("Please enter an OpenAI API key.")
                return None
            new_model = ChatOpenAI(
                temperature=0.7,
                api_key=st.session_state["openai_api_key"],
                model_name=selected_model
            )
        else:  # Hugging Face
            if not st.session_state["huggingface_api_key"]:
                st.error("Please enter a Hugging Face API key.")
                return None
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.session_state["huggingface_api_key"]

            # Configure task based on model
            task = "text-generation"
            # If T5-based model, use text2text-generation and limit max_new_tokens to 250 or below.
            if "t5" in selected_model.lower():
                task = "text2text-generation"
                max_tokens = 128  # T5-based models often have a lower limit, set to 128 to avoid errors.
            else:
                max_tokens = 256

            new_model = HuggingFaceHub(
                repo_id=selected_model,
                task=task,
                model_kwargs={
                    "max_new_tokens": max_tokens,
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "do_sample": True,
                    "pad_token_id": 50256  # Add padding token
                }
            )

        # Cache the new model and update tracking.
        st.session_state["model_obj"] = new_model
        st.session_state["cached_provider"] = provider
        st.session_state["cached_model"] = selected_model
        return st.session_state["model_obj"]

    except Exception as e:
        st.error(f"Error initializing model: {str(e)}")
        return None

def build_prompt(messages):
    """Builds a single string prompt from the list of messages for Hugging Face or other single-string LLMs."""
    prompt_parts = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            prompt_parts.append(f"System: {msg.content}")
        elif isinstance(msg, HumanMessage):
            prompt_parts.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            prompt_parts.append(f"Assistant: {msg.content}")
        else:
            prompt_parts.append(msg.content)
    return "\n".join(prompt_parts)

def load_answer(question):
    try:
        chat = initialize_chat_model()
        if not chat:
            return "Please configure API keys first."

        st.session_state.sessionMessages.append(HumanMessage(content=question))

        if st.session_state["provider"] == "OpenAI":
            assistant_answer = chat.invoke(st.session_state.sessionMessages)
            answer_content = assistant_answer.content
        else:  # Hugging Face
            hf_prompt = build_prompt(st.session_state.sessionMessages)
            answer_content = chat.invoke(hf_prompt)

        st.session_state.sessionMessages.append(AIMessage(content=answer_content))
        return answer_content
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Initialize model once outside
chat = initialize_chat_model()
if chat:
    st.write("Model initialized successfully.")
else:
    st.write("Error initializing model.")

def get_text():
    input_text = st.text_input("You: ")
    return input_text

user_input = get_text()
submit = st.button('Generate')

if submit:
    if user_input.strip():
        response = load_answer(user_input)
        st.subheader("Answer:")
        st.write(response)
    else:
        st.warning("Please enter a question before generating a response.")

# Display a conversation box showing the last few messages
st.markdown("## Conversation")

# We'll display the last 5 messages, excluding the initial system message.
messages_to_display = st.session_state.sessionMessages[1:][-5:] if len(st.session_state.sessionMessages) > 1 else []

# Build a small HTML container with scroll
conversation_html = ""
for msg in messages_to_display:
    if isinstance(msg, HumanMessage):
        conversation_html += f"<p><strong>User:</strong> {msg.content}</p>"
    elif isinstance(msg, AIMessage):
        conversation_html += f"<p><strong>Assistant:</strong> {msg.content}</p>"
    else:
        conversation_html += f"<p>{msg.content}</p>"

st.markdown(
    f"""<div style="height: 300px; overflow-y: auto; border: 1px solid #ccc; padding: 10px">
    {conversation_html}
    </div>""",
    unsafe_allow_html=True
)
