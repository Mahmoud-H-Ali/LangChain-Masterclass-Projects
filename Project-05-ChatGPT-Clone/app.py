import streamlit as st
from streamlit_chat import message
from langchain_openai import ChatOpenAI
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import os

# App initialization
st.set_page_config(page_title="Chat GPT Clone", page_icon=":robot_face:", layout="wide")
st.title("🤖 ChatGPT Clone")

# Sidebar for configurations
def sidebar_config():
    st.sidebar.title("🔑 API Keys")
    keys = {
        'openai': st.sidebar.text_input("OpenAI API key", type="password"),
        'hf': st.sidebar.text_input("Hugging Face API key", type="password")
    }
    temperature = st.sidebar.slider("Temperature", 0.0, 2.0, 0.0, 0.1)
    return keys, temperature

# Manage chat sessions
def manage_chat_sessions():
    if "chat_sessions" not in st.session_state:
        st.session_state.chat_sessions = {"Chat 1": {"memory": ConversationBufferMemory(), "messages": [], "provider": "OpenAI", "model": "gpt-3.5-turbo"}}
    
    chat_names = list(st.session_state.chat_sessions.keys())
    current_chat = st.sidebar.radio("Chat Sessions", chat_names + ["➕ New Chat"], index=chat_names.index(st.session_state.get("current_chat", "Chat 1")))

    if current_chat == "➕ New Chat":
        new_chat_name = st.sidebar.text_input("Chat Name")
        if st.sidebar.button("Create") and new_chat_name:
            st.session_state.chat_sessions[new_chat_name] = {"memory": ConversationBufferMemory(), "messages": [], "provider": "OpenAI", "model": "gpt-3.5-turbo"}
            st.session_state.current_chat = new_chat_name
            st.rerun()
    else:
        st.session_state.current_chat = current_chat

    return st.session_state.chat_sessions[st.session_state.current_chat]

# LLM initialization
def get_llm(provider, model, keys, temperature):
    if provider == "OpenAI" and keys['openai']:
        return ChatOpenAI(model_name=model, temperature=temperature, openai_api_key=keys['openai'])
    elif provider == "Hugging Face" and keys['hf']:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = keys['hf']
        return HuggingFaceEndpoint(repo_id=model, temperature=temperature, task="text-generation", stop=["Human:", "\n\n"])
    return None

# UI Elements for model selection
def model_selector(chat_session):
    with st.expander("Chat Options"):
        providers = ["OpenAI", "Hugging Face"]
        models = {
            "OpenAI": ["gpt-3.5-turbo", "gpt-4"],
            "Hugging Face": [
                "mistralai/Mistral-7B-Instruct-v0.2",
                "tiiuae/falcon-7b-instruct",
                "meta-llama/Llama-2-7b-chat-hf",
                "google/flan-t5-xl",
                "databricks/dolly-v2-7b"
            ]
        }

        provider = st.selectbox("Provider", providers, index=providers.index(chat_session["provider"]))

        if chat_session["provider"] != provider:
            chat_session["model"] = models[provider][0]

        model = st.selectbox("Model", models[provider], index=models[provider].index(chat_session["model"]))

        if chat_session["provider"] != provider or chat_session["model"] != model:
            chat_session.update({"provider": provider, "model": model})
            chat_session.pop("conversation", None)

# Chat message handling
def handle_chat(llm, chat_session):
    chat_window = st.container()
    with chat_window:
        for i, (msg, is_user, _) in enumerate(chat_session["messages"]):
            message(msg, is_user=is_user, key=f"msg_{i}")

    user_input = st.text_input("Your message:")
    if st.button("Send") and user_input and llm:
        chat_session["messages"].append((user_input, True, chat_session["model"]))
        response = llm.invoke({"input": user_input})["response"].strip()
        chat_session["messages"].append((response, False, chat_session["model"]))
        st.rerun()

# Main function
def main():
    keys, temperature = sidebar_config()
    chat_session = manage_chat_sessions()
    model_selector(chat_session)

    llm = get_llm(chat_session["provider"], chat_session["model"], keys, temperature)
    if not llm:
        st.warning("Please provide the required API keys.")
        return

    if "conversation" not in chat_session:
        conversation = ConversationChain(llm=llm, memory=chat_session["memory"], prompt=PromptTemplate(input_variables=["input", "history"], template="{history}\nHuman: {input}\nAI:"))
        chat_session["conversation"] = conversation

    handle_chat(chat_session["conversation"], chat_session)

if __name__ == "__main__":
    main()
