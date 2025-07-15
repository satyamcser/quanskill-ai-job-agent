import sys
import os
import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# ✅ Add project root for consistency
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# ✅ Load a lightweight LLM (cached so it doesn’t reload on every click)
@st.cache_resource
def load_model():
    model_name = "google/flan-t5-base"  # lightweight model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return pipeline("text2text-generation", model=model, tokenizer=tokenizer)

llm = load_model()

# ✅ Streamlit page config
st.set_page_config(page_title="Simple Chatbot", layout="centered")
st.title("💬 Simple Chatbot")
st.write("This is a plain LLM chatbot without job matching or resume parsing.")

# ✅ Maintain chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Chat Input ---
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Ask me anything:", "")
    send_button = st.form_submit_button("Send")

# --- On user input ---
if send_button and user_input.strip():
    # Save user message
    st.session_state.chat_history.append(("You", user_input))

    # Generate response
    with st.spinner("Thinking..."):
        response = llm(user_input, max_length=256, do_sample=False)
        bot_reply = response[0]["generated_text"]

    # Save bot reply
    st.session_state.chat_history.append(("Bot", bot_reply))

# --- Display chat history ---
st.markdown("### 🗨️ Conversation")
for role, message in st.session_state.chat_history:
    if role == "You":
        st.markdown(f"**🧑 You:** {message}")
    else:
        st.markdown(f"**🤖 Bot:** {message}")

# --- Clear chat button ---
if st.button("🧹 Clear Chat"):
    st.session_state.chat_history = []
    st.experimental_rerun()
