
   import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    return tokenizer, model

tokenizer, model = load_model()

# Sidebar info
st.sidebar.title("👤 About")
st.sidebar.markdown("""
**Name:** Your Name  
**Team:** Your Team Name  
**Department:** Your Department Name
""")

st.title("🎈 MY CHATBOX")
st.write("This is My Chat.")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat input
user_input = st.text_input("You:", key="input")

if user_input:
    st.session_state.chat_history.append(("You", user_input))

    # Encode input and generate response
    input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    output_ids = model.generate(
        input_ids,
        max_length=100,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95
    )
    response = tokenizer.decode(output_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    st.session_state.chat_history.append(("Bot", response))

# Display chat history
for sender, message in st.session_state.chat_history:
    st.markdown(f"**{sender}:** {message}")
