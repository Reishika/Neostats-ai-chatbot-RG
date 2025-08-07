import streamlit as st
from utils.rag_tool import answer_with_knowledge_base
from utils.web_search_tool import answer_with_web_search
import markdown
from rapidfuzz import fuzz
from models.llm import classify_response_and_relevance
from utils.azure_speech_to_text import transcribe_speech_from_mic
import os
from dotenv import load_dotenv

# ---- Load env ----
load_dotenv()
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_REGION = os.getenv("AZURE_REGION")

from models.embeddings import create_index_if_not_exists, upload_chunks_to_search

# --- Initialize embedding index only once per session ---
if "embedding_index_created" not in st.session_state:
    print("Creating embedding index...")
    create_index_if_not_exists()
    upload_chunks_to_search()
    st.session_state.embedding_index_created = True

# ---- Styling ----
st.markdown(f"""
    <style>
    html, body, .stApp {{ background-color: #03202b !important; color: #d2d2d6 !important; font-family: 'Segoe UI', 'Inter', 'Arial', sans-serif; overflow: hidden; }}
    .fixed-header {{ position: fixed; top: 50px; left: 0; right: 0; background-color: #03202b; z-index: 1000; padding: 0em; }}
    .scrollable-chat {{ position: fixed; top: 240px; bottom: 80px; left: 0; right: 0; overflow-y: auto; padding: 1.5em; }}
    .fixed-input {{ position: fixed; bottom: 0; left: 0; right: 0; background-color: #03202b; z-index: 1000; padding: 1em 1.5em; border-top: 1px solid #1cb3e050; }}
    .chat-bubble {{ box-shadow: 0 1px 18px #00000020, 0 2px 4px #d4d0db; border-radius: 1.05em; margin-bottom: 0.9em; max-width: 70%; min-width: 110px; padding: 0.93rem 1.15rem 0.7rem 1.2rem; position: relative; }}
    .user {{ background: linear-gradient(90deg,#281d3b 80%,#534173 103%); color: #d2d2d6; margin-left: 28%; text-align: right; border-right: 4px solid #1cb3e0; border-top-right-radius: 0.1em; }}
    .bot {{ background: linear-gradient(90deg,#344758 97%,#354153 100%); color: #d2d2d6; margin-right: 28%; text-align: left; border-left: 4px solid #1cb3e0; border-top-left-radius: 0.1em; }}
    .chat-list-container {{ background: #23263a11; border-radius: 1.6em; padding: 1.7em; min-height: 100px; box-shadow: 0 2px 14px #182c3e55; }}
    .neon-label {{ color: #1cb3e0 !important; font-weight: 700; letter-spacing: .5px; text-shadow: 0 0 10px #1cb3e033, 0 0 14px #1cb3e022; }}
    ::placeholder {{ color: #B0BBC7 !important; opacity: 1; }}
    </style>
""", unsafe_allow_html=True)

# ---- Fixed Header ----
st.markdown("""
    <div class="fixed-header" style="text-align: center;">
        <h3 class="neon-label">	📋 PolicyPath Assistant</h3> 
        <p class="neon-label" style="font-size: 0.9em; margin-top: 0.1em;">
            Easily find your perfect MediShield policy – now with smart web search support
        </p>
    </div>
""", unsafe_allow_html=True)


# --- Session State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chat_mode" not in st.session_state:
    st.session_state.chat_mode = "insurance"

# --- Sidebar: Chat Mode Selection ---
with st.sidebar:
    st.markdown("### 🤖 Chat Mode Settings")
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background-color: #1a2b3b; border-radius: 0.5rem;">
        <p style="color: #d2d2d6; font-size: 1rem; font-weight: 500; margin: 0;">
            Select your preferred chat mode below: <br>
            💼 <b>Insurance Mode</b> helps you find the best policy for your needs <br>🌐 <b>General Web</b> is ideal for everyday questions and information.
        </p>
    </div>
""", unsafe_allow_html=True)    
    mode = st.radio(
        "Select Chat Mode",
        ["💼 Insurance Assistant", "🌐 General Web Search"],
        index=0,
        horizontal=False,
        label_visibility="collapsed",
        key="mode_selector"
    )
    st.session_state.chat_mode = "insurance" if mode == "💼 Insurance" else "general"

# --- Preprocess ---
def preprocess(text):
    return " ".join(text.strip().lower().split())

def is_acknowledgment_message(user_input):
    user_input_clean = preprocess(user_input)
    acknowledgments = [
        "thank you", "thanks", "got it", "ok", "okay", "sure", "noted", "understood",
        "cool", "great", "awesome", "hi", "hello", "hey", "hi there", "hello there",
        "hi!", "hello!", "hey!", "hi there!", "hello there!"
    ]
    for phrase in acknowledgments:
        if fuzz.ratio(user_input_clean, phrase) > 85:
            return True
    return False

# --- Chat Output ---
st.markdown('<div class="scrollable-chat"><div class="chat-list-container">', unsafe_allow_html=True)
for msg in st.session_state.chat_history:
    role_class = 'user' if msg["role"] == "user" else 'bot'
    icon = "🧑‍💻" if msg["role"] == "user" else "🤖"
    rendered_text = markdown.markdown(msg["text"]) if msg["role"] == "bot" else msg["text"]
    st.markdown(f"""
        <div class="chat-bubble {role_class}">
            <b>{icon} {msg["role"].capitalize()} <span style="color:#1cb3e0;font-weight:500;">({msg['source']})</span></b><br>
            <div>{rendered_text}</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown('</div></div>', unsafe_allow_html=True)


st.markdown(
    """
    <style>

    .stRadio > div {
        flex-direction: row;
        justify-content: left;
        gap: 1rem;
    }

    .stRadio label {
        background: #1a2b3b;
        color: #d2d2d6;
        padding: 0.5rem 0.5rem;
        border-radius: 1.2rem;
        border: 2px solid #1cb3e0;
        cursor: pointer;
        transition: background 0.3s ease;
        font-weight: 600;
        
    }

    .stRadio input:checked + div > label {
        background: linear-gradient(90deg,#1cb3e0,#179dc9);
        color: #fff;
        border: none;
    }

    .main > div {
        padding-top: 10rem !important; /* Push the rest of the content down */
    }
    </style>

    <div class="fixed-tabs">
    """,
    unsafe_allow_html=True
)

# --- Response Mode (Fixed position with input) ---
response_mode = st.radio("Response Mode", ["Concise", "Detailed"], horizontal=True, label_visibility="collapsed",)

# --- Chat Input + Mic Button ---
st.markdown('<div class="chat-input-container">', unsafe_allow_html=True)
user_input = st.chat_input("Type your message here...")


st.markdown('</div>', unsafe_allow_html=True)

# --- Chat Logic ---
if user_input:
    st.session_state.chat_history.append({
        "role": "user",
        "text": user_input,
        "mode": response_mode,
        "source": "User Input"
    })

    with st.spinner("🤖 Generating response..."):
        if is_acknowledgment_message(user_input):
            bot_reply = "Welcome! 😊 I'm your assistant. Feel free to ask your question."
            response_source = "System"
        else:
            if st.session_state.chat_mode == "general":
                bot_reply = answer_with_web_search.invoke({
                    "query": user_input,
                    "mode": response_mode.lower()
                }) or "Sorry, couldn't find information on the web."
                response_source = "Web Search"
            else:
                kb_result = answer_with_knowledge_base(user_input, mode=response_mode.lower())
                kb_response = kb_result if isinstance(kb_result, str) else str(kb_result)
                classification = classify_response_and_relevance(kb_response, user_input)

                if classification["response_class"] == "negative" and classification["is_relevant"] == "yes":
                    bot_reply = answer_with_web_search.invoke({
                        "query": user_input,
                        "mode": response_mode.lower()
                    }) or "Sorry, couldn't find anything in policy documents or web."
                    response_source = "Web Search"
                else:
                    bot_reply = kb_response
                    response_source = "Knowledge Base"

    st.session_state.chat_history.append({
        "role": "bot",
        "text": bot_reply,
        "mode": response_mode,
        "source": response_source
    })
    st.rerun()
