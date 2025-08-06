import streamlit as st
from utils.rag_tool import answer_with_knowledge_base
from utils.web_search_tool import answer_with_web_search
import markdown
from rapidfuzz import fuzz
from models.llm import classify_response_and_relevance  # Updated import


# ---- Styling ----
st.markdown(f"""
    <style>
    html, body, .stApp {{
        background-color: #03202b !important;
        color: #d2d2d6 !important;
        font-family: 'Segoe UI', 'Inter', 'Arial', sans-serif;
        overflow: hidden;
    }}
    .fixed-header {{
        position: fixed;
        top: 50px;
        left: 0;
        right: 0;
        background-color: #03202b;
        z-index: 1000;
        padding: 0em;
    }}
    .scrollable-chat {{
        position: fixed;
        top: 200px;
        bottom: 80px;
        left: 0;
        right: 0;
        overflow-y: auto;
        padding: 1.5em;
    }}
    .fixed-input {{
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: #03202b;
        z-index: 1000;
        padding: 1em 1.5em;
        border-top: 1px solid #1cb3e050;
    }}
    .chat-bubble {{
        box-shadow: 0 1px 18px #00000020, 0 2px 4px #d4d0db;
        border-radius: 1.05em;
        margin-bottom: 0.9em;
        max-width: 70%;
        min-width: 110px;
        padding: 0.93rem 1.15rem 0.7rem 1.2rem;
        position: relative;
    }}
    .user {{
        background: linear-gradient(90deg,#281d3b 80%,#534173 103%);
        color: #d2d2d6;
        margin-left: 28%;
        text-align: right;
        border-right: 4px solid #1cb3e0;
        border-top-right-radius: 0.1em;
    }}
    .bot {{
        background: linear-gradient(90deg,#344758 97%,#354153 100%);
        color: #d2d2d6;
        margin-right: 28%;
        text-align: left;
        border-left: 4px solid #1cb3e0;
        border-top-left-radius: 0.1em;
    }}
    .chat-list-container {{
        background: #23263a11;
        border-radius: 1.6em;
        padding: 1.7em;
        min-height: 100px;
        box-shadow: 0 2px 14px #182c3e55;
    }}
    .neon-label {{
        color: #1cb3e0 !important;
        font-weight: 700;
        letter-spacing: .5px;
        text-shadow: 0 0 10px #1cb3e033, 0 0 14px #1cb3e022;
    }}
    ::placeholder {{
        color: #B0BBC7 !important;
        opacity: 1;
    }}
    </style>
""", unsafe_allow_html=True)

# ---- Fixed Header ----
st.markdown("""
    <div class="fixed-header">
        <h3 class="neon-label">📋 Insurance Policy Navigator</h3> 
        <p class="neon-label" style="font-size: 0.9em; margin-top: 0.1em; margin-left: 0.2em;">
            Choose the best insurance policy for your needs with our AI assistant.
        </p>
    </div>
""", unsafe_allow_html=True)

# --- Initialize session state ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

from utils.azure_speech_to_text import transcribe_speech_from_mic


# Optional: Get keys from environment or secrets
AZURE_SPEECH_KEY = "B0gnCpS51yged1LJwzsgnBrddWukdOZeGuAE9ZklJiC01OOiv3kxJQQJ99BHAC5RqLJXJ3w3AAAEACOGJLSN"
AZURE_REGION = "westeurope"

# Call the function when needed

def preprocess(text):
    return " ".join(text.strip().lower().split())

def is_acknowledgment_message(user_input):
    user_input_clean = preprocess(user_input)
    acknowledgments = [
        "thank you", "thanks", "got it", "ok", "okay", "sure",
        "noted", "understood", "cool", "great", "awesome", "hi", "hello",
        "hey", "hi there", "hello there", "hi!", "hello!", "hey!", "hi there!", "hello there!"
    ]
    for phrase in acknowledgments:
        if fuzz.ratio(user_input_clean, phrase) > 85:
            return True
    return False

# --- Scrollable chat area ---
st.markdown('<div class="scrollable-chat"><div class="chat-list-container">', unsafe_allow_html=True)

for msg in st.session_state.chat_history:
    role_class = 'user' if msg["role"] == "user" else 'bot'
    icon = "🧑‍💻" if msg["role"] == "user" else "🤖"
    rendered_text = markdown.markdown(msg["text"]) if msg["role"] == "bot" else msg["text"]

    st.markdown(
        f"""
        <div class="chat-bubble {role_class}">
            <b>{icon} {msg["role"].capitalize()}
            <span style="color:#1cb3e0;font-weight:500;">({msg["source"]})</span>
            </b><br>
            <div>{rendered_text}</div>
        </div>
        """, unsafe_allow_html=True
    )


response_mode = st.radio("Response Mode", ["Concise", "Detailed"], horizontal=True)


# --- Inject custom CSS to place mic beside input ---
st.markdown("""
    <style>
        
        .mic-button {
            background-color: #f0f0f0;
            padding: 0rem 1rem;
            border-radius: 2px;
            border: none;
            cursor: pointer;

        }
    </style>
""", unsafe_allow_html=True)

# --- Place custom HTML wrapper ---
st.markdown('<div class="chat-input-container">', unsafe_allow_html=True)

# --- Chat Input ---
user_input = st.chat_input("Type your message here...")

# --- Mic Button using JavaScript trigger ---
col1, col2 = st.columns([10, 1])
with col2:
    if st.button("🎤", key="mic_button"):
        st.session_state.is_recording = True
        with st.spinner("🎙️ Recording... Speak now"):
            query = transcribe_speech_from_mic(AZURE_SPEECH_KEY, AZURE_REGION)
            st.session_state.user_query = query
        st.session_state.is_recording = False

st.markdown('</div>', unsafe_allow_html=True)

user_input = query if 'query' in locals() else user_input
st.markdown('</div>', unsafe_allow_html=True)

# --- Handle user input ---
if user_input:
    st.session_state.chat_history.append({
        "role": "user",
        "text": user_input,
        "mode": response_mode,
        "source": "User Input"
    })

    with st.spinner("🤖 Generating response..."):
        if is_acknowledgment_message(user_input):
            bot_reply = "Welcome! 😊\n\nI am an Insurance Assistant Bot, feel free to ask your questions."
            response_source = "System"
        else:
            # Step 1: Get KB response
            kb_result = answer_with_knowledge_base(user_input, mode=response_mode.lower())
            kb_response = kb_result if isinstance(kb_result, str) else str(kb_result)

            # Step 2: Classify the bot response and user query
            classification = classify_response_and_relevance(kb_response, user_input)

            if (
                classification["response_class"] == "negative"
                and classification["is_relevant"] == "yes"
            ):
                bot_reply = answer_with_web_search.invoke({
                    "query": user_input,
                    "mode": response_mode.lower()
                })
                response_source = "Web Search"

                if not bot_reply:
                    bot_reply = "Sorry, I couldn't find relevant information in our policy documents or the web. Please contact support."
                    response_source = "System"
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