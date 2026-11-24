"""Streamlit User Interface for AI-Smart-Chatbot.

Provides multimodal conversational chat, interactive knowledge base management,
audio playback, image reasoning, and crisis resource access.
"""

import base64
import os
import uuid
import httpx
import streamlit as st

# Application configuration & backend target
BACKEND_HOST = os.getenv("BACKEND_HOST", "127.0.0.1")
BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8000"))
BACKEND_URL = f"http://{BACKEND_HOST}:{BACKEND_PORT}"

# Page configuration
st.set_page_config(
    page_title="Aura Wellness | Multimodal RAG Assistant",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown(
    """
    <style>
    .safety-banner {
        background-color: #FFF3CD;
        color: #856404;
        padding: 12px 18px;
        border-radius: 8px;
        border-left: 5px solid #FFEBAA;
        margin-bottom: 20px;
        font-size: 0.92rem;
    }
    .crisis-badge {
        background-color: #F8D7DA;
        color: #721C24;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .citation-card {
        background-color: #F8F9FA;
        border: 1px solid #DEE2E6;
        border-radius: 6px;
        padding: 10px;
        margin-top: 6px;
        font-size: 0.88rem;
    }
    .status-badge-ok {
        color: #155724;
        background-color: #D4EDDA;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .status-badge-down {
        color: #721C24;
        background-color: #F8D7DA;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def fetch_health():
    """Query backend health status."""
    try:
        resp = httpx.get(f"{BACKEND_URL}/health", timeout=2.5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def fetch_documents():
    """Query list of indexed documents."""
    try:
        resp = httpx.get(f"{BACKEND_URL}/documents", timeout=3.0)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return []


# Initialize Session State
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! I am **Aura**, your emotional resilience and wellness assistant. "
                "I can guide you through grounding techniques (like 5-4-3-2-1 and box breathing), "
                "listen with care, and answer questions about wellness guidelines and benefits. "
                "How are you feeling today?"
            ),
            "citations": [],
            "audio_base64": None,
        }
    ]

# -----------------------------------------------------------------------------
# Top Safety Disclaimer Banner
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="safety-banner">
        ⚠️ <strong>Prototype Limitations & Healthcare Notice:</strong> This is an AI wellness assistant demonstration. 
        It is <strong>not</strong> a licensed healthcare provider, medical clinic, or crisis intervention service. 
        If you or someone you know is in imminent danger or having thoughts of self-harm, please dial 
        <strong>988</strong> (US/Canada), <strong>111/999</strong> (UK), <strong>112</strong> (EU/India), or consult a medical professional.
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Sidebar: System Status, Settings, and Knowledge Base Management
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🌿 Aura Wellness")
    st.caption("Multimodal RAG Assistant")

    # System Status Indicator
    st.subheader("System Health")
    health_info = fetch_health()
    if health_info:
        db_type = health_info.get("db_type", "unknown").upper()
        llm = health_info.get("providers", {}).get("llm", "mock").upper()
        st.markdown(
            f'<span class="status-badge-ok">● Backend Online</span> &nbsp; '
            f'<code>DB: {db_type}</code> &nbsp; <code>LLM: {llm}</code>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="status-badge-down">● Backend Offline</span> (Check 127.0.0.1:8000)',
            unsafe_allow_html=True,
        )

    st.divider()

    # Modality & RAG Toggles
    st.subheader("Preferences")
    use_rag = st.checkbox("Enable RAG Grounding", value=True, help="Search indexed wellness handbook chunks.")
    enable_tts = st.checkbox("Synthesize Spoken Audio", value=False, help="Generate spoken audio playback for responses.")

    st.divider()

    # Knowledge Base Document Manager
    st.subheader("Knowledge Base")
    uploaded_doc = st.file_uploader(
        "Upload Reference Document",
        type=["pdf", "txt"],
        help="Upload .pdf or .txt wellness documents up to 10MB.",
    )

    if uploaded_doc is not None:
        if st.button("Ingest into Knowledge Base", use_container_width=True):
            with st.spinner("Extracting and indexing text chunks..."):
                try:
                    files = {"file": (uploaded_doc.name, uploaded_doc.getvalue(), uploaded_doc.type)}
                    resp = httpx.post(f"{BACKEND_URL}/upload/document", files=files, timeout=30.0)
                    if resp.status_code == 200:
                        data = resp.json()
                        st.success(f"Indexed '{data['filename']}' ({data['chunk_count']} chunks)!")
                        st.rerun()
                    else:
                        st.error(f"Upload failed ({resp.status_code}): {resp.text}")
                except Exception as exc:
                    st.error(f"Connection error: {exc}")

    # Document Library Table
    docs = fetch_documents()
    if docs:
        st.markdown(f"**Indexed Documents ({len(docs)}):**")
        for doc in docs:
            col1, col2 = st.columns([3, 1])
            col1.markdown(f"📄 **{doc['filename']}** ({doc['chunk_count']} chunks)")
            if col2.button("🗑️", key=f"del_{doc['id']}", help="Delete document"):
                try:
                    del_resp = httpx.delete(f"{BACKEND_URL}/documents/{doc['id']}", timeout=5.0)
                    if del_resp.status_code == 200:
                        st.toast(f"Deleted {doc['filename']}")
                        st.rerun()
                except Exception as exc:
                    st.error(f"Delete failed: {exc}")
    else:
        st.info("No documents indexed. Seed data or upload a file.")

    st.divider()

    # Emergency Support Directory
    with st.expander("🚨 Emergency Crisis Resources"):
        st.markdown(
            """
            * **United States & Canada:** Dial or text **988**
            * **United Kingdom:** Call **111** (Mental Health) or **999**
            * **European Union:** Dial **112**
            * **India:** Dial **14416** (Tele-MANAS) or **112**
            * **International Directory:** [findahelpline.com](https://findahelpline.com)
            """
        )

    # Clear Conversation State
    if st.button("🧹 Clear Chat History", use_container_width=True):
        try:
            httpx.post(f"{BACKEND_URL}/chat/clear", json={"conversation_id": st.session_state.conversation_id}, timeout=3.0)
        except Exception:
            pass
        st.session_state.conversation_id = str(uuid.uuid4())
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Conversation cleared. How can I help you reset or support you right now?",
                "citations": [],
                "audio_base64": None,
            }
        ]
        st.rerun()

# -----------------------------------------------------------------------------
# Main Chat Canvas
# -----------------------------------------------------------------------------
st.title("🌿 Aura Wellness Chat")

# Display Message History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Display Citations if available
        if msg.get("citations"):
            with st.expander(f"📚 Grounding Citations ({len(msg['citations'])} excerpts)"):
                for cite in msg["citations"]:
                    st.markdown(
                        f"""
                        <div class="citation-card">
                            <strong>{cite['filename']}</strong> (Chunk #{cite['chunk_index']} | Relevance Score: {cite['score']})<br/>
                            <em>"{cite['excerpt']}"</em>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        # Render Spoken Audio Player if audio synthesized
        if msg.get("audio_base64"):
            try:
                audio_bytes = base64.b64decode(msg["audio_base64"])
                st.audio(audio_bytes, format="audio/wav")
            except Exception:
                pass

# -----------------------------------------------------------------------------
# Inline Multimodal Attachments (Image and Audio)
# -----------------------------------------------------------------------------
with st.expander("📎 Multimodal Attachments (Image or Voice Query)"):
    tab1, tab2 = st.tabs(["🖼️ Image Query", "🎙️ Voice Audio Query"])

    with tab1:
        img_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg", "webp"], key="img_uploader")
        img_prompt = st.text_input("Question about Image", value="What does this image show from a wellness and mindfulness perspective?")
        if img_file is not None and st.button("Analyze Image"):
            with st.spinner("Analyzing image..."):
                try:
                    files = {"file": (img_file.name, img_file.getvalue(), img_file.type)}
                    data = {"prompt": img_prompt, "conversation_id": st.session_state.conversation_id}
                    resp = httpx.post(f"{BACKEND_URL}/upload/image", files=files, data=data, timeout=30.0)
                    if resp.status_code == 200:
                        res = resp.json()
                        st.session_state.messages.append({"role": "user", "content": f"[Uploaded Image: {img_file.name}] {img_prompt}"})
                        st.session_state.messages.append({"role": "assistant", "content": res["reply"], "citations": []})
                        st.rerun()
                    else:
                        st.error(f"Image analysis error: {resp.text}")
                except Exception as exc:
                    st.error(f"Request failed: {exc}")

    with tab2:
        audio_file = st.file_uploader("Upload Audio Clip", type=["wav", "mp3", "m4a"], key="audio_uploader")
        if audio_file is not None and st.button("Transcribe & Ask Aura"):
            with st.spinner("Transcribing speech and generating response..."):
                try:
                    files = {"file": (audio_file.name, audio_file.getvalue(), audio_file.type)}
                    data = {"conversation_id": st.session_state.conversation_id}
                    resp = httpx.post(f"{BACKEND_URL}/upload/audio", files=files, data=data, timeout=35.0)
                    if resp.status_code == 200:
                        res = resp.json()
                        st.session_state.messages.append({"role": "user", "content": f"🎙️ [Voice]: \"{res['transcribed_text']}\""})
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": res["reply"],
                            "citations": [],
                            "audio_base64": res.get("audio_base64"),
                        })
                        st.rerun()
                    else:
                        st.error(f"Audio processing error: {resp.text}")
                except Exception as exc:
                    st.error(f"Request failed: {exc}")

# -----------------------------------------------------------------------------
# Chat Input Prompt
# -----------------------------------------------------------------------------
user_query = st.chat_input("Share what is on your mind or ask about wellness resources...")
if user_query:
    # Append user message
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Prepare request payload
    payload = {
        "message": user_query,
        "conversation_id": st.session_state.conversation_id,
        "use_rag": use_rag,
        "synthesize_speech": enable_tts,
    }

    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Reflecting..."):
            try:
                resp = httpx.post(f"{BACKEND_URL}/chat", json=payload, timeout=25.0)
                if resp.status_code == 200:
                    chat_data = resp.json()
                    st.markdown(chat_data["reply"])

                    if chat_data.get("citations"):
                        with st.expander(f"📚 Grounding Citations ({len(chat_data['citations'])} excerpts)"):
                            for cite in chat_data["citations"]:
                                st.markdown(
                                    f"""
                                    <div class="citation-card">
                                        <strong>{cite['filename']}</strong> (Chunk #{cite['chunk_index']} | Score: {cite['score']})<br/>
                                        <em>"{cite['excerpt']}"</em>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                    if chat_data.get("audio_base64"):
                        try:
                            audio_bytes = base64.b64decode(chat_data["audio_base64"])
                            st.audio(audio_bytes, format="audio/wav")
                        except Exception:
                            pass

                    # Save to session history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": chat_data["reply"],
                        "citations": chat_data.get("citations", []),
                        "audio_base64": chat_data.get("audio_base64"),
                    })
                else:
                    st.error(f"Error ({resp.status_code}): {resp.text}")
            except Exception as exc:
                st.error(f"Unable to reach Aura backend server: {exc}")
