# Aura Wellness — Multimodal RAG Chatbot MVP

An empathetic, evidence-informed workplace mental wellness and emotional resilience assistant powered by **Retrieval-Augmented Generation (RAG)**, **Multimodal AI** (Text, Vision, Speech-to-Text, and Text-to-Speech), and **Deterministic Safety Guardrails**.

---

## 🌟 Key Features

1. **Deterministic Safety & Crisis Interception**:
   - Acute self-harm, suicide ideation, and domestic violence detection using pre-generation regex patterns.
   - Provides verified international crisis hotlines (US/Canada 988, UK 111/999, EU 112, India 14416).
   - Medical diagnosis and prescription tampering boundaries.
   - 5-4-3-2-1 sensory grounding failsafe during simulated outages or timeouts.
2. **Multimodal Capabilities**:
   - **Text**: Empathetic conversational chat and structured RAG citations.
   - **Vision**: Analyzes uploaded workspace/mindfulness photos for grounding cues.
   - **Voice (STT & TTS)**: Transcribes speech input and synthesizes spoken audio responses.
3. **Pluggable AI Providers**:
   - `mock`: 100% offline development with realistic grounding synthesis, image analysis, and WAV tone generation.
   - `gemini`: Google Gemini 1.5 Flash (Text & Vision).
   - `openai`: GPT-4o-mini, Whisper-1 (STT), and TTS-1 (Audio speech).
4. **Dual Database Architecture**:
   - Native support for **MySQL** with full-text search indexing.
   - **Automated SQLite fallback** (`ai_chatbot.db`) requiring zero external setup for rapid local testing.
5. **Secure & Privacy-Conscious**:
   - Strict loopback localhost bindings (`127.0.0.1:8000` & `127.0.0.1:8501`).
   - Redacting logger scrubbing API keys, tokens, and authorization headers.
   - Session-based conversation privacy with on-demand purge (`/chat/clear`).
   - Untrusted reference data delimiters protecting against prompt injection.

---

## 📂 Project Structure

```text
AI-Smart-Chatbot/
├── .env.example              # Environment variables template
├── .gitignore                # Git ignore rules
├── requirements.txt          # Python dependencies
├── README.md                 # System documentation
├── data/
│   └── sample_knowledge_base.txt # Aura Wellness & Resilience Handbook v2.4
├── scripts/
│   └── init_db.py            # Database initialization and seed ingestion
├── src/
│   ├── config.py             # Typed settings, limits, and safe config summary
│   ├── database/
│   │   ├── schema.sql        # MySQL & SQLite DDL schema
│   │   ├── connection.py     # Database manager with MySQL pooling & SQLite fallback
│   │   └── repositories.py   # Parameterized SQL repositories (docs, chunks, chats)
│   ├── safety/
│   │   ├── guardrails.py     # Crisis, hotline, and medical boundary enforcement
│   │   ├── prompts.py        # System prompts and untrusted delimiter templates
│   │   └── logger.py         # Redacting logger filtering sensitive secrets
│   ├── rag/
│   │   ├── extractor.py      # PDF & TXT text extraction, validation, and sanitization
│   │   ├── chunker.py        # Sliding-window chunker respecting semantic boundaries
│   │   └── retriever.py      # Natural language & token-overlap search context assembler
│   ├── providers/
│   │   ├── base.py           # Abstract base classes for LLM, Vision, STT, and TTS
│   │   ├── mock.py           # Offline mock implementations
│   │   ├── gemini.py         # Google Gemini adapter
│   │   ├── openai.py         # OpenAI adapter
│   │   └── factory.py        # Provider factory and credential verification
│   ├── api/
│   │   ├── models.py         # Pydantic schemas for request/response validation
│   │   ├── routes.py         # REST endpoints (/health, /chat, /upload/*, etc.)
│   │   └── main.py           # FastAPI application entry point with CORS
│   └── frontend/
│       └── app.py            # Streamlit interactive UI
└── tests/
    ├── conftest.py           # Isolated test fixtures
    ├── test_api_health.py    # Health check and secret leak tests
    ├── test_guardrails.py    # Crisis regex, hotline, and medical tests
    ├── test_upload_validation.py # Extension and size validation tests
    ├── test_rag_pipeline.py  # Chunking, extraction, and retrieval tests
    └── test_chat_multimodal.py # End-to-end chat, vision, and audio tests
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- Python 3.10+ (Tested on Python 3.13)
- Git

### 2. Environment Setup
Clone the repository and install dependencies:

```powershell
# Create and activate virtual environment (optional)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install required packages
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Copy `.env.example` to `.env`:

```powershell
cp .env.example .env
```

Default configuration runs completely **offline with zero paid API keys**:
```ini
LLM_PROVIDER=mock
VISION_PROVIDER=mock
STT_PROVIDER=mock
TTS_PROVIDER=mock
DB_HOST=127.0.0.1
DB_PORT=3306
DB_NAME=ai_smart_chatbot
```
*Note: If MySQL is not running on port 3306, the application automatically falls back to local SQLite (`ai_chatbot.db`).*

### 4. Initialize Database & Seed Knowledge Base
Run the initialization script to prepare tables and ingest the corporate wellness handbook:

```powershell
python scripts/init_db.py --seed
```

---

## 🖥️ Running the Application

### 1. Launch the FastAPI Backend
Start the backend server on `127.0.0.1:8000`:

```powershell
python src/api/main.py
```
*(Alternative: `uvicorn src.api.main:app --host 127.0.0.1 --port 8000`)*

API Documentation is accessible at:
- Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Health Check: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

### 2. Launch the Streamlit Frontend
In a new terminal window:

```powershell
streamlit run src/frontend/app.py
```

The UI will open in your browser at [http://localhost:8501](http://localhost:8501).

---

## 🧪 Automated Testing

Execute the test suite offline using mocked providers:

```powershell
pytest -v
```

All 18 tests cover:
- Service health, database connectivity, and secret redaction.
- Acute crisis interception, location-aware hotlines, and medical boundary guardrails.
- Document extension whitelisting, empty file rejection, and 10MB size limits.
- RAG extraction, sliding-window chunking, and grounded context citation retrieval.
- Multimodal image reasoning, voice transcription, and synthesized WAV audio playback.

---

## 🛡️ Security & Privacy Architecture

- **Localhost Only**: Services bind exclusively to loopback addresses (`127.0.0.1`).
- **Input Sanitization**: File type verification and control character stripping on all uploads.
- **SQL Injection Prevention**: 100% parameterized queries across MySQL and SQLite.
- **Credential Protection**: Redacting loggers scrub keys (`AIzaSy...`, `sk-...`, `Bearer ...`).
- **Prompt Injection Defense**: Reference documents are enclosed in explicit untrusted delimiters instructing the model to treat content strictly as factual excerpts.
