# 🛡️ Guardian AI — Real-Time Game Moderation Engine

An intelligent, context-aware Trust & Safety moderation engine for multiplayer gaming environments. Powered by **ChromaDB** vector search and **Gemini 3.5 Flash** structured outputs.

---

## 🌟 Key Features

* **Context-Aware Safety Audits:** Evaluates player messages within ongoing lobby context to detect subtle toxicity, grooming, and policy violations.
* **RAG-Powered Policy Enforcement:** Queries a vector database (**ChromaDB** using `all-MiniLM-L6-v2`) to pull exact platform guidelines dynamically before auditing.
* **Evasion & Leetspeak Resilience:** Detects obfuscated toxic phrasing (e.g., `g0 k1ll ur53lf`) using LLM semantic understanding.
* **Child Safety Priority:** High-priority detection for grooming, private info solicitation, and off-platform redirection.
* **Real-Time Interactive Dashboard:** A modern **Streamlit** dashboard with live chat feed, automated risk categorization, and RAG match visualizer.

---

## 🏗️ Architecture Overview

1. **Ingestion & Retrieval (`vector_db.py`):** Safety policies are indexed into ChromaDB. Incoming messages query top matching safety policies using cosine distance thresholds to avoid false positives.
2. **Context Engine (`engine.py`):** Maintains rolling chat session history and formats RAG context into structured safety prompts.
3. **Structured Audit (`prompts.py`):** Gemini evaluates chat state and outputs strict JSON matching the `SafetyAuditResult` schema.
4. **Moderation UI (`app.py`):** Streamlit interface displays live chat streams, action metrics, policy matches, and reasoning.

---

## 🚀 Quick Start

### 1. Prerequisites
* Python 3.10+
* A Google Gemini API Key

### 2. Installation & Setup

```bash
# Clone the repository
git clone [https://github.com/YOUR_USERNAME/guardian-ai.git](https://github.com/YOUR_USERNAME/guardian-ai.git)
cd guardian-ai

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install streamlit chromadb google-genai pydantic sentence-transformers