# Streamlit Dashboard: Run Instructions

The Streamlit dashboard is a separate interface from the master notebook. Run it from the repository root; it does not need a notebook kernel.

## 1. Install dependencies

From the repository root with Python 3.10 or later:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The requirements include Streamlit, the Gemini SDK, ChromaDB, Sentence Transformers, and Pydantic. On first launch, Sentence Transformers downloads `all-MiniLM-L6-v2` for the vector store.

## 2. Configure Gemini

Create `.env` at the repository root with:

```dotenv
GOOGLE_API_KEY=your_gemini_api_key_here
```

The modular `rag_llm` Gemini client uses `GOOGLE_API_KEY`. Keep the real key in `.env`; it is ignored by Git.

## 3. Start the app

With the virtual environment active and the current directory set to the repository root:

```powershell
streamlit run rag_llm/app.py
```

Streamlit prints a local URL in the terminal and opens the dashboard in a browser. Use the quick scenarios to check clean banter, filter evasion, harassment, and child-safety context. Use **Reset Lobby** to clear the chat buffer.

To stop the app, return to the terminal and press **Ctrl+C**. The Jupyter notebook and Streamlit dashboard can be run independently.
