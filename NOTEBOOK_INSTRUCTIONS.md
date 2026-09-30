# Master Notebook: Run Instructions

This notebook is the primary executable project deliverable. It includes a local copy of the `rag_llm/safety_policies.py` rules and its own vector search, rolling context, Gemini prompt, and lobby monitor. Running it does not require importing or starting the Streamlit app.

## 1. Prepare Python

From the repository root, use Python 3.10 or later:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The first install can take a while. `sentence-transformers` installs its model runtime; the first notebook run downloads `all-MiniLM-L6-v2` if it is not already cached. The model is used to embed the rule text for the in-memory ChromaDB collection.

## 2. Add the Gemini key

Copy `.env.example` to `.env` and put the key in either supported form:

```dotenv
GOOGLE_API_KEY=your_gemini_api_key_here
```

or:

```dotenv
GEMINI_API_KEY=your_gemini_api_key_here
```

The notebook accepts both variable names and never prints the key. Keep `.env` private; Git ignores `.env` and other local `.env.*` files while allowing `.env.example`.

The default model is `gemini-3.8-flash`. To use another model enabled for the key, set `GEMINI_MODEL` in `.env`. API quota and temporary service errors are reported as warnings; the message is still audited with the notebook's local policy fallback.

## 3. Run from Jupyter

1. Open `Guardian_AI_Master_Prototype.ipynb` in VS Code or JupyterLab.
2. Select the `.venv` Python kernel.
3. Run **Kernel -> Restart & Run All**.
4. Wait for the classical models and the vector model to finish setup. The TF-IDF classifier and ChromaDB store are created as part of the notebook run.
5. Review the seven-model leaderboard, confusion matrices, twelve-example benchmark, and final live lobby widget.

The notebook searches for `data/raw/tagged-data.csv`, `data/processed/train.csv`, and `data/processed/test.csv` first, and then falls back to the files under `ml_classifier/data/`. It writes the champion model and its fitted vectorizer to `models/champion_model.pkl`.

The benchmark is intentionally limited to twelve requests, with a 1.5-second pause between messages. It never sends the full test dataset to Gemini. If the vector dependencies or network are unavailable, a warning appears and the notebook uses local rule matching. If `ipywidgets` is unavailable, the final cell provides a terminal-style input loop.

## 4. Check the key without exposing it

The notebook reports whether a key was found, but never displays it. For an API test, run a single harmless prompt through the configured client; do not print environment values or paste a real key into notebook output.
