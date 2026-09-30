# Guardian AI

**Guardian AI: Dual-Arm Esports Toxicity Classification & Safety Engine** compares a classical TF-IDF classifier with a policy-aware Gemini moderation arm.

## Choose how to run it

- The primary academic deliverable is [`Guardian_AI_Master_Prototype.ipynb`](./Guardian_AI_Master_Prototype.ipynb). Its retrieval rules, Gemini prompt, rolling chat buffer, and live lobby view are inside the notebook. It does not import `rag_llm/` to run.
- The separate Streamlit dashboard is [`rag_llm/app.py`](./rag_llm/app.py). It is a standalone app and does not need the notebook to run.
- The Python modules in `ml_classifier/` and `rag_llm/` are retained as supporting software architecture.

## Shared setup

Use Python 3.10 or later and open PowerShell at the repository root:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set a valid Gemini key. The notebook accepts `GOOGLE_API_KEY` or `GEMINI_API_KEY`; the Streamlit modules currently use `GOOGLE_API_KEY`. `.env` is ignored by Git. Do not commit the real key.

## Run the master notebook

Open `Guardian_AI_Master_Prototype.ipynb` in Jupyter or VS Code, choose the environment above, and select **Kernel -> Restart & Run All**. For the full setup, data locations, API model, and live-chat instructions, see [`NOTEBOOK_INSTRUCTIONS.md`](./NOTEBOOK_INSTRUCTIONS.md).

The notebook prefers `data/raw/` and `data/processed/`, then checks the existing `ml_classifier/data/` paths. It trains and scores seven models, saves `models/champion_model.pkl`, builds an in-memory ChromaDB index, and compares both arms on exactly twelve curated messages. It does not send the entire test split to Gemini. If a vector download or API call is unavailable, the notebook shows a warning and uses its local rule fallback.

## Run the Streamlit dashboard

From the repository root with the shared virtual environment active:

```powershell
streamlit run rag_llm/app.py
```

The browser dashboard runs independently from Jupyter. See [`rag_llm/STREAMLIT_INSTRUCTIONS.md`](./rag_llm/STREAMLIT_INSTRUCTIONS.md) for its API key, dependency, and launch details.

## Dataset labels

The numeric dataset targets are **0: Neutral/Clean**, **1: Mild Toxicity**, and **2: Severe Toxicity**. The notebook reports held-out accuracy, macro-precision, macro-recall, macro-F1, and confusion matrices. Its hand-curated twelve-message benchmark is a small project comparison, not a population-level estimate.
