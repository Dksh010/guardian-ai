# Guardian AI

**Guardian AI: Dual-Arm Esports Toxicity Classification & Safety Engine** is an academic prototype comparing a classical TF-IDF machine-learning pipeline with a retrieval-augmented Gemini moderation arm.

## Primary executable deliverable

[`Guardian_AI_Master_Prototype.ipynb`](./Guardian_AI_Master_Prototype.ipynb) is the primary executable deliverable. The Python modules in `ml_classifier/` and `rag_llm/` are supporting modular architecture artifacts and remain part of the project.

### Run the notebook

1. Use Python 3.10 or later and open a terminal at the repository root.
2. (Recommended) Create and activate a virtual environment:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Install the notebook dependencies:

   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

4. Open `Guardian_AI_Master_Prototype.ipynb` in Jupyter or VS Code and select the Python environment from step 2.
5. Optionally copy `.env.example` to `.env` and replace the placeholder with a Gemini API key. Keep `.env` private; do not commit a real API key. Without a key, the RAG arm uses the local rule-based fallback and the notebook still runs.
6. Select **Kernel -> Restart & Run All**. The notebook loads the existing data, evaluates seven classical models, writes `models/champion_model.pkl`, builds the RAG rule index, benchmarks exactly 12 curated examples, and displays the live chat interface.
7. If `ipywidgets` does not render in the notebook, enable its Jupyter support and rerun the final cell. A non-blocking text-loop alternative is available by calling `text_chat_loop()` after the final cell.

The notebook resolves data files from root-level `data/raw/` and `data/processed/` first, then falls back to the existing `ml_classifier/data/` locations. It is intended to be launched from the repository root or one of its subdirectories. The in-memory ChromaDB collection and SentenceTransformer may download the `all-MiniLM-L6-v2` model the first time; if unavailable, a visible warning is issued and a local lexical rule retriever is used. `requirements.txt` includes the full notebook, RAG, Gemini, and optional Streamlit app dependencies. Sentence Transformers downloads its model files when the vector store is first initialized.

The Gemini benchmark is intentionally limited to 12 messages and waits 1.5 seconds between requests. API quota, network, and response-format errors are surfaced as warnings; an individual failed request uses the local safety-rule fallback rather than stopping evaluation.

The notebook imports the moderation policy, vector-store setup, chat buffer, prompt schema, engine, and quick-test scenarios from `rag_llm/`. It does not run `test_gemini.py` automatically because that smoke test sends an additional API request. To run the separate Streamlit interface, use `streamlit run rag_llm/app.py` from the repository root.

## Project layout

```text
Guardian_AI_Master_Prototype.ipynb   # Primary executable prototype
data/                                # Optional root-level canonical data location
models/champion_model.pkl            # Created by the notebook
ml_classifier/data/                  # Existing raw and processed data (fallback)
ml_classifier/src/                   # Modular classical ML scripts
rag_llm/                              # Modular RAG/Gemini software artifacts
```

## Data labels

The notebook evaluates the dataset's numeric targets as **0: Neutral/Clean**, **1: Mild Toxicity**, and **2: Severe Toxicity**. It reports held-out accuracy, macro-precision, macro-recall, macro-F1, and confusion matrices. The hand-curated 12-message benchmark is a small assignment evaluation set and should not be interpreted as a population-level estimate.
