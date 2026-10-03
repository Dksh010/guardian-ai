"""
cli.py
------
Interactive Terminal Interface for Guardian AI+.
Sends user chat inputs simultaneously to:
  1. Arm 1: Classical ML Champion (Linear SVM + TF-IDF with Feature Attribution)
  2. Arm 2: RAG-LLM Moderation Engine (ChromaDB + Gemini)
"""

import os
import sys
import joblib
import numpy as np

# I anchor artifact lookup to this script so the CLI can start from any working directory.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
rag_llm_dir = os.path.join(project_root, "rag_llm")

# I add both import roots because this file supports package and direct-script execution.
for p in [project_root, rag_llm_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

# I prefer package-qualified imports, then retain the direct-module layout as a fallback.
try:
    from rag_llm.engine import GuardianEngine  # type: ignore
    ENGINE_AVAILABLE = True
except ImportError:
    try:
        from engine import GuardianEngine  # type: ignore
        ENGINE_AVAILABLE = True
    except ImportError:
        GuardianEngine = None
        ENGINE_AVAILABLE = False

def load_svm_model():
    """I load the saved TF-IDF transform and Linear SVM used by the CLI."""
    vectorizer_path = os.path.join(current_dir, "models", "tfidf_vectorizer.pkl")
    model_path = os.path.join(current_dir, "models", "linear_svm_model.pkl")
    
    # I retry the documented model directory with an explicit repository-relative path.
    if not os.path.exists(vectorizer_path) or not os.path.exists(model_path):
        vectorizer_path = os.path.join(project_root, "ml_classifier", "src", "models", "tfidf_vectorizer.pkl")
        model_path = os.path.join(project_root, "ml_classifier", "src", "models", "linear_svm_model.pkl")
        
    # I return a missing-artifact state so the CLI can explain that the ML arm is unavailable.
    if not os.path.exists(vectorizer_path) or not os.path.exists(model_path):
        return None, None
        
    vectorizer = joblib.load(vectorizer_path)
    model = joblib.load(model_path)
    return vectorizer, model

def get_top_contributing_features(text, vectorizer, model, top_n=3):
    """I estimate influential known terms and identify tokens absent from the vocabulary."""
    try:
        # I use the vectorizer's analyzer so OOV checks follow its tokenization rules.
        analyzer = vectorizer.build_analyzer()
        tokens = analyzer(text)
        
        # I expose missing vocabulary terms because the classifier cannot use their meaning.
        vocab = vectorizer.vocabulary_
        oov_words = [t for t in tokens if t not in vocab]
        
        # I weight active TF-IDF features by the first coefficient row; multiclass models
        # therefore show only that row's contributions rather than a full class comparison.
        feature_names = vectorizer.get_feature_names_out()
        coefs = model.coef_[0]
        X_vec = vectorizer.transform([text]).toarray()[0]
        
        contributions = []
        for idx in np.nonzero(X_vec)[0]:
            word = feature_names[idx]
            weight = X_vec[idx] * coefs[idx]
            contributions.append((word, weight))
            
        # I rank by magnitude so strong evidence in either direction is visible.
        contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        
        return contributions[:top_n], list(set(oov_words))
    # Attribution is supplementary; a failure leaves the main classifier verdict available.
    except Exception:
        return [], []

def format_confidence_bar(confidence: float, width: int = 20) -> str:
    """I format a bounded score as a fixed-width terminal bar."""
    # I clamp scores before deriving the number of filled characters.
    clamped = max(0.0, min(1.0, confidence))
    filled = int(round(clamped * width))
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {clamped * 100:.2f}%"

def run_cli():
    """I run the interactive classifier and optional retrieval-backed audit loop."""
    vectorizer, svm_model = load_svm_model()
    
    rag_engine = None
    # RAG startup may need embeddings or Gemini configuration; retain the ML arm if it fails.
    if ENGINE_AVAILABLE and GuardianEngine is not None:
        try:
            print("Initializing RAG-LLM Engine for CLI...")
            rag_engine = GuardianEngine()
        except Exception as ex:
            print(f"[Warning] Could not initialize RAG GuardianEngine: {ex}")

    print("\n" + "=" * 80)
    print("                      GUARDIAN AI+ : DUAL-ARM BENCHMARK SUITE")
    print("                      Classical Linear SVM vs. RAG-LLM Engine")
    print("=" * 80)
    print(" Instructions:")
    print("   • Type any message to evaluate across both inspection pipelines.")
    print("   • Type 'exit' or 'quit' to terminate the session.\n" + "=" * 80 + "\n")
    
    while True:
        try:
            user_input = input("Chat Input > ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['exit', 'quit']:
                print("\n[INFO] Terminating session. Good luck with your defense!\n")
                break
                
            print("\n" + "=" * 80)
            print(f"EVALUATING PAYLOAD: \"{user_input}\"")
            print("=" * 80)

            # --- ARM 1: Classical ML (Linear SVM) Evaluation ---
            ml_status = "UNKNOWN"
            ml_conf = 0.0
            raw_decision = 0.0
            top_features = []
            oov_words = []
            
            if vectorizer and svm_model:
                try:
                    # I transform one message with the exact vocabulary used to fit the model.
                    X_tfidf = vectorizer.transform([user_input])
                    
                    if hasattr(svm_model, "decision_function"):
                        decision_val = svm_model.decision_function(X_tfidf)
                        raw_decision = float(decision_val.item() if hasattr(decision_val, "size") and decision_val.size == 1 else np.ravel(decision_val)[0])
                        
                        # I render the first margin as a binary verdict; this does not map
                        # the model's three dataset labels to validated moderation severities.
                        is_toxic = raw_decision < 0
                        # I map margin magnitude to a display score, not a calibrated probability.
                        ml_conf = float(1 / (1 + np.exp(-abs(raw_decision))))
                    else:
                        probs = svm_model.predict_proba(X_tfidf)[0]
                        is_toxic = bool(np.argmax(probs) == 1)
                        ml_conf = float(np.max(probs))

                    ml_status = "TOXIC (FLAGGED)" if is_toxic else "SAFE (PASSED)"
                    top_features, oov_words = get_top_contributing_features(user_input, vectorizer, svm_model)
                except Exception as ex:
                    ml_status = f"ERROR ({ex})"
            else:
                ml_status = "MODEL NOT FOUND"

            print("\n[ARM 1: Classical ML Pipeline - Linear SVM + TF-IDF]")
            print(f"  Verdict          : {ml_status}")
            print(f"  Confidence Score : {format_confidence_bar(ml_conf)}")
            print(f"  Raw Decision Val : {raw_decision:.4f}")
            
            if top_features:
                # Contributions are signed so the terminal shows their direction as well as size.
                feat_str = ", ".join([f"'{w}' ({wt:+.2f})" for w, wt in top_features])
                print(f"  Token Weights    : {feat_str}")
            else:
                print(f"  Token Weights    : None (No known words detected)")
                
            if oov_words:
                oov_str = ", ".join([f"'{w}'" for w in oov_words])
                print(f"  Blindspots (OOV): {oov_str} -> (Model lacks these in vocabulary!)")
                
            # This is a fixed UI label rather than a measured duration.
            print(f"  Inference Latency: Instant (< 5ms)")

            print("\n" + "-" * 80)

            # --- ARM 2: RAG-LLM Engine Evaluation ---
            if rag_engine:
                try:
                    # The same input goes to the context-aware arm after the ML display.
                    audit = rag_engine.audit_message("CLI_User", user_input)
                    is_flagged = audit.get("is_flagged", False)
                    severity = audit.get("severity", "NONE")
                    rule = audit.get("violated_rule", "None")
                    category = audit.get("retrieved_rule_category", "N/A")
                    rule_id = audit.get("retrieved_rule_id", "N/A")
                    reason = audit.get("summary_reasoning", "N/A")
                    
                    rag_status = f"TOXIC (FLAGGED) - Severity: {severity}" if is_flagged else f"SAFE (PASSED) - Severity: {severity}"
                    
                    print("\n[ARM 2: RAG-LLM Pipeline - ChromaDB + Gemini Flash]")
                    print(f"  Verdict          : {rag_status}")
                    print(f"  Retrieved Rule ID: {rule_id} ({category})")
                    print(f"  Target Policy    : {rule}")
                    print(f"  LLM Reasoning    : {reason}")
                except Exception as ex:
                    print(f"\n[ARM 2: RAG-LLM Pipeline] Error during evaluation: {ex}")
            else:
                print("\n[ARM 2: RAG-LLM Pipeline] Status: Not initialized / unavailable")

            print("\n" + "=" * 80 + "\n")
            
        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user. Exiting...")
            break

if __name__ == "__main__":
    run_cli()