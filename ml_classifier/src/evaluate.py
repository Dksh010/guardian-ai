"""
evaluate.py
-----------
Evaluates all 7 trained models on the held-out test split, computes comprehensive 
metrics (Accuracy, Macro Precision, Recall, Macro-F1), generates confusion matrices, 
and exports a master comparison CSV for your report.
"""

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def evaluate_models():
    """I evaluate saved classifiers once on held-out data and write metrics and plots."""
    # I keep test rows separate from fitting so these scores estimate held-out performance.
    test_path = os.path.join("ml_classifier", "data", "processed", "test.csv")
    print(f"Loading test data from {test_path}...")
    test_df = pd.read_csv(test_path)
    
    X_test = test_df['text'].fillna("")
    y_test = test_df['target']
    
    # I reuse the training vocabulary and IDF weights rather than fitting features on test text.
    vectorizer_path = "ml_classifier/src/models/tfidf_vectorizer.pkl"
    print(f"Loading TF-IDF vectorizer from {vectorizer_path}...")
    vectorizer = joblib.load(vectorizer_path)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # These names define the artifact lookup order and must match train.py's serialized files.
    model_names = [
        "Naive_Bayes", 
        "Logistic_Regression", 
        "Linear_SVM", 
        "Random_Forest", 
        "XGBoost", 
        "Neural_Network",
        "Voting_Ensemble"
    ]
    
    results = []
    # Evaluation artifacts share one directory so the table and per-model plots stay together.
    os.makedirs("comparison/results", exist_ok=True)
    
    print("\n--- EVALUATING 7 MODELS ON HELD-OUT TEST SET ---")
    for name in model_names:
        model_path = f"ml_classifier/src/models/{name.lower()}_model.pkl"
        # Missing artifacts are skipped so partially trained model sets can still be inspected.
        if not os.path.exists(model_path):
            print(f"Warning: Model file {model_path} not found. Skipping.")
            continue
            
        model = joblib.load(model_path)
        y_pred = model.predict(X_test_tfidf)
        
        # Macro averaging weights each class equally; zero_division keeps undefined class
        # precision/recall explicit as zero instead of aborting this evaluation.
        acc = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)
        
        print(f"-> {name:<20} | Accuracy: {acc:.4f} | Macro-F1: {f1:.4f}")
        
        results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Macro_Precision": round(precision, 4),
            "Macro_Recall": round(recall, 4),
            "Macro_F1": round(f1, 4)
        })
        
        # Rows in this matrix are true labels and columns are predicted labels.
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f"Confusion Matrix: {name}")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        
        plot_path = f"comparison/results/confusion_matrix_{name.lower()}.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()

    # The report is ranked by macro-F1 so minority classes contribute equally to model order.
    results_df = pd.DataFrame(results).sort_values(by="Macro_F1", ascending=False)
    csv_output = "comparison/results/model_comparison.csv"
    results_df.to_csv(csv_output, index=False)
    
    print(f"\nMaster comparison table saved successfully to {csv_output}!")
    print("\nFinal Test Set Leaderboard:")
    print(results_df.to_string(index=False))

if __name__ == "__main__":
    evaluate_models()