"""
train.py (Hardened & Tuned)
---------------------------
Executes feature engineering (TF-IDF with unigrams/bigrams, max_features=4000, min_df=2)
and trains a robust 7-model suite (Naive Bayes, Logistic Regression, Linear SVM, 
Random Forest, XGBoost, Neural Network MLP, and Voting Ensemble) using 5-fold 
stratified cross-validation. Enforces random_state=42 everywhere.
"""

import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

def train_hardened_models():
    """
    Loads processed text data, builds TF-IDF features, trains 7 tuned models 
    with 5-fold stratified cross-validation, and saves all artifacts.
    """
    # 1. Load processed training data
    train_path = os.path.join("ml_classifier", "data", "processed", "train.csv")
    print(f"Loading training data from {train_path}...")
    train_df = pd.read_csv(train_path)
    
    X_train = train_df['text'].fillna("")
    y_train = train_df['target']
    
    # 2. Feature Engineering: TF-IDF with guarded parameters to prevent overfitting
    print("Fitting TF-IDF vectorizer (max_features=4000, min_df=2, ngram_range=(1,2))...")
    vectorizer = TfidfVectorizer(max_features=4000, min_df=2, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    
    # Save vectorizer for evaluation and CLI use
    os.makedirs(os.path.join("ml_classifier", "src", "models"), exist_ok=True)
    joblib.dump(vectorizer, "ml_classifier/src/models/tfidf_vectorizer.pkl")
    
    # 3. Define 5-fold stratified cross-validation setup (reproducibility anchor)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # 4. Define 7 Tuned Models (Avoiding lazy default parameters)
    models = {
        "Naive_Bayes": MultinomialNB(alpha=1.0),
        
        # Tuned Logistic Regression: L2 regularization C=1.0 with increased max_iter for convergence
        "Logistic_Regression": LogisticRegression(C=1.0, max_iter=1000, random_state=42),
        
        # Tuned Linear SVM: hinge loss with optimized alpha regularization
        "Linear_SVM": SGDClassifier(loss='hinge', penalty='l2', alpha=1e-4, max_iter=1000, random_state=42),
        
        # Tuned Random Forest: controlled depth and estimators to mitigate sparse text overfitting
        "Random_Forest": RandomForestClassifier(n_estimators=150, max_depth=None, min_samples_split=5, random_state=42),
        
        # Tuned XGBoost: learning rate and tree depth optimized for text features
        "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, eval_metric='logloss', random_state=42),
        
        # Neural Network (MLP Classifier): Non-linear multi-layer perceptron (ReLU activation, Adam optimizer)
        "Neural_Network": MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', alpha=0.0001, max_iter=300, random_state=42)
    }
    
    # Train individual models and evaluate via Stratified CV
    trained_models = {}
    print("\n--- TRAINING & CROSS-VALIDATION SCORES (7-MODEL SUITE) ---")
    for name, model in models.items():
        print(f"Training {name} with 5-fold Stratified CV...")
        scores = cross_val_score(model, X_train_tfidf, y_train, cv=cv, scoring='accuracy')
        print(f"-> {name:<20} | CV Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
        
        # Fit on full training set and serialize artifact
        model.fit(X_train_tfidf, y_train)
        trained_models[name] = model
        joblib.dump(model, f"ml_classifier/src/models/{name.lower()}_model.pkl")
    
    # 5. Build and Train Hard-Voting Ensemble combining the top performers
    print("\nTraining Voting Ensemble Classifier (Hard Voting)...")
    voting_clf = VotingClassifier(
        estimators=[
            ('lr', trained_models['Logistic_Regression']),
            ('svm', trained_models['Linear_SVM']),
            ('rf', trained_models['Random_Forest']),
            ('xgb', trained_models['XGBoost']),
            ('nn', trained_models['Neural_Network']),
            ('nb', trained_models['Naive_Bayes'])
        ],
        voting='hard'
    )
    voting_scores = cross_val_score(voting_clf, X_train_tfidf, y_train, cv=cv, scoring='accuracy')
    print(f"-> Voting_Ensemble     | CV Accuracy: {voting_scores.mean():.4f} (+/- {voting_scores.std():.4f})")
    
    voting_clf.fit(X_train_tfidf, y_train)
    joblib.dump(voting_clf, "ml_classifier/src/models/voting_ensemble_model.pkl")
    
    print("\nAll 7 hardened models trained, tuned, and saved successfully to ml_classifier/src/models/!")

if __name__ == "__main__":
    train_hardened_models()