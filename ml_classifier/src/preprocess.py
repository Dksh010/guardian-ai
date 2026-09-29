"""
preprocess.py
-------------
Handles text cleaning, normalisation, and stratified train/val/test splitting
for the GOSU.ai gaming chat dataset, strictly enforcing reproducibility (random_state=42).
"""

import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split

def clean_text(text: str) -> str:
    """
    Cleans raw chat text by lowercasing, stripping URLs/HTML, 
    expanding basic contractions, and normalising whitespace.
    """
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Strip HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Strip URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Basic contractions and normalisation
    text = re.sub(r"can't", "can not", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"'s", " is", text)
    text = re.sub(r"'d", " would", text)
    text = re.sub(r"'ll", " will", text)
    text = re.sub(r"'t", " not", text)
    text = re.sub(r"'ve", " have", text)
    text = re.sub(r"'m", " am", text)
    
    # Remove special characters/punctuation keep letters/numbers/spaces
    text = re.sub(r'[^a-z0-9\s]', '', text)
    
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def load_and_preprocess_data(filepath: str):
    """
    Loads raw CSV data, drops nulls, applies text cleaning, 
    and executes a stratified 80/10/10 split.
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Using the exact columns from your inspect script: 'text' and 'target'
    df = df.dropna(subset=['text', 'target'])
    
    print("Cleaning text data...")
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # Remove empty strings after cleaning
    df = df[df['cleaned_text'].str.len() > 0]
    
    X = df['cleaned_text']
    y = df['target'].astype(int)
    
    print(f"Class distribution:\n{y.value_counts(normalize=True)}")
    
    print("Performing stratified 80/10/10 split (random_state=42)...")
    # First split: 80% train, 20% temp (for val/test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # Second split: 50% val, 50% test of that 20% (giving 10% / 10% overall)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    
    print(f"Split complete -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Save processed splits for easy loading later
    os.makedirs(os.path.join("ml_classifier", "data", "processed"), exist_ok=True)
    pd.DataFrame({'text': X_train, 'target': y_train}).to_csv("ml_classifier/data/processed/train.csv", index=False)
    pd.DataFrame({'text': X_val, 'target': y_val}).to_csv("ml_classifier/data/processed/val.csv", index=False)
    pd.DataFrame({'text': X_test, 'target': y_test}).to_csv("ml_classifier/data/processed/test.csv", index=False)
    print("Processed splits saved successfully to ml_classifier/data/processed/")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == "__main__":
    csv_path = os.path.join("ml_classifier", "data", "raw", "tagged-data.csv")
    load_and_preprocess_data(csv_path)