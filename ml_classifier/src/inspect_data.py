import pandas as pd
import os

data_path = os.path.join("ml_classifier", "data", "raw", "tagged-data.csv")

if not os.path.exists(data_path):
    print(f"Error: Could not find dataset at {data_path}. Please move tagged-data.csv there.")
else:
    df = pd.read_csv(data_path)
    print("--- DATASET LOADED SUCCESSFULLY ---")
    print(f"Total rows and columns: {df.shape}")
    print("\nColumn names:")
    print(df.columns.tolist())
    print("\nFirst 3 rows:")
    print(df.head(3))


    