import pandas as pd
import os

# I resolve the expected raw CSV from the repository root because this script uses
# project-relative paths rather than paths relative to its own location.
data_path = os.path.join("ml_classifier", "data", "raw", "tagged-data.csv")

# I report a clear location when the expected input is absent instead of reading elsewhere.
if not os.path.exists(data_path):
    print(f"Error: Could not find dataset at {data_path}. Please move tagged-data.csv there.")
else:
    df = pd.read_csv(data_path)
    # These fields make the input schema and a small sample visible before preprocessing.
    print("--- DATASET LOADED SUCCESSFULLY ---")
    print(f"Total rows and columns: {df.shape}")
    print("\nColumn names:")
    print(df.columns.tolist())
    print("\nFirst 3 rows:")
    print(df.head(3))


    