import pandas as pd
import joblib

# Paths
aligned_test_data_path = "/Users/ayesha/Desktop/Project/data/aligned_test_dataset_fixed.csv"

training_columns_path = "/Users/ayesha/Desktop/Project/models/training_columns.pkl"

# Load data
aligned_test_data = pd.read_csv(aligned_test_data_path)
training_columns = joblib.load(training_columns_path)

# Identify extra and missing columns
aligned_test_columns = aligned_test_data.columns.tolist()
extra_cols = set(aligned_test_columns) - set(training_columns)
missing_cols = set(training_columns) - set(aligned_test_columns)

print("Extra Columns in Test Dataset:", extra_cols)
print("Missing Columns in Test Dataset:", missing_cols)
