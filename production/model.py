# %%
# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define the dataset path
data_path = "/Users/ayesha/Desktop/Project/data/dataset.csv"
# In data_exploration.ipynb
test_data_path = "/Users/ayesha/Desktop/Project/data/test_dataset.csv"
test_data = pd.read_csv(test_data_path)
test_data_cleaned = preprocess_data(test_data)
test_data_cleaned.to_csv('/Users/ayesha/Desktop/Project/data/processed_test.csv', index=False)

# Load the dataset
data = pd.read_csv(data_path)

# Display the first few rows of the dataset
print("\nDataset Preview:\n", data.head())

# Get basic information about the dataset
print("\nDataset Information:")
data.info()

# Check for missing values
print("\nMissing Values:\n", data.isnull().sum())

# Summary statistics of numerical columns
print("\nSummary Statistics:\n", data.describe())

# Plot the distribution of the target variable (e.g., 'Activity')
if 'Activity' in data.columns:
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Activity', data=data)
    plt.title("Distribution of Activities")
    plt.xlabel("Activity Type")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.show()

# Check for class imbalance in the target variable
if 'Activity' in data.columns:
    print("\nClass Distribution:\n", data['Activity'].value_counts())

# Correlation heatmap for numerical features
# Exclude non-numeric columns for correlation computation
numeric_data = data.select_dtypes(include=['number'])
plt.figure(figsize=(12, 8))
sns.heatmap(numeric_data.corr(), cmap="coolwarm", annot=False, cbar=True)
plt.title("Correlation Heatmap")
plt.show()

# Check for duplicate rows
duplicates = data.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicates}")

# Save preprocessed dataset (Optional, if changes are made)
# data.to_csv("/Users/ayesha/Desktop/Project/data/cleaned_dataset.csv", index=False)

# End of data exploration


# %%
# Print column names to debug features
print("\nDataset Columns:\n", data.columns)

# Print dataset shape
print("Dataset Shape (Rows, Columns):", data.shape)

# Identify the target column (assuming it's the last column)
target_column = data.columns[-1]
print("\nTarget Column:", target_column)

# Number of features excluding the target column
num_features = data.shape[1] - 1
print("\nNumber of Features (excluding target column):", num_features)


# %% [markdown]
# for Evualation 

# %%
def preprocess_test_data(test_data):
    # Apply same preprocessing as training data
    test_data = test_data.fillna(0)
    
    # Remove outliers using IQR
    Q1 = test_data.quantile(0.25)
    Q3 = test_data.quantile(0.75)
    IQR = Q3 - Q1
    test_data = test_data[~((test_data < (Q1 - 1.5 * IQR)) | (test_data > (Q3 + 1.5 * IQR))).any(axis=1)]
    
    return test_data

# %% [markdown]
# adding feature Selection

# %%
def select_features(X_test, importance_threshold=0.01):
    feature_importances = pd.DataFrame({
        'feature': training_columns,
        'importance': rf.feature_importances_
    })
    selected_features = feature_importances[feature_importances['importance'] > importance_threshold]['feature']
    return X_test[selected_features]

# %% [markdown]
# modify evaluate model 

# %%
# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define the test dataset path
test_data_path = "/Users/ayesha/Desktop/Project/data/test_dataset.csv"

# Load the test dataset
test_data = pd.read_csv(test_data_path)

# Display the first few rows of the test dataset
print("\nTest Dataset Preview:\n", test_data.head())

# Get basic information about the test dataset
print("\nTest Dataset Information:")
test_data.info()

# Check for missing values in the test dataset
print("\nMissing Values in Test Dataset:\n", test_data.isnull().sum())

# Summary statistics of numerical columns in the test dataset
print("\nSummary Statistics for Test Dataset:\n", test_data.describe())

# Print column names to debug features in the test dataset
print("\nTest Dataset Columns:\n", test_data.columns)

# Print dataset shape for the test dataset
print("Test Dataset Shape (Rows, Columns):", test_data.shape)

# Identify the target column in the test dataset (assuming it's the last column)
test_target_column = test_data.columns[-1]
print("\nTarget Column in Test Dataset:", test_target_column)

# Number of features excluding the target column in the test dataset
num_test_features = test_data.shape[1] - 1
print("\nNumber of Features (excluding target column) in Test Dataset:", num_test_features)

# Plot the distribution of the target variable in the test dataset
if 'Activity' in test_data.columns:
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Activity', data=test_data)
    plt.title("Distribution of Activities in Test Dataset")
    plt.xlabel("Activity Type")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.show()

# Check for class imbalance in the target variable for the test dataset
if 'Activity' in test_data.columns:
    print("\nClass Distribution in Test Dataset:\n", test_data['Activity'].value_counts())

# Correlation heatmap for numerical features in the test dataset
# Exclude non-numeric columns for correlation computation
numeric_test_data = test_data.select_dtypes(include=['number'])
plt.figure(figsize=(12, 8))
sns.heatmap(numeric_test_data.corr(), cmap="coolwarm", annot=False, cbar=True)
plt.title("Correlation Heatmap for Test Dataset")
plt.show()

# Check for duplicate rows in the test dataset
test_duplicates = test_data.duplicated().sum()
print(f"\nNumber of duplicate rows in Test Dataset: {test_duplicates}")

# Save preprocessed test dataset (Optional, if changes are made)
# test_data.to_csv("/Users/ayesha/Desktop/Project/data/cleaned_test_dataset.csv", index=False)

# End of test dataset exploration


# %%
import pandas as pd
import joblib

# Define paths
test_data_path = "/Users/ayesha/Desktop/Project/data/test_dataset.csv"
aligned_test_data_path = "/Users/ayesha/Desktop/Project/data/aligned_test_dataset.csv"
training_columns_path = "/Users/ayesha/Desktop/Project/models/training_columns.pkl"

# Load the test dataset
test_data = pd.read_csv(test_data_path)

# Load training columns
training_columns = joblib.load(training_columns_path)

# Align test data columns with training columns
X_test = test_data.drop(columns=["Activity"], errors="ignore")  # Drop target column
X_test_aligned = X_test.reindex(columns=training_columns, fill_value=0)  # Reindex to match training columns

# Re-attach the target column if available
if "Activity" in test_data.columns:
    X_test_aligned["Activity"] = test_data["Activity"]

# Save aligned test dataset
X_test_aligned.to_csv(aligned_test_data_path, index=False)
print(f"Aligned test dataset saved to: {aligned_test_data_path}")


# %%
import joblib

# Load training columns
training_columns_path = "/Users/ayesha/Desktop/Project/models/training_columns.pkl"
training_columns = joblib.load(training_columns_path)

print("Training Columns:", training_columns)
print(f"Number of Training Columns: {len(training_columns)}")


# %%
import pandas as pd
import joblib

# Define paths
test_data_path = "/Users/ayesha/Desktop/Project/data/test_dataset.csv"
aligned_test_data_path = "/Users/ayesha/Desktop/Project/data/aligned_test_dataset_fixed.csv"
training_columns_path = "/Users/ayesha/Desktop/Project/models/training_columns.pkl"

# Load the test dataset
test_data = pd.read_csv(test_data_path)

# Load training columns
training_columns = joblib.load(training_columns_path)

# Align test data columns with training columns (exclude "Activity" initially)
X_test = test_data.drop(columns=["Activity"], errors="ignore")  # Drop target column
X_test_aligned = X_test.reindex(columns=training_columns, fill_value=0)  # Reindex to match training columns

# Re-attach the target column
if "Activity" in test_data.columns:
    X_test_aligned["Activity"] = test_data["Activity"]

# Save the fixed aligned test dataset
X_test_aligned.to_csv(aligned_test_data_path, index=False)
print(f"Aligned test dataset saved to: {aligned_test_data_path}")


# %%
import pandas as pd

# Load aligned test dataset
aligned_test_data_path = "/Users/ayesha/Desktop/Project/data/aligned_test_dataset.csv"
aligned_test_data = pd.read_csv(aligned_test_data_path)

print("Aligned Test Dataset Columns:", aligned_test_data.columns.tolist())
print(f"Number of Aligned Test Columns: {aligned_test_data.shape[1]}")


