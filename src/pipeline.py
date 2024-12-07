# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Define dataset path
data_path = "/Users/ayesha/Desktop/Project/data/dataset.csv"

# Load the dataset
print("Loading dataset...")
data = pd.read_csv(data_path)

# Preprocessing
print("Preprocessing data...")
X = data.drop(columns=['Activity'])
y = data['Activity']

# Encode target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Normalize numerical features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Save the scaler and encoder for future use
joblib.dump(scaler, "/Users/ayesha/Desktop/Project/models/scaler.pkl")
joblib.dump(label_encoder, "/Users/ayesha/Desktop/Project/models/label_encoder.pkl")
print("Scaler and label encoder saved.")

# Train-test split
print("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save test data for evaluation
test_data = pd.DataFrame(X_test)
test_data['Activity'] = y_test
test_data.to_csv("/Users/ayesha/Desktop/Project/data/test_dataset.csv", index=False)
print("Test dataset saved to /Users/ayesha/Desktop/Project/data/test_dataset.csv")

# Initialize the model
print("Initializing the model...")
model = RandomForestClassifier(random_state=42)

# Train the model


def train_pipeline():
    train_data = pd.read_csv('/Users/ayesha/Desktop/Project/data/processed_train.csv')
    
    X = train_data.drop('Activity', axis=1)
    y = train_data['Activity']
    
    selector = SelectFromModel(RandomForestClassifier())
    X_selected = selector.fit_transform(X, y)
    joblib.dump(selector, '/Users/ayesha/Desktop/Project/models/feature_selector.pkl')
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_scaled, y)
    
    joblib.dump(rf, '/Users/ayesha/Desktop/Project/models/random_forest.pkl')
    joblib.dump(scaler, '/Users/ayesha/Desktop/Project/models/scaler.pkl')
print("Training the model...")
model.fit(X_train, y_train)


# Save training column names for later
joblib.dump(data.drop(columns=['Activity']).columns.tolist(), "/Users/ayesha/Desktop/Project/models/training_columns.pkl")

print("Trained model saved to /Users/ayesha/Desktop/Project/models/random_forest.pkl")

# Evaluate the model
print("Evaluating the model...")
y_pred = model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
