import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import yaml
import logging
import os
import seaborn as sns
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config():
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
    logger.info(f"Loading config from: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            content = f.read().strip()
            if not content:
                raise ValueError("Config file is empty")
            return yaml.safe_load(content)
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        raise

def check_class_balance(y, title="Class Distribution"):
    """Analyze and visualize class distribution"""
    class_counts = pd.Series(y).value_counts()
    logger.info(f"\nClass Distribution:\n{class_counts}")
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=class_counts.index, y=class_counts.values)
    plt.title(title)
    plt.xlabel("Activity")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('class_distribution.png')
    
    return class_counts

def validate_features(X_train_cols, X_test):
    """Validate feature alignment between training and test sets"""
    missing_cols = set(X_train_cols) - set(X_test.columns)
    extra_cols = set(X_test.columns) - set(X_train_cols)
    
    if missing_cols:
        logger.warning(f"Missing columns in test set: {missing_cols}")
    if extra_cols:
        logger.warning(f"Extra columns in test set: {extra_cols}")
    
    return missing_cols, extra_cols

def validate_scaling(X_scaled):
    """Validate scaled features"""
    means = np.mean(X_scaled, axis=0)
    stds = np.std(X_scaled, axis=0)
    
    if not np.allclose(means, 0, atol=1e-2):
        logger.warning("Scaled features mean significantly differs from 0")
    if not np.allclose(stds, 1, atol=1e-2):
        logger.warning("Scaled features std significantly differs from 1")

def evaluate_model():
    

    config = load_config()
    logger.info("Loaded config successfully")

    # Load data and models
    # Change test data loading line to:
    test_data = pd.read_csv('/Users/ayesha/Desktop/Project/data/test_dataset.csv')


    rf = joblib.load(config['paths']['models']['random_forest'])
    scaler = joblib.load(config['paths']['models']['scaler'])
    label_encoder = joblib.load(config['paths']['models']['label_encoder'])
    training_columns = joblib.load(config['paths']['models']['columns'])

    # Handle activities encoding
    if test_data['Activity'].dtype in ['int64', 'float64']:
        test_data['Activity'] = label_encoder.inverse_transform(test_data['Activity'])

    X_test = test_data.drop('Activity', axis=1)
    y_test = test_data['Activity']

    # Check class balance
    logger.info("Checking class balance...")
    class_distribution = check_class_balance(y_test)

    # Validate features
    logger.info("Validating features...")
    missing_cols, extra_cols = validate_features(training_columns, X_test)
    X_test = X_test.reindex(columns=training_columns, fill_value=0)

    # Scale features
    logger.info("Scaling features...")
    X_test_scaled = scaler.transform(X_test)
    validate_scaling(X_test_scaled)



    # Predictions
    logger.info("Making predictions...")
    y_pred = rf.predict(X_test_scaled)
    y_pred_labels = label_encoder.inverse_transform(y_pred)

    # Evaluation
    accuracy = accuracy_score(y_test, y_pred_labels)
    report = classification_report(y_test, y_pred_labels)
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Classification Report:\n{report}")

    # Save results
    predictions = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred_labels
    })
    predictions.to_csv('predictions.csv', index=False)
    logger.info("Evaluation complete. Results saved.")

if __name__ == "__main__":
    evaluate_model()