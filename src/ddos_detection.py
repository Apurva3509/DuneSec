# ddos_detection.py

import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    roc_curve, auc
)
from xgboost import XGBClassifier
import joblib

def create_run_directory():
    """
    Creates a unique directory based on the current timestamp
    for saving plots, model artifacts, etc.
    Returns the created directory path as a string.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"run_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def save_model(model, filepath):
    """
    Save the trained model to disk using joblib.
    """
    joblib.dump(model, filepath)
    print(f"Model saved to '{filepath}'.")

def load_data(csv_path):
    """
    Load dataset from a CSV file into a pandas DataFrame.
    """
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Data loaded. Shape: {df.shape}")
    df.columns = [col.strip() for col in df.columns]  # strip whitespace
    return df

def data_cleaning(df):
    """
    Clean the dataset: handle missing values, duplicates, and infinite values.
    """
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    print(f"Shape after cleaning: {df.shape}")
    return df

def exploratory_data_analysis(df, run_dir):
    """
    Basic EDA: descriptive stats, label distribution, correlation heatmap, etc.
    Saves the label distribution plot in run_dir.
    """
    print("\n--- Exploratory Data Analysis ---\n")
    
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        print("Descriptive Statistics (numeric columns):")
        print(numeric_df.describe().transpose())
    else:
        print("No numeric columns found.")
    
    obj_df = df.select_dtypes(include=[object])
    if not obj_df.empty:
        print("\nDescriptive Statistics (object columns):")
        print(obj_df.describe().transpose())
    
    # Label distribution
    if "Label" in df.columns:
        label_counts = df["Label"].value_counts()
        print("\nLabel Distribution:")
        print(label_counts)

        plt.figure()
        sns.countplot(x="Label", data=df)
        plt.title("Label Distribution")
        label_dist_path = os.path.join(run_dir, "label_distribution.png")
        plt.savefig(label_dist_path)
        plt.show()
    else:
        print("\n'Label' column not found in the DataFrame for EDA.")
    
    # Correlation among numeric features
    if not numeric_df.empty:
        plt.figure(figsize=(30, 30))
        corr_matrix = numeric_df.corr()
        sns.heatmap(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1)
        plt.title("Feature Correlation Heatmap")
        corr_heatmap_path = os.path.join(run_dir, "correlation_heatmap.png")
        plt.savefig(corr_heatmap_path)
        plt.show()

def feature_engineering(df):
    """
    Label-encode 'Label' and drop non-numeric columns 
    that are not needed for modeling (Flow ID, IPs, Timestamp, etc.).
    """
    if "Label" in df.columns:
        le = LabelEncoder()
        df["Label"] = le.fit_transform(df["Label"])
    else:
        print("Warning: 'Label' column not found, skipping label encoding.")
    
    # Drop columns that are not useful or are string-based
    drop_cols = ["Flow ID", "Source IP", "Destination IP", "Timestamp"]
    df.drop(columns=drop_cols, errors="ignore", inplace=True)
    
    # If there are still any object columns that you want to keep,
    # you could do one-hot encoding here.
    
    return df

def split_and_scale(df):
    """
    Split the dataset into train/test sets and apply StandardScaler to numeric features.
    """
    if "Label" not in df.columns:
        raise ValueError("The DataFrame must contain a 'Label' column for classification.")
    
    X = df.drop(columns=["Label"])
    y = df["Label"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, scaler

def train_and_tune_model(X_train, y_train):
    """
    Train an XGBoost model with GridSearchCV for hyperparameter tuning.
    Returns the best model and best_params (for logging).
    """
    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42
    )
    
    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [4, 6],
        "learning_rate": [0.1, 0.01],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0]
    }
    
    grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        scoring="f1",
        cv=3,
        verbose=1,
        n_jobs=-1
    )
    
    print("Starting hyperparameter tuning...")
    grid_search.fit(X_train, y_train)
    print("Tuning completed.")
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    print("Best params:", best_params)
    print("Best score:", best_score)
    
    return best_model, best_params, best_score

def evaluate_model(model, X_test, y_test, run_dir):
    """
    Evaluate the trained model on the test set and display classification metrics.
    Also plots & saves ROC curve and confusion matrix into run_dir.
    """
    print("\n--- Model Evaluation ---\n")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # for ROC-AUC
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc_val = roc_auc_score(y_test, y_proba)
    
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1-score:", f1)
    print("ROC AUC:", roc_auc_val)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    cm_path = os.path.join(run_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.show()
    
    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc_curve = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc_curve:.2f})")
    plt.plot([0,1], [0,1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    roc_curve_path = os.path.join(run_dir, "roc_curve.png")
    plt.savefig(roc_curve_path)
    plt.show()

def main():
    # suppress_warnings()

    # 1. Create a unique folder for this run
    run_dir = create_run_directory()
    print(f"Created run directory: {run_dir}")
    
    # 2. Load data
    csv_path = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
    df = load_data(csv_path)
    
    # 3. Clean data
    df = data_cleaning(df)
    
    # 4. EDA (saves label/correlation plots in run_dir)
    exploratory_data_analysis(df, run_dir)
    
    # 5. Feature Engineering
    df = feature_engineering(df)
    
    # 6. Split & Scale
    X_train, X_test, y_train, y_test, scaler = split_and_scale(df)
    
    # 7. Train & Tune Model
    best_model, best_params, best_score = train_and_tune_model(X_train, y_train)
    
    # 8. Evaluate Model (saves confusion matrix & ROC curve in run_dir)
    evaluate_model(best_model, X_test, y_test, run_dir)
    
    # 9. Save the trained model in the new run folder
    model_path = os.path.join(run_dir, "best_xgb_model-v1.pkl")
    save_model(best_model, model_path)
    
    # 10. Save best hyperparams to a text file
    hyperparam_file = os.path.join(run_dir, "best_hyperparams.txt")
    with open(hyperparam_file, "w") as f:
        f.write("Best Hyperparameters:\n")
        f.write(str(best_params) + "\n")
        f.write(f"Best F1 Score (CV): {best_score:.4f}\n")
    print(f"Best hyperparams saved to '{hyperparam_file}'")
    
    # 11. Basic Monitoring
    print("\n--- Basic Monitoring Discussion ---")
    print("1. Track model output distribution (fraction of DDoS) over time.")
    print("2. Log predictions/metrics for analysis over time.")
    print("3. Periodically retrain or fine-tune if new labeled data arrives.")

if __name__ == "__main__":
    main()
