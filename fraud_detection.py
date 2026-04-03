import pandas as pd
import numpy as np
from pyod.models.auto_encoder import AutoEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt

def main():
    print("Loading dataset...")
    # 1. Load Dataset (Ensure creditcard.csv is in the directory)
    try:
        df = pd.read_csv('creditcard.csv')
        print("Dataset loaded successfully.")
    except FileNotFoundError:
        print("Error: 'creditcard.csv' not found. Please ensure it is in the current directory.")
        return

    # 2. Preprocessing
    print("Preprocessing data...")
    scaler = StandardScaler()
    df['Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    df = df.drop(['Time'], axis=1)

    X = df.drop(['Class'], axis=1)
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Build & Train AutoEncoder
    # Contamination is the approximate ratio of outliers (fraudulent transactions)
    # The creditcard dataset has 492 frauds out of 284,807 transactions (~0.17%)
    contamination = 0.0017
    
    print(f"Training AutoEncoder model (contamination={contamination})... this may take a moment.")
    clf = AutoEncoder(hidden_neuron_list=[29, 16, 8, 16, 29], 
                      epoch_num=10, 
                      batch_size=256, 
                      contamination=contamination,
                      verbose=1) 
    clf.fit(X_train)

    print("\nModel trained. Generating predictions...")
    # 4. Predict
    # Binary labels (0: normal, 1: outlier)
    y_test_pred = clf.predict(X_test)
    
    # Raw outlier scores (Reconstruction Error)
    y_test_scores = clf.decision_function(X_test)

    # 5. Evaluation
    print("\n--- Evaluation Metrics ---")
    roc_score = roc_auc_score(y_test, y_test_scores)
    print(f"ROC AUC Score: {roc_score:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred))

    # 6. Generate Plot for the experiment output
    print("Generating output plot...")
    plt.figure(figsize=(10, 6))
    plt.hist(y_test_scores, bins=50, alpha=0.75, color='blue', edgecolor='black')
    plt.axvline(x=clf.threshold_, color='red', linestyle='--', label=f'Threshold: {clf.threshold_:.2f}')
    plt.title('Distribution of Anomaly Scores (Reconstruction Error)')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Frequency')
    plt.legend()
    
    # Saving plot to disk
    plt.savefig('experiment_output.png')
    print("Screenshot/plot saved as 'experiment_output.png'. You can use this image in your Word document.")

if __name__ == "__main__":
    main()