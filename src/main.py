"""Python example script extracted for reproducibility.
This script trains a simple Logistic Regression on the Iris dataset
and saves a small plot. It's a runnable example for the converted Colab.
"""
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

def run_example(output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)

    iris = load_iris()
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = LogisticRegression(max_iter=200)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Test accuracy: {acc:.4f}")

    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix (acc={acc:.2f})")
    plot_path = Path(output_dir) / "confusion_matrix.png"
    plt.savefig(plot_path)
    print(f"Saved plot to {plot_path}")

    model_path = Path(output_dir) / "model.joblib"
    joblib.dump(clf, model_path)
    print(f"Saved model to {model_path}")

if __name__ == "__main__":
    run_example()
