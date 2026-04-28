# test_overfitting.py
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report
import joblib

def check_overfitting(model_path, X, y):
    """
    ሞዴሉ overfitting መሆኑን ይፈትሻል
    """
    model = joblib.load(model_path)
    
    # መጀመሪያ: Train-Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"Train Accuracy: {train_score*100:.2f}%")
    print(f"Test Accuracy:  {test_score*100:.2f}%")
    print(f"Difference:     {(train_score - test_score)*100:.2f}%")
    
    if (train_score - test_score) > 0.10:
        print("⚠️ HIGH OVERFITTING DETECTED!")
        print("   ሞዴሉ በስልጠና ውሂብ ላይ ብቻ ቃኝቷል")
        print("   አዲስ ውሂብ ላይ አይሰራም")
    elif (train_score - test_score) > 0.05:
        print("⚠️ MODERATE OVERFITTING")
        print("   ማስተካከያ ያስፈልጋል")
    else:
        print("✅ GOOD! No overfitting detected")
    
    return train_score, test_score

# Load data
X = np.load('data/X_features.npy')
y = np.load('data/y_labels.npy')

# ይህን ሮጡ!
check_overfitting("models/baseline_model.pkl", X, y)