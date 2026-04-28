import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import os

# የተዘጋጀውን ውሂብ ጫን
print("📂 Loading prepared data...")
X = np.load('data/X_features.npy')
y = np.load('data/y_labels.npy')
label_encoder = joblib.load('models/label_encoder.pkl')

print(f"✅ Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
print(f"🎭 Classes: {list(label_encoder.classes_)}")

# ውሂብን ይከፋፍሉ (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"📊 Train size: {len(X_train)}, Test size: {len(X_test)}")

# ============================================
# BASELINE MODEL: Logistic Regression
# ============================================
print("\n" + "="*60)
print("🎯 TRAINING BASELINE MODEL: Logistic Regression")
print("="*60)

# ቀላል እና የተስተካከለ ሞዴል
lr_model = LogisticRegression(max_iter=5000, random_state=42)
lr_model.fit(X_train, y_train)

# ትንበያ
y_pred_lr = lr_model.predict(X_test)

# ውጤቶች
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print(f"\n✅ Accuracy: {accuracy_lr:.4f} ({accuracy_lr*100:.2f}%)")

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred_lr, target_names=label_encoder.classes_))

# Cross-validation
cv_scores = cross_val_score(lr_model, X, y, cv=5)
print(f"\n🔄 Cross-validation scores: {cv_scores}")
print(f"📊 Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_lr)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=label_encoder.classes_, 
            yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix - Logistic Regression')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
os.makedirs('models', exist_ok=True)
plt.savefig('models/baseline_confusion_matrix.png')
plt.close()
print("\n📊 Confusion matrix saved to 'models/baseline_confusion_matrix.png'")

# ሞዴሉን ያስቀምጡ
joblib.dump(lr_model, 'models/baseline_model.pkl')
print("\n💾 Baseline model saved to 'models/baseline_model.pkl'")

# ውጤቶችን ያስቀምጡ
results = {
    'model': 'Logistic Regression',
    'accuracy': accuracy_lr,
    'cv_mean': cv_scores.mean(),
    'cv_std': cv_scores.std()
}
pd.DataFrame([results]).to_csv('models/baseline_results.csv', index=False)
print("✅ Results saved to 'models/baseline_results.csv'")

print("\n🎉 Baseline training complete!")