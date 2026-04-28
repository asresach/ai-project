import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import os

# ውሂብ ጫን
print("📂 Loading prepared data...")
X = np.load('data/X_features.npy')
y = np.load('data/y_labels.npy')
label_encoder = joblib.load('models/label_encoder.pkl')

print(f"✅ Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
print(f"🎭 Classes: {list(label_encoder.classes_)}")

# ውሂብን ይከፋፍሉ
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"📊 Train size: {len(X_train)}, Test size: {len(X_test)}")

# ============================================
# ADVANCED MODEL: Random Forest
# ============================================
print("\n" + "="*60)
print("🎯 TRAINING ADVANCED MODEL: Random Forest")
print("="*60)

# ቀላል ሞዴል በመጀመሪያ
print("\n🔍 Training Random Forest with default parameters...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# ትንበያ
y_pred_rf = rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)

print(f"✅ Test Accuracy: {accuracy_rf:.4f} ({accuracy_rf*100:.2f}%)")

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred_rf, target_names=label_encoder.classes_))

# Cross-validation
cv_scores = cross_val_score(rf_model, X, y, cv=5)
print(f"\n🔄 Cross-validation scores: {cv_scores}")
print(f"📊 Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

# Feature Importance
feature_importance = pd.DataFrame({
    'feature': [f'MFCC_{i}' for i in range(X.shape[1])],
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n📊 Top 10 Important Features:")
print(feature_importance.head(10))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
            xticklabels=label_encoder.classes_, 
            yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix - Random Forest')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
os.makedirs('models', exist_ok=True)
plt.savefig('models/advanced_confusion_matrix.png')
plt.close()
print("\n📊 Confusion matrix saved to 'models/advanced_confusion_matrix.png'")

# Feature Importance Graph
plt.figure(figsize=(10, 6))
plt.barh(feature_importance.head(10)['feature'], feature_importance.head(10)['importance'])
plt.xlabel('Importance')
plt.title('Top 10 Feature Importances - Random Forest')
plt.gca().invert_yaxis()
plt.savefig('models/feature_importance.png')
plt.close()
print("📊 Feature importance graph saved to 'models/feature_importance.png'")

# ሞዴሉን ያስቀምጡ
joblib.dump(rf_model, 'models/advanced_model.pkl')
print("\n💾 Advanced model saved to 'models/advanced_model.pkl'")

# ውጤቶችን አስቀምጡ
results = {
    'model': 'Random Forest',
    'accuracy': accuracy_rf,
    'cv_mean': cv_scores.mean(),
    'cv_std': cv_scores.std()
}
pd.DataFrame([results]).to_csv('models/advanced_results.csv', index=False)
print("✅ Results saved to 'models/advanced_results.csv'")

print("\n🎉 Advanced model training complete!")

# ============================================
# ሁለቱን ሞዴሎች ማነፃፀር
# ============================================
print("\n" + "="*60)
print("📊 MODEL COMPARISON")
print("="*60)

baseline_results = pd.read_csv('models/baseline_results.csv')
print(f"\n🔹 Logistic Regression (Baseline):")
print(f"   Accuracy: {baseline_results['accuracy'].iloc[0]*100:.2f}%")
print(f"   CV Mean: {baseline_results['cv_mean'].iloc[0]*100:.2f}%")

print(f"\n🔹 Random Forest (Advanced):")
print(f"   Accuracy: {accuracy_rf*100:.2f}%")
print(f"   CV Mean: {cv_scores.mean()*100:.2f}%")

print("\n🎉 Both models are ready for deployment!")