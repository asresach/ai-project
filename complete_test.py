# complete_test.py
"""
ሙሉ የሞዴል ፍተሻ - ለGroup 9 Speech Emotion Recognition
"""

import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.datasets import make_classification

def run_complete_test():
    """
    ሙሉ ፍተሻን በአንድ ጊዜ ያሂዳል
    """
    print("="*60)
    print("COMPLETE MODEL TEST FOR GROUP 9")
    print("Speech Emotion Recognition System")
    print("="*60)
    
    # ደረጃ 1: ፋይሎችን ፈልግ
    print("\n📁 Step 1: Looking for files...")
    
    # ሞዴሎችን ፈልግ
    model_files = []
    model_dirs = ['models/saved', 'models', '.']
    for d in model_dirs:
        if os.path.exists(d):
            for f in os.listdir(d):
                if f.endswith('.pkl') or f.endswith('.joblib'):
                    model_files.append(os.path.join(d, f))
    
    if not model_files:
        print("❌ No model files found!")
        print("   Please train a model first.")
        return
    
    print(f"✅ Found {len(model_files)} model(s):")
    for mf in model_files:
        print(f"   - {mf}")
    
    # ውሂብን ፈልግ
    data_files = []
    data_dirs = ['data/processed', 'data', '.']
    for d in data_dirs:
        if os.path.exists(d):
            for f in os.listdir(d):
                if f.endswith('.npy') and ('X' in f or 'features' in f):
                    data_files.append(os.path.join(d, f))
    
    if not data_files:
        print("\n⚠️ No feature files found!")
        print("   Creating sample data for testing...")
        
        # የሙከራ ውሂብ ፍጠር
        X, y = make_classification(n_samples=500, n_features=40, 
                                   n_classes=7, n_informative=35,
                                   random_state=42)
        print(f"✅ Created sample data: {X.shape}")
    else:
        print(f"✅ Found data file: {data_files[0]}")
        X = np.load(data_files[0])
        
        # y ፋይል ፈልግ
        y_file = data_files[0].replace('X', 'y').replace('features', 'labels')
        if os.path.exists(y_file):
            y = np.load(y_file)
        else:
            y = np.random.randint(0, 7, X.shape[0])
        
        print(f"   X shape: {X.shape}")
        print(f"   y shape: {y.shape}")
    
    # ደረጃ 2: ለእያንዳንዱ ሞዴል ፈትሽ
    print("\n" + "="*60)
    print("📊 Step 2: Testing Models")
    print("="*60)
    
    results = []
    
    for model_path in model_files:
        print(f"\n🔍 Testing: {os.path.basename(model_path)}")
        print("-"*40)
        
        try:
            model = joblib.load(model_path)
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            train_acc = model.score(X_train, y_train)
            test_acc = model.score(X_test, y_test)
            
            print(f"Train Accuracy: {train_acc*100:.2f}%")
            print(f"Test Accuracy:  {test_acc*100:.2f}%")
            print(f"Difference:     {(train_acc - test_acc)*100:.2f}%")
            
            # Cross-validation
            cv_scores = cross_val_score(model, X, y, cv=5)
            print(f"CV Mean:        {cv_scores.mean()*100:.2f}%")
            print(f"CV Std:         {cv_scores.std()*100:.2f}%")
            
            # ሁኔታ ወስን
            if (train_acc - test_acc) > 0.10:
                status = "⚠️ OVERFITTING"
            elif (train_acc - test_acc) > 0.05:
                status = "⚠️ MODERATE OVERFITTING"
            elif test_acc < 0.70:
                status = "⚠️ UNDERFITTING"
            else:
                status = "✅ GOOD"
            
            print(f"Status: {status}")
            
            results.append({
                'model': os.path.basename(model_path),
                'train_acc': train_acc,
                'test_acc': test_acc,
                'cv_mean': cv_scores.mean(),
                'status': status
            })
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # ደረጃ 3: ማጠቃለያ
    print("\n" + "="*60)
    print("📈 SUMMARY")
    print("="*60)
    
    for r in results:
        print(f"{r['model']:30} {r['status']:20} Test: {r['test_acc']*100:.1f}%")
    
    print("\n" + "="*60)
    print("✅ Test complete!")
    print("="*60)
    
    return results

# ሮጡ!
if __name__ == "__main__":
    run_complete_test()