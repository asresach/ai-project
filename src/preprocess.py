import os
import numpy as np
import pandas as pd
import librosa
from sklearn.preprocessing import LabelEncoder
import joblib

def load_and_extract_features(data_path):
    """
    ሁሉንም የድምጽ ፋይሎች ይዞ ባህሪያትን ያውጣል
    """
    features = []
    labels = []
    
    # በእያንዳንዱ አቃፊ ውስጥ ያሉትን ፋይሎች ለማየት
    for folder_name in os.listdir(data_path):
        folder_path = os.path.join(data_path, folder_name)
        
        if os.path.isdir(folder_path):
            # ከአቃፊ ስም ስሜቱን ያውጡ
            # ለምሳሌ: "OAF_angry" -> "angry"
            emotion = folder_name.split('_')[-1].lower()
            # "pleasantsurprise" ወይም "pleasant_surprise" ከሆነ "surprise" ብለን እንይዘው
            if 'surprise' in emotion or 'surprised' in emotion:
                emotion = 'surprise'
            
            print(f"Processing {folder_name} -> {emotion}")
            
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.wav'):
                    file_path = os.path.join(folder_path, file_name)
                    
                    try:
                        # ድምጹን ጫን (3 ሰከንድ ብቻ)
                        audio, sr = librosa.load(file_path, sr=22050, duration=3)
                        
                        # MFCC ባህሪያት ማውጣት
                        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
                        mfcc_mean = np.mean(mfcc.T, axis=0)
                        
                        features.append(mfcc_mean)
                        labels.append(emotion)
                        
                    except Exception as e:
                        print(f"Error processing {file_name}: {e}")
    
    return np.array(features), np.array(labels)

def prepare_data():
    """
    ውሂቡን ለሞዴል ዝግጁ ያደርጋል
    """
    # ውሂብ ያለበት መንገድ
    data_path = "C:/Users/user/Desktop/emotion project/data/TESS Toronto emotional speech set data"
    
    print("📂 Loading audio files...")
    X, y = load_and_extract_features(data_path)
    
    print(f"✅ Loaded {len(X)} samples")
    print(f"📊 Feature shape: {X.shape}")
    
    # ስሜቶችን ወደ ቁጥር ይቀይሩ
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"🎭 Emotions found: {list(label_encoder.classes_)}")
    print(f"📊 Class distribution:")
    for i, emotion in enumerate(label_encoder.classes_):
        count = sum(y_encoded == i)
        print(f"   {emotion}: {count} samples")
    
    # አቃፊዎችን ይፍጠሩ
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # ውሂቡን ያስቀምጡ
    np.save('data/X_features.npy', X)
    np.save('data/y_labels.npy', y_encoded)
    joblib.dump(label_encoder, 'models/label_encoder.pkl')
    
    print("✅ Data prepared and saved!")
    print(f"   - Features saved to: data/X_features.npy")
    print(f"   - Labels saved to: data/y_labels.npy")
    print(f"   - Label encoder saved to: models/label_encoder.pkl")
    
    return X, y_encoded, label_encoder

if __name__ == "__main__":
    X, y, le = prepare_data()