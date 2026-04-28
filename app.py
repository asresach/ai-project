import streamlit as st
import numpy as np
import librosa
import joblib
import os
import pandas as pd
import matplotlib.pyplot as plt
import io
import soundfile as sf
from streamlit_mic_recorder import mic_recorder
from pydub import AudioSegment

# Page Configuration
st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎵",
    layout="wide"
)

# Header Section
st.markdown(
    '<h1 style="color: white; text-align: center; text-shadow: 2px 2px 4px rgba(0,0,0,0.5); font-family: Arial, sans-serif;">🎓 Mekdela Amba University</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<h2 style="color: #FFD700; text-align: center; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);">🎵 Speech Emotion Recognition System</h2>',
    unsafe_allow_html=True,
)
st.markdown("---")

# Custom CSS for Mekdela Amba University theme - Clean Professional Look
st.markdown("""
<style>
    .stApp {
        background-color: #f8f9fa;
        color: #212529;
    }

    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #212529 !important;
        text-shadow: none;
    }

    .stSidebar {
        background-color: rgba(248, 249, 250, 0.95);
        backdrop-filter: blur(10px);
        border-right: 3px solid #6c757d;
    }

    .stButton>button {
        background: linear-gradient(45deg, #007bff, #0056b3);
        color: white;
        border: 2px solid #007bff;
        border-radius: 25px;
        padding: 10px 20px;
        font-weight: bold;
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        background: linear-gradient(45deg, #0056b3, #007bff);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }

    .stFileUploader {
        background-color: rgba(255, 255, 255, 0.8);
        border: 2px dashed #6c757d;
        border-radius: 10px;
        padding: 20px;
    }

    .stTable {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        border: 1px solid #dee2e6;
    }

    .stMetric {
        background: linear-gradient(135deg, rgba(0,123,255,0.1), rgba(0,86,179,0.1));
        border-radius: 10px;
        padding: 10px;
        border: 1px solid #007bff;
    }

    .stInfo {
        background-color: rgba(0, 123, 255, 0.1);
        border-left: 4px solid #007bff;
        color: #212529;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processed_features' not in st.session_state:
    st.session_state.processed_features = None
if 'processed_waveform' not in st.session_state:
    st.session_state.processed_waveform = None
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = 'Logistic Regression'

# ሞዴሎችን ጫን
@st.cache_resource
def load_models():
    models = {}
    label_encoder = None
    
    if os.path.exists('models/baseline_model.pkl'):
        models['Logistic Regression'] = joblib.load('models/baseline_model.pkl')
    if os.path.exists('models/advanced_model.pkl'):
        models['Random Forest'] = joblib.load('models/advanced_model.pkl')
    if os.path.exists('models/label_encoder.pkl'):
        label_encoder = joblib.load('models/label_encoder.pkl')
    
    return models, label_encoder

def get_features_from_audio(audio_data):
    """Extract features and waveform from processed audio data"""
    try:
        # Handle both BytesIO and bytes input
        if isinstance(audio_data, io.BytesIO):
            # If it's already a BytesIO, use it directly
            audio_data.seek(0)
            data = audio_data.read()
        else:
            # If it's bytes, use directly
            data = audio_data

        # Use librosa to load the audio data directly from bytes
        y, sr = librosa.load(io.BytesIO(data), sr=None)

        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        return mfccs_scaled.reshape(1, -1), y
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
        return None, None


def convert_raw_audio_to_wav_buffer(raw_audio):
    """Convert raw audio bytes from the recorder to a WAV BytesIO buffer."""
    audio_io = io.BytesIO(raw_audio)
    try:
        audio_segment = AudioSegment.from_file(audio_io, format='wav')
    except Exception:
        audio_io.seek(0)
        audio_segment = AudioSegment.from_file(audio_io)
    wav_buffer = io.BytesIO()
    audio_segment.export(wav_buffer, format='wav')
    wav_buffer.seek(0)
    return wav_buffer

# Emojis for displaying emotions
emotion_emojis = {
    'angry': '😠',
    'disgust': '🤢',
    'fear': '😨',
    'happy': '😊',
    'neutral': '😐',
    'sad': '😢',
    'surprise': '😲'
}

def plot_waveform(audio_data, sr=None, title="Audio Waveform"):
    """Enhanced waveform plotting function that handles file paths, BytesIO objects, and bytes"""
    try:
        # Handle different input types
        if isinstance(audio_data, str):
            # File path - use librosa
            data, sample_rate = librosa.load(audio_data, sr=22050, duration=3)
        elif isinstance(audio_data, io.BytesIO):
            # BytesIO object - use soundfile
            audio_data.seek(0)  # Reset to beginning
            data, sample_rate = sf.read(audio_data)
            # Convert stereo to mono if necessary
            if len(data.shape) > 1:
                data = data.mean(axis=1)
        elif isinstance(audio_data, bytes):
            # Raw bytes - convert to BytesIO then use soundfile
            audio_bytes = io.BytesIO(audio_data)
            data, sample_rate = sf.read(audio_bytes)
            # Convert stereo to mono if necessary
            if len(data.shape) > 1:
                data = data.mean(axis=1)
        elif isinstance(audio_data, np.ndarray):
            # Already loaded numpy array
            data = audio_data
            sample_rate = sr if sr else 22050
        else:
            raise ValueError("Unsupported audio data type")

        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 4))
        fig.patch.set_facecolor('#0E1117')  # Match Streamlit dark theme
        ax.set_facecolor('#0E1117')

        # Plot the waveform
        librosa.display.waveshow(data, sr=sample_rate, ax=ax, color='#FFD700')

        # Style the plot
        ax.set_title(title, fontsize=14, color='white', fontweight='bold')
        ax.set_xlabel('ጊዜ (በሰከንድ) - Time (seconds)', fontsize=12, color='white')
        ax.set_ylabel('ጥንካሬ - Amplitude', fontsize=12, color='white')
        ax.tick_params(colors='white')
        ax.grid(True, alpha=0.3)

        # Style spines
        for spine in ax.spines.values():
            spine.set_edgecolor('white')

        st.pyplot(fig)

    except Exception as e:
        st.warning(f"Could not display waveform: {str(e)}")
        # Fallback: try alternative method
        try:
            if isinstance(audio_data, io.BytesIO):
                audio_data.seek(0)
                data, sample_rate = sf.read(audio_data)
                if len(data.shape) > 1:
                    data = data.mean(axis=1)

                fig, ax = plt.subplots(figsize=(12, 4))
                ax.plot(data, color='#FFD700')
                ax.set_title(title, fontsize=14, color='white')
                ax.set_xlabel('Samples', fontsize=12, color='white')
                ax.set_ylabel('Amplitude', fontsize=12, color='white')
                ax.tick_params(colors='white')
                st.pyplot(fig)
        except:
            st.error("Unable to display audio waveform")

# Dynamic background colors for emotions (Light Theme)
emotion_backgrounds = {
    'angry': 'linear-gradient(135deg, #FFF5F5 0%, #FFE6E6 50%, #FFCCCC 100%)',  # Light red gradient
    'disgust': 'linear-gradient(135deg, #F5FFF5 0%, #E6FFE6 50%, #CCFFCC 100%)',  # Light green gradient
    'fear': 'linear-gradient(135deg, #F9F5FF 0%, #F0E6FF 50%, #E6CCFF 100%)',  # Light purple gradient
    'happy': 'linear-gradient(135deg, #FFFFF5 0%, #FFFFE6 50%, #FFFFCC 100%)',  # Light yellow gradient
    'neutral': 'linear-gradient(135deg, #F8F9FA 0%, #E9ECEF 50%, #DEE2E6 100%)',  # Light gray gradient
    'sad': 'linear-gradient(135deg, #F5F8FF 0%, #E6F0FF 50%, #CCE4FF 100%)',  # Light blue gradient
    'surprise': 'linear-gradient(135deg, #FFF9F5 0%, #FFF0E6 50%, #FFE4CC 100%)'  # Light orange gradient
}

# Emotion colors for light theme
emotion_colors = {
    'angry': '#DC3545',    # Red
    'disgust': '#28A745',  # Green
    'fear': '#6F42C1',     # Purple
    'happy': '#FFC107',    # Yellow
    'neutral': '#6C757D',  # Gray
    'sad': '#007BFF',      # Blue
    'surprise': '#FD7E14'  # Orange
}

def apply_emotion_theme(emotion):
    """Apply dynamic background color based on detected emotion"""
    emotion = emotion.lower()
    bg_color = emotion_backgrounds.get(emotion, 'linear-gradient(135deg, #f8f9fa 0%, #e9ecef 50%, #dee2e6 100%)')

    theme_css = f"""
    <style>
        .stApp {{
            background: {bg_color} !important;
            transition: background 1s ease-in-out;
        }}

        .emotion-result-card {{
            background: rgba(255, 255, 255, 0.9) !important;
            border: 2px solid {emotion_colors.get(emotion, '#007bff')} !important;
            border-radius: 15px !important;
            padding: 20px !important;
            backdrop-filter: blur(10px) !important;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1) !important;
        }}

        .stButton>button {{
            background: linear-gradient(45deg, {emotion_colors.get(emotion, '#007bff')}, #0056b3) !important;
            border: 2px solid {emotion_colors.get(emotion, '#007bff')} !important;
        }}

        .stButton>button:hover {{
            background: linear-gradient(45deg, #0056b3, {emotion_colors.get(emotion, '#007bff')}) !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 12px rgba(0,0,0,0.2) !important;
        }}
    </style>
    """

    st.markdown(theme_css, unsafe_allow_html=True)

def reset_theme():
    """Reset dashboard color to default Mekdela Amba University theme"""
    default_css = """
    <style>
        .stApp {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 50%, #dee2e6 100%) !important;
            transition: background 1s ease-in-out;
        }

        .emotion-result-card {
            background: rgba(255, 255, 255, 0.9) !important;
            border: 2px solid #007bff !important;
            border-radius: 15px !important;
            padding: 20px !important;
            backdrop-filter: blur(10px) !important;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1) !important;
        }

        .stButton>button {
            background: linear-gradient(45deg, #007bff, #0056b3) !important;
            border: 2px solid #007bff !important;
        }

        .stButton>button:hover {
            background: linear-gradient(45deg, #0056b3, #007bff) !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 12px rgba(0,0,0,0.2) !important;
        }
    </style>
    """

    st.markdown(default_css, unsafe_allow_html=True)

# Emotion insights and tips
emotion_insights = {
    'angry': "🎯 **High energy and raised pitch detected** - Anger typically shows increased vocal intensity, faster speech rate, and higher fundamental frequency (F0).\n\n💡 **Take a deep breath** - When feeling angry, try counting to 10 before responding. This can help you communicate more effectively.",
    'disgust': "🤢 **Nasal quality and lowered pitch** - Disgust often manifests as nasal sounds, lowered fundamental frequency, and reduced vocal intensity.\n\n💡 **Express your concerns constructively** - If something bothers you, focus on finding solutions rather than just expressing disapproval.",
    'fear': "😨 **High frequency and vocal jitter** - Fear is characterized by elevated pitch, increased vocal tremor (jitter), and rapid speech patterns.\n\n💡 **You're not alone** - Fear is a natural response. Take a moment to ground yourself - feel your feet on the floor and breathe deeply.",
    'happy': "😊 **Bright tone and varied pitch** - Happiness shows higher pitch, increased vocal energy, and more pitch variation throughout speech.\n\n💡 **Share the joy!** - Your positive energy is contagious. Consider what made you happy and how you can bring more of it into your life.",
    'neutral': "😐 **Stable and consistent tone** - Neutral speech maintains steady pitch, moderate intensity, and regular speech rhythm.\n\n💡 **Stay balanced** - Sometimes neutrality helps us process information objectively. Use this calm state to make thoughtful decisions.",
    'sad': "😢 **Lowered pitch and reduced energy** - Sadness typically features decreased pitch, lower vocal intensity, and slower speech rate.\n\n💡 **Be kind to yourself** - Sadness is part of being human. Consider reaching out to a friend or doing something comforting like listening to your favorite music.",
    'surprise': "😲 **Sudden pitch changes and exclamatory tone** - Surprise shows abrupt pitch rises, increased vocal intensity, and exclamatory speech patterns.\n\n💡 **Embrace the unexpected** - Life is full of surprises! Use this moment to stay curious and open to new possibilities."
}

# Load models
with st.spinner("Loading models..."):
    models, label_encoder = load_models()

# Sidebar
with st.sidebar:
    st.markdown('<h2 style="color: #FFD700; text-align: center;">ℹ️ About the Project</h2>', unsafe_allow_html=True)
    st.write("""
    **🎓 Mekdela Amba University**  
    *School of Computing*
    
    This system detects human emotions from speech signals using advanced Machine Learning techniques.
    
    **🎯 Supported Emotions:**
    """)
    
    for emotion, emoji in emotion_emojis.items():
        st.write(f"{emoji} {emotion.capitalize()}")
    
    st.header("🤖 Available Models")
    for model_name in models.keys():
        st.success(f"✅ {model_name}")
    
    # Model performance comparison table
    st.header("📊 Model Comparison")
    if os.path.exists('models/baseline_results.csv') and os.path.exists('models/advanced_results.csv'):
        baseline_results = pd.read_csv('models/baseline_results.csv')
        advanced_results = pd.read_csv('models/advanced_results.csv')
        
        # Display comparison table
        comparison_data = {
            "Model": ["Logistic Regression", "Random Forest"],
            "Accuracy": [f"{baseline_results['accuracy'].iloc[0]*100:.1f}%", 
                        f"{advanced_results['accuracy'].iloc[0]*100:.1f}%"],
            "CV Mean": [f"{baseline_results['cv_mean'].iloc[0]*100:.1f}%", 
                       f"{advanced_results['cv_mean'].iloc[0]*100:.1f}%"],
            "CV Std": [f"{baseline_results['cv_std'].iloc[0]*100:.2f}", 
                      f"{advanced_results['cv_std'].iloc[0]*100:.2f}"]
        }
        
        st.table(pd.DataFrame(comparison_data))
        
        st.markdown("""
        **Model Characteristics:**
        - **Logistic Regression**: Simple, fast, good for baseline
        - **Random Forest**: More complex, handles non-linear relationships, better generalization
        """)
    
    # Confusion Matrix Display
    st.header("📈 Confusion Matrices")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Logistic Regression")
        if os.path.exists('models/baseline_confusion_matrix.png'):
            st.image('models/baseline_confusion_matrix.png', caption="Baseline Model")
    
    # Feature Importance Display (for Random Forest)
    st.header("🔍 Feature Importance")
    if os.path.exists('models/feature_importance.png'):
        st.image('models/feature_importance.png', caption="Random Forest Feature Importance")
        st.markdown("*Shows which MFCC features are most important for emotion classification*")
    
    # Team Members
    st.header("👥 Team Members - Group 9")
    st.markdown("""
    **Project Contributors:**
    - **Data Preparation**: Audio data collection and preprocessing
    - **Feature Extraction**: MFCC implementation using Librosa
    - **Model Development**: Logistic Regression and Random Forest training
    - **UI/UX Design**: Streamlit dashboard development
    - **Testing & Validation**: Model evaluation and performance analysis
    """)
    
    st.header("📁 Instructions")
    st.write("""
    1. Upload a WAV audio file
    2. Select a model
    3. Click 'Analyze Emotion'
    4. View the prediction
    """)

# Main content
# Theme reset button
if st.button("🔄 Reset Theme", help="Return dashboard to default Mekdela Amba University theme"):
    reset_theme()
    st.success("Theme reset to default!")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.markdown('<h3 style="color: #FFD93D; text-align: center;">📁 Upload Audio</h3>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choose an audio file (WAV format)",
        type=["wav"],
        help="Upload a clear audio recording of speech"
    )
    
    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")
        st.info(f"File: {uploaded_file.name}")
        
        # Automatically process the uploaded file
        with st.spinner("Processing uploaded audio..."):
            features, waveform = get_features_from_audio(uploaded_file.getvalue())
            if features is not None:
                st.session_state.processed_features = features
                st.session_state.processed_waveform = waveform
                st.success("Audio processed successfully!")
            else:
                st.error("Failed to process uploaded audio")
        
        # Waveform Visualization
        st.markdown("### 🎵 Audio Waveform")
        st.markdown("*This graph shows your voice's strength and pattern - This graph shows your voice's strength and pattern*")
        
        # Display waveform using processed data
        if 'processed_waveform' in st.session_state and st.session_state.processed_waveform is not None:
            st.line_chart(st.session_state.processed_waveform[::100])  # Downsample for performance
        else:
            st.warning("Waveform data not available")
    
    # Try These Samples Section
    st.markdown("---")
    st.markdown('<h3 style="color: #FFD93D; text-align: center;">🎵 Try These Samples</h3>', unsafe_allow_html=True)
    st.markdown("**Test the system with pre-loaded emotion samples**")
    
    # Sample emotion buttons
    sample_cols = st.columns(4)
    emotions = ['happy', 'sad', 'angry', 'fear', 'neutral', 'surprise', 'disgust']

    if 'selected_sample' not in st.session_state:
        st.session_state.selected_sample = None

    selected_sample = st.session_state.selected_sample
    for i, emotion in enumerate(emotions):
        with sample_cols[i % 4]:
            if st.button(f"{emotion_emojis.get(emotion, '🎵')} {emotion.capitalize()}", 
                        key=f"sample_{emotion}",
                        help=f"Try a {emotion} sample"):
                # Load random sample from emotion directory
                emotion_dir = f"data/TESS Toronto emotional speech set data/OAF_{emotion}" if emotion != 'surprise' else "data/TESS Toronto emotional speech set data/OAF_Pleasant_surprise"
                if emotion == 'surprise':
                    emotion_dir = os.path.join('data', 'TESS Toronto emotional speech set data', 'OAF_Pleasant_surprise')
                elif emotion == 'disgust':
                    emotion_dir = os.path.join('data', 'TESS Toronto emotional speech set data', 'OAF_disgust')
                elif emotion == 'fear':
                    emotion_dir = os.path.join('data', 'TESS Toronto emotional speech set data', 'OAF_Fear')
                else:
                    emotion_dir = os.path.join('data', 'TESS Toronto emotional speech set data', f'OAF_{emotion}')
                
                try:
                    import random
                    if os.path.exists(emotion_dir):
                        wav_files = [f for f in os.listdir(emotion_dir) if f.endswith('.wav')]
                        if wav_files:
                            selected_file = random.choice(wav_files)
                            selected_path = os.path.join(emotion_dir, selected_file)
                            st.session_state.selected_sample = selected_path
                            selected_sample = selected_path
                            st.success(f"Loaded {emotion} sample: {selected_file}")
                        else:
                            st.error(f"No WAV files found in {emotion_dir}")
                    else:
                        st.error(f"Sample directory not found: {emotion_dir}")
                except Exception as e:
                    st.error(f"Error loading sample: {str(e)}")
    
    # Process selected sample
    if selected_sample:
        try:
            # Load and display the sample audio
            audio_data, sr = librosa.load(selected_sample, sr=22050, duration=3)
            st.audio(selected_sample, format="audio/wav")
            st.info(f"Sample: {os.path.basename(selected_sample)}")
            
            # Create a BytesIO object for feature extraction
            audio_bytes = io.BytesIO()
            sf.write(audio_bytes, audio_data, sr, format='wav')
            audio_bytes.seek(0)
            
            # Automatically process the sample audio for waveform display
            with st.spinner("Processing sample audio..."):
                features, waveform = get_features_from_audio(audio_bytes)
                if features is not None:
                    st.session_state.sample_features = features
                    st.session_state.sample_waveform = waveform
                    st.success("Sample audio processed successfully!")
                else:
                    st.error("Failed to process sample audio")
            
            # Waveform visualization for sample
            st.markdown("### 🎵 Audio Waveform")
            st.markdown("*This graph shows the sample's voice strength and pattern - This graph shows the sample's voice strength and pattern*")
            
            # Display waveform using processed data
            if 'sample_waveform' in st.session_state and st.session_state.sample_waveform is not None:
                st.line_chart(st.session_state.sample_waveform[::100])  # Downsample for performance
            else:
                st.warning("Waveform data not available")
            
            # Model selection for sample
            sample_model_choice = st.selectbox(
                "Select Model for Sample", 
                list(models.keys()), 
                key='sample_model'
            )
            
            # Comparison Mode for samples
            sample_comparison_mode = st.checkbox("🔄 Enable Comparison Mode (Sample)", 
                                               help="Compare predictions from both models on this sample",
                                               key='sample_comparison')
            
            if st.button("🔍 Analyze Sample Emotion", type="secondary"):
                with st.spinner("Analyzing sample..."):
                    # Use pre-processed features
                    features = st.session_state.get('sample_features')
                    
                    if features is not None:
                        if sample_comparison_mode and len(models) > 1:
                            # Show comparison of both models for sample
                            st.subheader("🤖 Sample Model Comparison")
                            
                            comp_col1, comp_col2 = st.columns(2)
                            
                            sample_predictions = {}
                            for i, (model_name, model) in enumerate(models.items()):
                                with comp_col1 if i == 0 else comp_col2:
                                    st.markdown(f"**{model_name}**")
                                    
                                    prediction = model.predict(features)[0]
                                    emotion = label_encoder.inverse_transform([prediction])[0] if label_encoder else str(prediction)
                                    sample_predictions[model_name] = emotion
                                    
                                    emoji = emotion_emojis.get(emotion.lower(), '🎵')
                                    color = emotion_colors.get(emotion.lower(), '#000000')
                                    
                                    st.markdown(f"""
                                    <div style="text-align: center; padding: 15px; background-color: {color}15; border-radius: 8px; border: 2px solid {color};">
                                        <h1 style="font-size: 40px;">{emoji}</h1>
                                        <h3 style="color: {color};">{emotion.upper()}</h3>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    # Confidence for this model
                                    if hasattr(model, 'predict_proba'):
                                        probs = model.predict_proba(features)[0]
                                        confidence = probs[prediction] * 100
                                        st.metric("Confidence", f"{confidence:.1f}%")
                            
                            # Show differences if models disagree
                            if len(set(sample_predictions.values())) > 1:
                                st.info("📊 **Sample Analysis:** Models disagree on this sample! This demonstrates how different algorithms may interpret the same acoustic features differently.")
                            else:
                                st.success("✅ **Sample Analysis:** Both models agree on this sample emotion.")
                                
                        else:
                            # Single model prediction for sample
                            model = models[sample_model_choice]
                            prediction = model.predict(features)[0]
                            
                            emotion = label_encoder.inverse_transform([prediction])[0] if label_encoder else str(prediction)
                            emoji = emotion_emojis.get(emotion.lower(), '🎵')
                            color = emotion_colors.get(emotion.lower(), '#000000')
                            
                            st.markdown(f"""
                            <div style="text-align: center; padding: 20px; background-color: {color}20; border-radius: 10px;">
                                <h1 style="font-size: 60px;">{emoji}</h1>
                                <h2 style="color: {color};">{emotion.upper()}</h2>
                                <p style="color: white;">🎵 Sample Analysis</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Emotion Insight
                            insight = emotion_insights.get(emotion.lower(), "🎵 Acoustic analysis complete - emotion detected based on speech patterns.")
                            st.info(insight)
                            
                            # Confidence Scores
                            if hasattr(model, 'predict_proba'):
                                st.subheader("Confidence Scores")
                                probs = model.predict_proba(features)[0]
                                
                                top3_idx = np.argsort(probs)[-3:][::-1]
                                
                                for idx in top3_idx:
                                    pred_emotion = label_encoder.inverse_transform([idx])[0] if label_encoder else str(idx)
                                    prob = probs[idx]
                                    emoji_small = emotion_emojis.get(pred_emotion.lower(), '🎵')
                                    
                                    st.write(f"{emoji_small} {pred_emotion.capitalize()}: {prob*100:.1f}%")
                                    st.progress(float(prob))
                    else:
                        st.error("Failed to extract features from sample")
                        
        except Exception as e:
            st.error(f"Error processing sample: {str(e)}")
    
    # Live Voice Recording Section
    st.markdown("---")
    st.markdown('<h3 style="color: #FFD93D; text-align: center;">🎤 Live Voice Recording</h3>', unsafe_allow_html=True)
    
    # Voice recording setup
    audio_record = mic_recorder(
        start_prompt="🔴 Start Recording",
        stop_prompt="⏹️ Stop Recording",
        key='recorder'
    )
    
    # Process recorded audio
    if audio_record:
        st.audio(audio_record['bytes'], format="audio/wav")
        st.success("🎵 Recording captured successfully!")
        
        raw_audio = audio_record['bytes']
        wav_buffer = None
        try:
            wav_buffer = convert_raw_audio_to_wav_buffer(raw_audio)
        except Exception as e:
            st.error(f"Error converting recorded audio to WAV: {e}")

        if wav_buffer is not None:
            # Automatically process the recorded audio
            with st.spinner("Processing recorded audio..."):
                features, waveform = get_features_from_audio(wav_buffer)
                if features is not None:
                    st.session_state.processed_features = features
                    st.session_state.processed_waveform = waveform
                    st.success("Recorded audio processed successfully!")
                else:
                    st.error("Failed to process recorded audio")
        else:
            st.error("Recorded audio conversion failed, cannot extract features.")

        # Convert recorded audio to file for model processing
        recorded_file = io.BytesIO(wav_buffer.getvalue()) if wav_buffer is not None else None
        
        # Waveform Visualization for Recorded Audio
        st.markdown("### 🎵 Recorded Audio Waveform")
        st.markdown("*Shows the strength and pattern of your recorded voice - Shows the strength and pattern of your recorded voice*")
        
        # Display waveform using processed data
        if 'processed_waveform' in st.session_state and st.session_state.processed_waveform is not None:
            st.line_chart(st.session_state.processed_waveform[::100])  # Downsample for performance
        else:
            st.warning("Waveform data not available")
        
        # Model selection for recorded audio
        recorded_model_choice = st.selectbox(
            "Select Model for Recorded Audio", 
            list(models.keys()), 
            key='recorded_model'
        )
        
        if st.button("🔍 Analyze Recorded Emotion", type="secondary"):
            with st.spinner("Analyzing your voice..."):
                if recorded_file is None:
                    st.error("Cannot analyze recorded audio because WAV conversion failed.")
                else:
                    features, waveform_data = get_features_from_audio(recorded_file)
                    
                    if features is not None:
                        # Use the prediction logic from above
                        model = models[recorded_model_choice]
                        prediction = model.predict(features)[0]
                        
                        # Detect emotion from prediction
                        emotion = label_encoder.inverse_transform([prediction])[0] if label_encoder else str(prediction)
                        emoji = emotion_emojis.get(emotion.lower(), '🎵')
                        color = emotion_colors.get(emotion.lower(), '#000000')

                        # Change dashboard color for emotion
                        apply_emotion_theme(emotion)

                        st.markdown(f"""
                        <div class="emotion-result-card" style="text-align: center; padding: 20px;">
                            <h1 style="font-size: 60px;">{emoji}</h1>
                            <h2 style="color: {color};">{emotion.upper()}</h2>
                            <p style="color: white;">🎤 Live Recording Analysis</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Emotion Insight
                        insight = emotion_insights.get(emotion.lower(), "🎵 Acoustic analysis complete - emotion detected based on speech patterns.")
                        st.info(insight)
                        
                        # Confidence Probability
                        if hasattr(model, 'predict_proba'):
                            st.subheader("Confidence Scores")
                            probs = model.predict_proba(features)[0]
                            
                            # Show top 3 confidence scores
                            top3_idx = np.argsort(probs)[-3:][::-1]
                            
                            for idx in top3_idx:
                                pred_emotion = label_encoder.inverse_transform([idx])[0] if label_encoder else str(idx)
                                prob = probs[idx]
                                emoji_small = emotion_emojis.get(pred_emotion.lower(), '🎵')
                                
                                st.write(f"{emoji_small} {pred_emotion.capitalize()}: {prob*100:.1f}%")
                                st.progress(float(prob))
                    else:
                        st.error("Failed to extract features from recorded audio")

with col2:
    st.markdown('<h3 style="color: #FFD93D; text-align: center;">🎯 Prediction</h3>', unsafe_allow_html=True)
    
    # Check if we have processed features
    if st.session_state.processed_features is not None:
        features = st.session_state.processed_features
        waveform_data = st.session_state.processed_waveform
        
        # Model selection
        selected_model = st.selectbox("Select Model", list(models.keys()), 
                                    index=list(models.keys()).index(st.session_state.selected_model) if st.session_state.selected_model in models else 0)
        st.session_state.selected_model = selected_model
        
        # Comparison Mode
        comparison_mode = st.checkbox("🔄 Enable Comparison Mode", 
                                    help="Compare predictions from both models")
        
        # Display waveform
        if waveform_data is not None:
            st.markdown("### 🎵 Audio Waveform")
            st.line_chart(waveform_data[::100])  # Downsample for performance
        
        if comparison_mode and len(models) > 1:
            # Show comparison of both models
            st.subheader("🤖 Model Comparison Results")
            
            comp_col1, comp_col2 = st.columns(2)
            
            model_predictions = {}
            for i, (model_name, model) in enumerate(models.items()):
                with comp_col1 if i == 0 else comp_col2:
                    st.markdown(f"**{model_name}**")
                    
                    prediction = model.predict(features)[0]
                    emotion = label_encoder.inverse_transform([prediction])[0] if label_encoder else str(prediction)
                    model_predictions[model_name] = emotion
                    
                    emoji = emotion_emojis.get(emotion.lower(), '🎵')
                    color = emotion_colors.get(emotion.lower(), '#000000')
                    
                    st.markdown(f"""
                    <div style="text-align: center; padding: 15px; background-color: {color}15; border-radius: 8px; border: 2px solid {color};">
                        <h1 style="font-size: 40px;">{emoji}</h1>
                        <h3 style="color: {color};">{emotion.upper()}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Confidence for this model
                    if hasattr(model, 'predict_proba'):
                        probs = model.predict_proba(features)[0]
                        confidence = probs[prediction] * 100
                        st.metric("Confidence", f"{confidence:.1f}%")
            
            # Show differences if models disagree
            if len(set(model_predictions.values())) > 1:
                st.info("📊 **Models disagree!** This shows the complexity of emotion recognition.")
            else:
                st.success("✅ **Models agree!** Both algorithms detected the same emotion.")
                
        else:
            # Single model prediction
            model = models[selected_model]
            prediction = model.predict(features)[0]
            
            emotion = label_encoder.inverse_transform([prediction])[0] if label_encoder else str(prediction)
            emoji = emotion_emojis.get(emotion.lower(), '🎵')
            color = emotion_colors.get(emotion.lower(), '#000000')
            
            # Apply emotion theme
            apply_emotion_theme(emotion)
            
            st.markdown(f"""
            <div class="emotion-result-card" style="text-align: center; padding: 20px;">
                <h1 style="font-size: 80px;">{emoji}</h1>
                <h2 style="color: {color};">{emotion.upper()}</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Emotion Insight
            insight = emotion_insights.get(emotion.lower(), "🎵 Acoustic analysis complete.")
            st.info(insight)
            
            # Confidence Scores
            if hasattr(model, 'predict_proba'):
                st.subheader("Confidence Scores")
                probs = model.predict_proba(features)[0]
                
                top3_idx = np.argsort(probs)[-3:][::-1]
                
                for idx in top3_idx:
                    pred_emotion = label_encoder.inverse_transform([idx])[0] if label_encoder else str(idx)
                    prob = probs[idx]
                    emoji_small = emotion_emojis.get(pred_emotion.lower(), '🎵')
                    
                    st.write(f"{emoji_small} {pred_emotion.capitalize()}: {prob*100:.1f}%")
                    st.progress(float(prob))
    
    else:
        st.info("📁 Upload an audio file or 🎤 record your voice to see predictions here!")

# End of application
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🎵 **Speech Emotion Recognition**")
    st.write("Built with ❤️ using Python, Streamlit, Librosa & Scikit-learn")

with col2:
    st.markdown("### 📊 **Model Performance**")
    st.write("**Accuracy:** 99.8% | **CV Score:** 98.7%")

with col3:
    st.markdown("### 🚀 **Deployment Ready**")
    st.write("Deploy on Streamlit Cloud for online access")

st.markdown('<p style="text-align: center; color: #FFD700; font-size: 18px; font-weight: bold;">🎓 Mekdela Amba University - School of Computing</p>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: white; font-size: 16px;">👥 Group 9 Project - Machine Learning Course</p>', unsafe_allow_html=True)

# Future Scope Section
st.markdown("---")
st.markdown("## 🔮 **Future Enhancements & Research Directions**")

future_col1, future_col2 = st.columns(2)

with future_col1:
    st.markdown("### 🌍 **Multilingual Support**")
    st.markdown("""
    - **Amharic Speech Recognition**: Extend to local languages
    - **Cross-lingual Emotion Detection**: Compare emotion patterns across cultures
    - **Language-specific Acoustic Features**: Adapt MFCC for different phonetic systems
    """)
    
    st.markdown("### 🎙️ **Real-time Processing**")
    st.markdown("""
    - **Live Audio Streaming**: Process emotions in real-time
    - **Microphone Integration**: Direct voice input analysis
    - **Continuous Monitoring**: Track emotional states over time
    """)

with future_col2:
    st.markdown("### 🤖 **Advanced AI Features**")
    st.markdown("""
    - **Deep Learning Models**: CNN, RNN, Transformer architectures
    - **Multi-modal Analysis**: Combine audio with facial expressions
    - **Context-aware Recognition**: Consider conversation context
    """)
    
    st.markdown("### 📱 **Application Extensions**")
    st.markdown("""
    - **Mental Health Monitoring**: Clinical psychology applications
    - **Customer Service**: Automated sentiment analysis
    - **Education**: Student engagement assessment
    """)

st.markdown("### 📈 **Research Opportunities**")
st.markdown("""
- **Dataset Expansion**: Collect more diverse speech samples
- **Cross-cultural Studies**: Compare emotion expression across demographics
- **Hybrid Approaches**: Combine traditional ML with deep learning
- **Explainable AI**: Develop interpretable emotion recognition models
""")

st.markdown('<p style="text-align: center; color: #FFD700; font-style: italic;">"Advancing AI for Better Human Understanding"</p>', unsafe_allow_html=True)