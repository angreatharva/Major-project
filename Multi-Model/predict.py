import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import librosa
import librosa.display
from matplotlib.figure import Figure

from src.config import (
    MODELS_DIR, RESULTS_DIR, EMOTION_LABELS, NUM_CLASSES,
    FER_MODEL_PATH, VER_MODEL_PATH, FUSION_MODEL_PATH,
    IMAGE_SIZE, AUDIO_SAMPLE_RATE, AUDIO_DURATION, AUDIO_N_MELS
)
from src.data_processors import extract_audio_features, check_directory_structure

def load_models():
    """Load the trained models for inference"""
    models = {}
    
    # Check if model files exist
    if os.path.exists(FER_MODEL_PATH):
        models['fer'] = tf.keras.models.load_model(FER_MODEL_PATH)
        models['fer_embedding'] = tf.keras.Model(
            inputs=models['fer'].input,
            outputs=models['fer'].get_layer('fer_embedding').output
        )
    else:
        print(f"FER model not found at {FER_MODEL_PATH}")
    
    if os.path.exists(VER_MODEL_PATH):
        models['ver'] = tf.keras.models.load_model(VER_MODEL_PATH)
        models['ver_embedding'] = tf.keras.Model(
            inputs=models['ver'].input,
            outputs=models['ver'].get_layer('ver_embedding').output
        )
    else:
        print(f"VER model not found at {VER_MODEL_PATH}")
    
    if os.path.exists(FUSION_MODEL_PATH):
        models['fusion'] = tf.keras.models.load_model(FUSION_MODEL_PATH)
    else:
        print(f"Fusion model not found at {FUSION_MODEL_PATH}")
    
    return models

def predict_emotion_from_image(model, image_path):
    """Predict emotion from a facial image"""
    # Load and preprocess image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error loading image: {image_path}")
        return None
    
    # Resize image
    img = cv2.resize(img, IMAGE_SIZE)
    
    # Preprocess image
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    
    # Make prediction
    pred = model.predict(img)[0]
    emotion_idx = np.argmax(pred)
    emotion = EMOTION_LABELS[emotion_idx]
    confidence = pred[emotion_idx]
    
    return {
        'emotion': emotion,
        'confidence': float(confidence),
        'probabilities': {EMOTION_LABELS[i]: float(p) for i, p in enumerate(pred)}
    }

def predict_emotion_from_audio(model, audio_path):
    """Predict emotion from an audio file"""
    # Extract audio features
    features = extract_audio_features(audio_path)
    if features is None:
        print(f"Error extracting features from audio: {audio_path}")
        return None
    
    # Add batch dimension
    features = np.expand_dims(features, axis=0)
    
    # Make prediction
    pred = model.predict(features)[0]
    emotion_idx = np.argmax(pred)
    emotion = EMOTION_LABELS[emotion_idx]
    confidence = pred[emotion_idx]
    
    return {
        'emotion': emotion,
        'confidence': float(confidence),
        'probabilities': {EMOTION_LABELS[i]: float(p) for i, p in enumerate(pred)}
    }

def predict_emotion_multimodal(models, image_path, audio_path):
    """Predict emotion using both facial and vocal information"""
    # Check if all required models are loaded
    required_models = ['fer', 'ver', 'fusion', 'fer_embedding', 'ver_embedding']
    if not all(model in models for model in required_models):
        print("Not all required models are loaded")
        return None
    
    # Get face embedding
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error loading image: {image_path}")
        return None
    
    img = cv2.resize(img, IMAGE_SIZE)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    
    fer_embedding = models['fer_embedding'].predict(img)
    
    # Get audio embedding
    features = extract_audio_features(audio_path)
    if features is None:
        print(f"Error extracting features from audio: {audio_path}")
        return None
    
    features = np.expand_dims(features, axis=0)
    
    ver_embedding = models['ver_embedding'].predict(features)
    
    # Fusion prediction
    fusion_pred = models['fusion'].predict([fer_embedding, ver_embedding])[0]
    emotion_idx = np.argmax(fusion_pred)
    emotion = EMOTION_LABELS[emotion_idx]
    confidence = fusion_pred[emotion_idx]
    
    # Get individual model predictions for comparison
    fer_pred = models['fer'].predict(img)[0]
    ver_pred = models['ver'].predict(features)[0]
    
    fer_emotion = EMOTION_LABELS[np.argmax(fer_pred)]
    ver_emotion = EMOTION_LABELS[np.argmax(ver_pred)]
    
    return {
        'fusion': {
            'emotion': emotion,
            'confidence': float(confidence),
            'probabilities': {EMOTION_LABELS[i]: float(p) for i, p in enumerate(fusion_pred)}
        },
        'fer': {
            'emotion': fer_emotion,
            'confidence': float(fer_pred[np.argmax(fer_pred)]),
            'probabilities': {EMOTION_LABELS[i]: float(p) for i, p in enumerate(fer_pred)}
        },
        'ver': {
            'emotion': ver_emotion,
            'confidence': float(ver_pred[np.argmax(ver_pred)]),
            'probabilities': {EMOTION_LABELS[i]: float(p) for i, p in enumerate(ver_pred)}
        }
    }

def visualize_prediction(result, image_path, audio_path, save_path=None):
    """Visualize the prediction results"""
    # Load image for display
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create figure
    fig = plt.figure(figsize=(12, 8))
    
    # Plot image
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(img)
    ax1.set_title("Input Image")
    ax1.axis('off')
    
    # Plot audio waveform
    ax2 = fig.add_subplot(2, 2, 2)
    y, sr = librosa.load(audio_path, sr=AUDIO_SAMPLE_RATE)
    librosa.display.waveshow(y, sr=sr, ax=ax2)
    ax2.set_title("Input Audio Waveform")
    
    # Plot emotion probabilities for each model
    ax3 = fig.add_subplot(2, 2, 3)
    labels = EMOTION_LABELS
    fer_probs = [result['fer']['probabilities'][emotion] for emotion in labels]
    ver_probs = [result['ver']['probabilities'][emotion] for emotion in labels]
    fusion_probs = [result['fusion']['probabilities'][emotion] for emotion in labels]
    
    x = np.arange(len(labels))
    width = 0.25
    
    ax3.bar(x - width, fer_probs, width, label='FER')
    ax3.bar(x, ver_probs, width, label='VER')
    ax3.bar(x + width, fusion_probs, width, label='Fusion')
    
    ax3.set_ylabel('Probability')
    ax3.set_title('Emotion Probabilities by Model')
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, rotation=45)
    ax3.legend()
    
    # Display final prediction
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    
    prediction_text = (
        f"FER Prediction: {result['fer']['emotion']} ({result['fer']['confidence']:.2f})\n"
        f"VER Prediction: {result['ver']['emotion']} ({result['ver']['confidence']:.2f})\n"
        f"Fusion Prediction: {result['fusion']['emotion']} ({result['fusion']['confidence']:.2f})"
    )
    
    ax4.text(0.5, 0.5, prediction_text, ha='center', va='center', fontsize=12)
    ax4.set_title('Prediction Results')
    
    plt.tight_layout()
    
    # Save or show figure
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":
    # Create results directory if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Check directory structure
    check_directory_structure()
    
    # Load models
    models = load_models()
    
    # Example file paths (adjust these to your actual file paths)
    # Try to find a valid test image and audio file
    test_image = None
    for root, dirs, files in os.walk('data/FER-Dataset'):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                test_image = os.path.join(root, file)
                break
        if test_image:
            break
    
    test_audio = None
    for root, dirs, files in os.walk('data/VER-Dataset'):
        for file in files:
            if file.endswith('.wav'):
                test_audio = os.path.join(root, file)
                break
        if test_audio:
            break
    
    if not test_image:
        print("No test image found. Please specify an image path.")
        test_image = input("Enter path to test image: ")
    
    if not test_audio:
        print("No test audio found. Please specify an audio path.")
        test_audio = input("Enter path to test audio: ")
    
    print(f"Using test image: {test_image}")
    print(f"Using test audio: {test_audio}")
    
    # Individual predictions
    if 'fer' in models:
        fer_result = predict_emotion_from_image(models['fer'], test_image)
        print(f"FER Prediction: {fer_result['emotion']} (Confidence: {fer_result['confidence']:.2f})")
    
    if 'ver' in models:
        ver_result = predict_emotion_from_audio(models['ver'], test_audio)
        print(f"VER Prediction: {ver_result['emotion']} (Confidence: {ver_result['confidence']:.2f})")
    
    # Multi-modal prediction
    if all(model in models for model in ['fer', 'ver', 'fusion', 'fer_embedding', 'ver_embedding']):
        fusion_result = predict_emotion_multimodal(models, test_image, test_audio)
        print(f"Fusion Prediction: {fusion_result['fusion']['emotion']} (Confidence: {fusion_result['fusion']['confidence']:.2f})")
        
        # Visualize results
        save_path = os.path.join(RESULTS_DIR, "prediction_visualization.png")
        visualize_prediction(fusion_result, test_image, test_audio, save_path)
        print(f"Prediction visualization saved to: {save_path}")
