import os
import json
import sys
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import cv2
from keras.models import load_model
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename

# ==================== GPU CONFIGURATION ====================
print("\n" + "="*70)
print("FLASK APP STARTUP - GPU CONFIGURATION FOR RTX 3050")
print("="*70)

def setup_gpu():
    """Configure TensorFlow for optimal GPU performance"""
    print(f"\nTensorFlow version: {tf.__version__}")
    print(f"Python version: {sys.version.split()[0]}")
    
    # List all devices
    all_devices = tf.config.list_physical_devices()
    print(f"\nDetected devices ({len(all_devices)}):")
    for dev in all_devices:
        print(f"  - {dev}")
    
    # Configure GPU
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        print(f"\n✓ GPU(s) detected: {len(gpus)}")
        try:
            # Enable memory growth to prevent OOM errors
            for i, gpu in enumerate(gpus):
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"  ✓ GPU {i}: Memory growth enabled")
                
                # Get GPU details
                try:
                    details = tf.config.experimental.get_device_details(gpu)
                    if details:
                        print(f"    Device name: {details.get('device_name', 'Unknown')}")
                        print(f"    Compute capability: {details.get('compute_capability', 'Unknown')}")
                except:
                    pass
            
            print("\n✓ GPU READY FOR INFERENCE")
            return True
        except RuntimeError as e:
            print(f"  ✗ Error configuring GPU: {e}")
            print("  Falling back to CPU")
            return False
    else:
        print("\n✗ No GPU devices found!")
        print("  Possible causes:")
        print("  1. NVIDIA drivers not installed")
        print("  2. CUDA toolkit not installed")
        print("  3. cuDNN not installed/configured")
        print("  4. TensorFlow compiled for CPU only")
        print("\n  Falling back to CPU for inference")
        print("  Run 'python gpu_setup_verification.py' to diagnose")
        return False

gpu_available = setup_gpu()

# Print status
print("\n" + "-"*70)
if gpu_available:
    print("STATUS: GPU ENABLED - Using RTX 3050 for inference")
else:
    print("STATUS: GPU NOT AVAILABLE - Using CPU (slower)")
print("-"*70 + "\n")

app = Flask(__name__)

# =======================  MODEL LOADING  =======================
print("Loading trained model...")
try:
    # Load model on GPU if available, CPU otherwise
    with tf.device('/GPU:0' if gpu_available else '/CPU:0'):
        model = load_model('model.h5')
    print(f'✓ Model loaded successfully on {"GPU" if gpu_available else "CPU"}')
    print(f'  Check inference at: http://127.0.0.1:5000/')
except Exception as e:
    print(f'✗ ERROR: Failed to load model.h5: {e}')
    print('  Make sure model.h5 exists in the project root')
    sys.exit(1)

# Load label mapping
print("\nLoading class labels...")
labels = {}
labels_path = os.path.join(os.path.dirname(__file__), 'classes.json')
if os.path.exists(labels_path):
    try:
        with open(labels_path, 'r', encoding='utf-8') as f:
            class_names = json.load(f)
        labels = {i: name for i, name in enumerate(class_names)}
        print(f'✓ Loaded {len(labels)} class labels from classes.json')
        print(f'  Classes: {", ".join(list(labels.values())[:5])}{"..." if len(labels) > 5 else ""}')
    except Exception as e:
        print(f'✗ ERROR: Failed to load classes.json: {e}')
        labels = {0: 'Healthy', 1: 'Powdery', 2: 'Rust'}
        print(f'  Using fallback labels: {list(labels.values())}')

if not labels:
    print("⚠ WARNING: No classes loaded, using default labels")
    labels = {0: 'Healthy', 1: 'Powdery', 2: 'Rust'}

REMEDY_MAP = {
    'Pepper__bell___Bacterial_spot': (
        'Cause: Bacterial infection. Spray copper fungicide or streptomycin in early stages. ' 
        'Prevent with disease-free seeds, crop rotation, and avoid overhead watering.'
    ),
    'Pepper__bell___healthy': (
        'Plant is good. Maintain balanced NPK fertilizer, proper spacing, and regular inspection.'
    ),
    'Potato___Early_blight': (
        'Cause: Alternaria fungus. Spray Mancozeb or Chlorothalonil and remove infected leaves. ' 
        'Prevent with crop rotation and consistent watering.'
    ),
    'Potato___Late_blight': (
        'Cause: Phytophthora. Spray Metalaxyl+Mancozeb or Cymoxanil and remove infected plants. ' 
        'Prevent with good drainage and avoid wet leaves.'
    ),
    'Potato___healthy': (
        'Plant is good. Maintain loose soil, proper irrigation, and disease monitoring.'
    ),
    'Tomato_Bacterial_spot': (
        'Spray copper treatment and remove infected leaves. Prevent by avoiding leaf wetness and using resistant varieties.'
    ),
    'Tomato_Early_blight': (
        'Spray Mancozeb or Chlorothalonil. Prevent with mulching and crop rotation.'
    ),
    'Tomato_Late_blight': (
        'Spray Metalaxyl-based fungicides and remove infected plants quickly.'
    ),
    'Tomato_Leaf_Mold': (
        'Spray copper or sulfur fungicides. Prevent by improving ventilation and reducing humidity.'
    ),
    'Tomato_Septoria_leaf_spot': (
        'Spray Chlorothalonil or Mancozeb. Prevent by removing lower leaves and avoiding water splashes.'
    ),
    'Tomato_Spider_mites_Two_spotted_spider_mite': (
        'Treat with neem oil or insecticidal soap, and abamectin for severe cases. Prevent by maintaining humidity and washing leaves.'
    ),
    'Tomato__Target_Spot': (
        'Spray Azoxystrobin or Chlorothalonil to control target spot.'
    ),
    'Tomato__Tomato_YellowLeaf__Curl_Virus': (
        'Viral disease spread by whiteflies. Remove infected plants and control whiteflies with neem oil or imidacloprid.'
    ),
    'Tomato__Tomato_mosaic_virus': (
        'Viral disease with no cure. Remove infected plants, disinfect tools, and avoid tobacco contact.'
    ),
    'Tomato_healthy': (
        'Plant is good. Keep balanced nutrients, proper spacing, and regular pest monitoring.'
    ),
    'PlantVillage': (
        'This label is a dataset category, not a plant disease. Please use a real crop image for diagnosis.'
    ),
}

MEDICINE_MAP = {
    'Pepper__bell___Bacterial_spot': {
        'text': 'Copper-based fungicide is recommended. Example: Casa De Amor Copper Sulphate Fungicide.',
        'button_text': 'Buy Copper Fungicide',
        'button_url': 'https://www.example.com/casa-de-amor-copper-sulphate'
    },
    'Pepper__bell___healthy': {
        'text': 'No medicine needed. Use preventive sprays like Kocide or neem oil for protection.',
        'button_text': 'Preventive Spray Guide',
        'button_url': 'https://www.example.com/preventive-sprays'
    },
    'Potato___Early_blight': {
        'text': 'Mancozeb fungicide is recommended. Example product: POMAIS Mancozeb.',
        'button_text': 'Buy Mancozeb',
        'button_url': 'https://www.pomais.com/product/mancozeb/?utm_source=chatgpt.com'
    },
    'Potato___Late_blight': {
        'text': 'Metalaxyl + Mancozeb combination fungicide is recommended. Example product: Metalaxyl Mancozeb WP.',
        'button_text': 'Buy Metalaxyl + Mancozeb',
        'button_url': 'https://linuxcrop.com/public/shop/metalaxy-mancozeb-wp?utm_source=chatgpt.com'
    },
    'Potato___healthy': {
        'text': 'No medicine needed. Use preventive sprays like Kocide or neem oil for plant health.',
        'button_text': 'Preventive Spray Guide',
        'button_url': 'https://www.example.com/preventive-sprays'
    },
    'Tomato_Bacterial_spot': {
        'text': 'Copper fungicide is recommended. Example products: Hicopper or Gozaru Copper Oxychloride.',
        'button_text': 'Buy Copper Fungicide',
        'button_url': 'https://www.example.com/copper-oxychloride'
    },
    'Tomato_Early_blight': {
        'text': 'Mancozeb or Chlorothalonil fungicide is recommended.',
        'button_text': 'Buy Mancozeb',
        'button_url': 'https://www.pomais.com/product/mancozeb/?utm_source=chatgpt.com'
    },
    'Tomato_Late_blight': {
        'text': 'Metalaxyl-based fungicide is recommended. Example product: Metalaxyl + Chlorothalonil.',
        'button_text': 'Buy Metalaxyl Fungicide',
        'button_url': 'https://linuxcrop.com/public/shop/metalaxy-mancozeb-wp?utm_source=chatgpt.com'
    },
    'Tomato_Leaf_Mold': {
        'text': 'Copper fungicide is recommended. Example products: Hicopper or Gozaru Copper Oxychloride.',
        'button_text': 'Buy Copper Fungicide',
        'button_url': 'https://www.example.com/copper-oxychloride'
    },
    'Tomato_Septoria_leaf_spot': {
        'text': 'Chlorothalonil or Mancozeb fungicide is recommended.',
        'button_text': 'Buy Mancozeb',
        'button_url': 'https://www.pomais.com/product/mancozeb/?utm_source=chatgpt.com'
    },
    'Tomato_Spider_mites_Two_spotted_spider_mite': {
        'text': 'Neem oil or organic pesticide is recommended. Example product: Garden Genie Neem Oil Spray.',
        'button_text': 'Buy Neem Oil',
        'button_url': 'https://www.example.com/neem-oil-spray'
    },
    'Tomato__Target_Spot': {
        'text': 'Mancozeb or Chlorothalonil fungicide is recommended.',
        'button_text': 'Buy Mancozeb',
        'button_url': 'https://www.pomais.com/product/mancozeb/?utm_source=chatgpt.com'
    },
    'Tomato__Tomato_YellowLeaf__Curl_Virus': {
        'text': 'No cure. Control whiteflies with neem oil or imidacloprid.',
        'button_text': 'Whitefly Control',
        'button_url': 'https://www.example.com/whitefly-control'
    },
    'Tomato__Tomato_mosaic_virus': {
        'text': 'No cure. Remove infected plants and control vectors with neem oil.',
        'button_text': 'Whitefly Control',
        'button_url': 'https://www.example.com/whitefly-control'
    },
    'Tomato_healthy': {
        'text': 'No medicine needed. Use preventive sprays like Kocide or neem oil for protection.',
        'button_text': 'Preventive Spray Guide',
        'button_url': 'https://www.example.com/preventive-sprays'
    },
    'PlantVillage': {
        'text': 'This is a dataset category. No treatment applies.',
        'button_text': 'Learn More',
        'button_url': 'https://www.example.com/dataset-info'
    },
}

print("\n" + "="*70)
print("APP READY FOR INFERENCE")
print("="*70 + "\n")


def getResult(image_path):
    """
    Inference function fully optimized for GPU (RTX 3050)
    Uses TF-optimized execution and batch processing when possible
    """
    try:
        from tensorflow.keras.preprocessing.image import load_img, img_to_array
        
        # Load and preprocess image
        img = load_img(image_path, target_size=(224, 224))
        x = img_to_array(img)
        x = x.astype('float32') / 255.0
        x = np.expand_dims(x, axis=0)
        
        # Determine device placement
        device = '/GPU:0' if gpu_available else '/CPU:0'
        
        # Run inference with explicit device placement
        with tf.device(device):
            # Use tf.function for graph optimization on GPU
            @tf.function
            def predict_fn(x):
                return model(x, training=False)
            
            predictions = predict_fn(x)[0].numpy()
        
        return predictions
    except Exception as e:
        print(f"Error in inference: {e}")
        import traceback
        traceback.print_exc()
        return None


INVALID_LABELS = {'PlantVillage'}

def get_top_predictions(predictions, k=3, invalid_labels=None):
    """Return top k predictions with confidence scores, excluding invalid labels."""
    if predictions is None:
        return []
    if invalid_labels is None:
        invalid_labels = set()

    scored = [
        (idx, float(predictions[idx]) * 100)
        for idx in range(len(predictions))
        if labels.get(idx) not in invalid_labels
    ]
    scored.sort(key=lambda x: x[1], reverse=True)

    results = []
    for idx, score in scored[:k]:
        results.append({
            'class': labels.get(idx, 'Unknown'),
            'confidence': score
        })
    return results

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        f = request.files['file']
        if f.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        try:
            basepath = os.path.dirname(__file__)
            file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
            f.save(file_path)
            
            predictions = getResult(file_path)
            if predictions is None:
                return jsonify({'error': 'Failed to process image'}), 500
            
            # Get top 3 predictions excluding PlantVillage
            top_preds = get_top_predictions(predictions, k=3, invalid_labels=INVALID_LABELS)
            
            # Choose highest valid prediction by excluding PlantVillage
            valid_scores = [
                (idx, float(predictions[idx]) * 100)
                for idx in range(len(predictions))
                if labels.get(idx) not in INVALID_LABELS
            ]
            if valid_scores:
                valid_scores.sort(key=lambda x: x[1], reverse=True)
                best_idx, best_conf = valid_scores[0]
                predicted_label = labels[best_idx]
                confidence = best_conf
            else:
                best_idx = np.argmax(predictions)
                predicted_label = labels[best_idx]
                confidence = float(predictions[best_idx]) * 100
            
            # Clean up uploaded file
            try:
                os.remove(file_path)
            except:
                pass
            
            remedy_text = REMEDY_MAP.get(
                predicted_label,
                'No specific remedy available. Maintain good plant care and monitor closely.'
            )
            medicine = MEDICINE_MAP.get(predicted_label, {
                'text': 'No specific medicine recommendation available. Follow good plant care and consult a local supplier.',
                'button_text': 'View plant care guide',
                'button_url': 'https://www.example.com/preventive-sprays'
            })
            # Return only the primary prediction, confidence, remedy guidance, and medicine recommendation
            return jsonify({
                'primary_prediction': predicted_label,
                'confidence': confidence,
                'remedy': remedy_text,
                'medicine': medicine['text'],
                'medicine_button_text': medicine['button_text'],
                'medicine_button_url': medicine['button_url']
            })
        except Exception as e:
            print(f"Error in prediction: {e}")
            return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
    
    return None


if __name__ == '__main__':
    app.run(debug=True)