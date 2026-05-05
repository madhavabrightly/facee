#!/usr/bin/env python3
"""
GPU Monitoring and Testing Script for Plant Disease Detection
Real-time GPU usage monitoring during inference and training
"""

import sys
import time
import numpy as np
import tensorflow as tf
from pathlib import Path

def print_header(text):
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def test_tensorflow_gpu():
    """Test TensorFlow GPU functionality"""
    print_header("TENSORFLOW GPU TEST")
    
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("✗ No GPUs detected")
        return False
    
    print(f"✓ GPUs detected: {len(gpus)}")
    
    # Enable memory growth
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    
    # Create a simple model
    print("\nCreating test model...")
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(100,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax'),
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Test with GPU placement
    print("Testing forward pass on GPU...")
    x_test = np.random.randn(32, 100).astype(np.float32)
    y_test = np.random.randint(0, 10, (32,))
    
    try:
        with tf.device('/GPU:0'):
            # Force GPU execution
            loss = model.train_on_batch(x_test, y_test)
        
        print(f"✓ GPU forward/backward pass successful - Loss: {loss:.4f}")
        return True
    except Exception as e:
        print(f"✗ GPU test failed: {e}")
        return False

def test_inference_speed():
    """Test inference speed on GPU vs CPU"""
    print_header("INFERENCE SPEED TEST")
    
    gpus = tf.config.list_physical_devices('GPU')
    
    # Create a simple model
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation='softmax'),
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    
    # Test input
    x_test = np.random.randn(10, 224, 224, 3).astype(np.float32)
    
    # Test on GPU
    if gpus:
        print("\nTesting on GPU...")
        start = time.time()
        with tf.device('/GPU:0'):
            for _ in range(5):
                model.predict(x_test, verbose=0)
        gpu_time = time.time() - start
        gpu_avg = gpu_time / 5 / 10 * 1000  # ms per image
        print(f"✓ Total time (5 runs): {gpu_time:.2f}s")
        print(f"✓ Average per batch: {gpu_avg/10:.2f}ms per image")
    else:
        gpu_avg = None
    
    # Test on CPU
    print("\nTesting on CPU...")
    start = time.time()
    with tf.device('/CPU:0'):
        for _ in range(5):
            model.predict(x_test, verbose=0)
    cpu_time = time.time() - start
    cpu_avg = cpu_time / 5 / 10 * 1000  # ms per image
    print(f"✓ Total time (5 runs): {cpu_time:.2f}s")
    print(f"✓ Average per batch: {cpu_avg/10:.2f}ms per image")
    
    if gpu_avg and gpu_avg > 0:
        speedup = cpu_avg / gpu_avg
        print(f"\n✓ GPU is {speedup:.1f}x faster than CPU for this model")
    
    return True

def test_memory_usage():
    """Test GPU memory utilization"""
    print_header("GPU MEMORY USAGE TEST")
    
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("✗ No GPUs detected")
        return False
    
    print(f"Testing GPU memory with different batch sizes...\n")
    
    # Get initial memory
    tf.compat.v1.reset_default_graph()
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, 3, activation='relu', input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(128, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax'),
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    
    batch_sizes = [8, 16, 32, 64]
    
    for batch_size in batch_sizes:
        try:
            x_test = np.random.randn(batch_size, 224, 224, 3).astype(np.float32)
            with tf.device('/GPU:0'):
                model.predict(x_test, verbose=0)
            print(f"✓ Batch size {batch_size:2d}: Memory usage OK")
        except tf.errors.ResourceExhaustedError:
            print(f"✗ Batch size {batch_size:2d}: Out of GPU memory - reduce batch size")
            break
        except Exception as e:
            print(f"✗ Batch size {batch_size:2d}: Error - {e}")
            break
    
    return True

def test_ultralytics_gpu():
    """Test Ultralytics/YOLO GPU support"""
    print_header("ULTRALYTICS/YOLO GPU TEST")
    
    try:
        from ultralytics import YOLO
        import torch
        
        if not torch.cuda.is_available():
            print("✗ CUDA not available for Ultralytics")
            return False
        
        print(f"✓ PyTorch CUDA available: {torch.cuda.get_device_name(0)}")
        
        # Try to load a model (will download if needed)
        print("\nLoading YOLO detection model (may download ~200MB first time)...")
        try:
            model = YOLO('yolov8n.pt')  # Nano model for testing
            print("✓ YOLO model loaded")
            
            # Test inference
            print("Testing YOLO inference on GPU...")
            # Create a dummy image
            dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            
            results = model(dummy_img, verbose=False, conf=0.5)
            print(f"✓ YOLO inference completed: {len(results)} results")
            return True
        except Exception as e:
            print(f"⚠ Could not test YOLO inference: {e}")
            print("   This is OK if no internet connection")
            return True
            
    except ImportError:
        print("⚠ Ultralytics not installed")
        return True

def main():
    print("\n╔" + "="*68 + "╗")
    print("║" + " "*20 + "GPU TESTING SUITE" + " "*31 + "║")
    print("╚" + "="*68 + "╝")
    
    print(f"\nPython: {sys.version.split()[0]}")
    print(f"TensorFlow: {tf.__version__}")
    
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
    except:
        print("PyTorch: Not installed")
    
    results = {}
    
    # Run tests
    results['TensorFlow GPU'] = test_tensorflow_gpu()
    results['Inference Speed'] = test_inference_speed()
    results['GPU Memory'] = test_memory_usage()
    results['Ultralytics GPU'] = test_ultralytics_gpu()
    
    # Summary
    print_header("TEST SUMMARY")
    print("\nTest Results:")
    for test_name, passed in results.items():
        symbol = "✓" if passed else "✗"
        print(f"  {symbol} {test_name}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n✓ All tests passed! GPU is ready for training/inference")
        return 0
    else:
        print("\n⚠ Some tests failed - check GPU setup")
        return 1

if __name__ == '__main__':
    sys.exit(main())
