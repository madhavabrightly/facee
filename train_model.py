"""Train a simple CNN on three selected classes from the downloaded dataset.

The dataset folder produced by ``download_dataset.py`` contains dozens of
PlantVillage categories.  This script lets you pick three of them and
trains the same architecture used in the notebook, saving a ``model.h5`` in
the repository root when finished.

Example::

    python train_model.py \
        --dataset Dataset \
        --classes Tomato_healthy Tomato_Leaf_Mold Tomato_Septoria_leaf_spot \
        --epochs 3

Parameters
----------
- ``--dataset``: root directory that contains class subfolders.
- ``--classes``: three class names (subfolder names) to train on.  Order is
  important because the output labels will be 0,1,2 in the same sequence.
- ``--epochs``: number of passes over the training data.

The script uses a 80/20 validation split via ``ImageDataGenerator`` and
``flow_from_directory`` so there is no need to reorganize the files manually.
"""

import argparse
import json
import os
import sys

# TensorFlow and GPU setup
import tensorflow as tf
from tensorflow.keras import mixed_precision

print("\n" + "="*70)
print("TRAINING GPU CONFIGURATION FOR RTX 3050")
print("="*70)
print(f"TensorFlow version: {tf.__version__}")
print(f"Python version: {sys.version}")

# List all devices
all_devices = tf.config.list_physical_devices()
print(f"\nAll devices detected: {len(all_devices)}")
for dev in all_devices:
    print(f"  - {dev}")

# GPU-specific setup
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"\n✓ GPU(s) found: {len(gpus)}")
    for i, gpu in enumerate(gpus):
        try:
            # Enable memory growth to prevent OOM
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"  ✓ GPU {i}: Memory growth enabled")
            # Get device details
            details = tf.config.experimental.get_device_details(gpu)
            if details:
                print(f"     Details: {details}")
        except RuntimeError as e:
            print(f"  ✗ GPU {i}: Could not configure - {e}")
    
    # Enable mixed precision for faster training on supported GPUs (RTX 3050 supports this)
    try:
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
        print(f"\n✓ Mixed precision enabled: {policy}")
        print("  Note: RTX 3050 has Tensor Cores that benefit from mixed precision training")
    except Exception as e:
        print(f"\n⚠ Mixed precision not available: {e}")
        print("  Training will continue with float32 (slower)")
else:
    print("\n⚠ NO GPU DEVICES FOUND - Using CPU for training")
    print("   Training will be much slower than GPU")
    print("   On Linux: install tensorflow[and-cuda] for GPU support")

print("="*70 + "\n")

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam


def find_classes(dataset_dir):
    class_dirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    class_dirs.sort()
    if not class_dirs:
        raise RuntimeError(f"No class subdirectories found in {dataset_dir}")
    return class_dirs


def build_args():
    p = argparse.ArgumentParser(description="Train CNN on PlantVillage classes")
    p.add_argument("--dataset", default="Dataset", help="root folder containing class subdirectories")
    p.add_argument(
        "--classes",
        nargs="*",
        metavar="CLASS",
        help="class names (subfolders) to use for training; default = all",
        default=None,
    )
    p.add_argument("--epochs", type=int, default=15, help="number of epochs to train")
    p.add_argument("--batch-size", type=int, default=32, help="batch size")
    p.add_argument("--img-size", type=int, default=224, help="image size")
    p.add_argument("--model-path", default="model.h5", help="output model path")
    p.add_argument("--labels-path", default="classes.json", help="output label mapping path")
    return p.parse_args()


def build_model(num_classes, input_shape=(224, 224, 3)):
    """
    Build MobileNetV2-based model optimized for GPU training
    Using transfer learning for better performance and faster training
    """
    print("Building neural network model...")
    print(f"  Base model: MobileNetV2 (pre-trained on ImageNet)")
    print(f"  Input shape: {input_shape}")
    print(f"  Output classes: {num_classes}")
    
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    base_model.trainable = False  # Freeze base model weights for transfer learning

    model = Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),          # Added for stability
        layers.Dropout(0.4),
        # Using float32 for dense layers when mixed_float16 is enabled
        layers.Dense(256, activation='relu', dtype='float32'),
        layers.BatchNormalization(),          # Added for stability
        layers.Dropout(0.3),
        # Output layer MUST be float32 for numerical stability
        layers.Dense(num_classes, activation='softmax', dtype='float32'),
    ])

    # Use a lower learning rate with GPU training and transfer learning
    optimizer = Adam(learning_rate=1e-4, clipnorm=1.0)  # Added gradient clipping for stability
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy')]  # Added top-2 accuracy
    )
    
    print("✓ Model compiled successfully")
    return model


def main():
    args = build_args()
    
    print("\n" + "="*70)
    print("TRAINING CONFIGURATION")
    print("="*70)
    print(f"Dataset location: {args.dataset}")
    print(f"Batch size: {args.batch_size} (larger = faster but more VRAM)")
    print(f"Image size: {args.img_size}x{args.img_size}")
    print(f"Epochs: {args.epochs}")
    print(f"GPU devices available: {len(tf.config.list_physical_devices('GPU'))}")

    if args.classes:
        class_list = args.classes
    else:
        class_list = find_classes(args.dataset)

    print(f"Classes to train: {class_list}")
    print("="*70 + "\n")

    # Save class mapping
    with open(args.labels_path, 'w', encoding='utf-8') as f:
        json.dump(class_list, f, ensure_ascii=False, indent=2)
    print(f"✓ Class labels saved to {args.labels_path}")

    image_size = (args.img_size, args.img_size)

    # Data augmentation with optimization for GPU training
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,            # Slightly increased for better augmentation
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',          # Added for consistency
        validation_split=0.2,
    )

    print("Loading training data from disk...")
    train_gen = train_datagen.flow_from_directory(
        args.dataset,
        target_size=image_size,
        classes=class_list,
        class_mode='sparse',
        subset='training',
        shuffle=True,
        batch_size=args.batch_size,
        follow_links=True,
    )

    print("Loading validation data from disk...")
    val_gen = train_datagen.flow_from_directory(
        args.dataset,
        target_size=image_size,
        classes=class_list,
        class_mode='sparse',
        subset='validation',
        shuffle=False,
        batch_size=args.batch_size,
        follow_links=True,
    )

    model = build_model(num_classes=len(class_list), input_shape=(args.img_size, args.img_size, 3))
    
    # Display model architecture
    print("\nModel architecture:")
    model.summary()

    # Define callbacks for GPU training
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        args.model_path,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1,
        save_freq='epoch'  # Save after each epoch
    )
    
    early_stop_cb = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,              # Increased patience for more stable training
        restore_best_weights=True,
        verbose=1
    )
    
    # Learning rate reduction for fine-tuning
    reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )

    print("\n" + "="*70)
    print("STARTING TRAINING ON GPU")
    print(f"Total steps per epoch: {len(train_gen)}")
    print(f"Total validation steps: {len(val_gen)}")
    print("="*70 + "\n")

    # Train with GPU optimization
    history = model.fit(
        train_gen,
        epochs=args.epochs,
        validation_data=val_gen,
        callbacks=[checkpoint_cb, early_stop_cb, reduce_lr_cb],
        verbose=1,
        # Optimized for GPU - use multiprocessing carefully on WSL2
        workers=2,
        use_multiprocessing=False,  # False for WSL2, True for native Linux if available
        max_queue_size=10,
    )

    # Final summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"✓ Model saved to: {args.model_path}")
    print(f"✓ Classes saved to: {args.labels_path}")
    print(f"✓ Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
    if 'top_2_accuracy' in history.history:
        print(f"✓ Best top-2 accuracy: {max(history.history['top_2_accuracy']):.4f}")
    print("="*70 + "\n")
    print("To run inference, execute:")
    print("  python app.py")
    print("  Then visit http://127.0.0.1:5000/")


if __name__ == "__main__":
    main()
