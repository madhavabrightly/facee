# RTX 3050 GPU Setup - Quick Reference

Complete GPU acceleration setup for your plant disease detection project on Linux.

## Files Added

1. **gpu_setup_verification.py** - Diagnostic tool to check GPU setup
2. **gpu_test.py** - Performance testing suite
3. **requirements_gpu_linux.txt** - GPU-enabled Python packages for Linux
4. **setup_gpu_linux.sh** - Automated setup script
5. **LINUX_GPU_SETUP.md** - Detailed setup guide
6. **Updated app.py** - Enhanced with full GPU configuration
7. **Updated train_model.py** - Optimized for GPU training

## Quick Start (30 minutes)

### If you already have CUDA + cuDNN installed:

```bash
cd ~/plant/plant_disease_detection-main

# Quick setup (Python environment only)
bash setup_gpu_linux.sh --python-only

# Verify GPU is working
source venv_gpu/bin/activate
python gpu_setup_verification.py
python gpu_test.py

# Start training
python train_model.py --epochs 20

# Run inference
python app.py
```

### If you need to install CUDA + cuDNN first:

```bash
# Step 1: Install NVIDIA driver
nvidia-smi  # Should show your RTX 3050

# Step 2: Install CUDA 12.4
# See step 2 in LINUX_GPU_SETUP.md

# Step 3: Install cuDNN
# See step 3 in LINUX_GPU_SETUP.md

# Step 4: Run automated setup
cd ~/plant/plant_disease_detection-main
bash setup_gpu_linux.sh

# This will:
# ✓ Verify NVIDIA driver
# ✓ Verify CUDA toolkit
# ✓ Verify cuDNN
# ✓ Create Python virtual environment
# ✓ Install all GPU packages
# ✓ Run verification tests
```

---

## Verification Tests

```bash
# Check that everything is installed correctly
python gpu_setup_verification.py

# Run performance tests
python gpu_test.py

# This will show:
# ✓ TensorFlow GPU detection
# ✓ PyTorch GPU detection  
# ✓ Inference speed (GPU vs CPU)
# ✓ GPU memory usage
# ✓ Ultralytics YOLO GPU support
```

---

## Training with GPU

```bash
# Train model with full GPU acceleration
python train_model.py \
    --dataset Dataset \
    --classes Tomato_healthy Tomato_Leaf_Mold Tomato_Septoria_leaf_spot \
    --epochs 20 \
    --batch-size 32

# Will show:
# ✓ GPU(s) found: 1
# ✓ Mixed precision enabled
# ✓ Training on tensor cores
```

---

## Inference with GPU

```bash
# Start Flask web server (uses GPU automatically)
python app.py

# Open http://127.0.0.1:5000/
# Upload images for instant GPU-accelerated predictions
```

---

## Environment Activation

Every time you use the project:

```bash
# Activate the GPU environment
source venv_gpu/bin/activate

# Run your commands
python train_model.py ...
python app.py
```

**Deactivate when done:**
```bash
deactivate
```

---

## Code Changes Summary

### app.py
- ✅ Enhanced GPU detection and configuration
- ✅ Optimized inference with tf.function for graph compilation
- ✅ GPU memory growth settings
- ✅ Better error handling and diagnostics

### train_model.py
- ✅ Mixed precision training (float16 for speed, float32 for stability)
- ✅ GPU memory optimization
- ✅ Batch normalization for training stability
- ✅ Learning rate reduction callback
- ✅ Comprehensive GPU status reporting

---

## RTX 3050 Performance

**Your GPU:**
- **Memory:** 8 GB VRAM
- **Architecture:** Ampere (GA107)
- **Tensor Cores:** 2560
- **Max Batch Size:** 32-64 (recommended: 32)
- **Inference Speed:** ~10-20ms per 224×224 image
- **Training Speed:** ~2-5s per 32-image batch

**Comparison (relative to CPU):**
- Training: **10-30x faster** than CPU
- Inference: **5-15x faster** than CPU

---

## Troubleshooting

### Problem: "No GPU found"
```bash
# Check driver
nvidia-smi

# Check CUDA
nvcc --version

# Set CUDA paths
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Run diagnostic
python gpu_setup_verification.py
```

### Problem: Out of Memory
```bash
# Reduce batch size
python train_model.py --batch-size 16
```

### Problem: WSL2 GPU not working
```bash
# Inside WSL2, install CUDA (not Windows CUDA)
# Ensure Windows has NVIDIA driver installed
# Run from WSL: wsl --shutdown (from Windows PowerShell)
# Then restart WSL
```

---

## Next Steps

1. **Verify Setup:**
   ```bash
   python gpu_setup_verification.py
   ```

2. **Download Dataset (if needed):**
   ```bash
   python download_dataset.py
   ```

3. **Train Your Model:**
   ```bash
   python train_model.py --epochs 20
   ```

4. **Run Web Interface:**
   ```bash
   python app.py
   ```

---

## Documentation

- **Detailed Setup:** See [LINUX_GPU_SETUP.md](LINUX_GPU_SETUP.md)
- **Troubleshooting:** See [LINUX_GPU_SETUP.md - Troubleshooting section](LINUX_GPU_SETUP.md#troubleshooting)
- **NVIDIA CUDA:** https://developer.nvidia.com/cuda-downloads
- **NVIDIA cuDNN:** https://developer.nvidia.com/cudnn
- **TensorFlow GPU:** https://www.tensorflow.org/install/gpu
- **PyTorch GPU:** https://pytorch.org/get-started/locally/

---

## System Requirements Checklist

- ☑️ NVIDIA RTX 3050 GPU
- ☑️ NVIDIA Driver 535+ (check: `nvidia-smi`)
- ☑️ CUDA Toolkit 12.4 (check: `nvcc --version`)
- ☑️ cuDNN 9.x (check: `ls /usr/local/cuda/include/cudnn.h`)
- ☑️ Python 3.9+ (check: `python3 --version`)
- ☑️ 8+ GB disk space for datasets
- ☑️ 4+ GB RAM for Python environment

---

## Performance Monitoring

Watch GPU usage during training:

```bash
# In another terminal, monitor GPU
watch -n 1 nvidia-smi

# Or continuous output
nvidia-smi dmon

# Or with Python monitoring
python gpu_test.py
```

---

Version 1.0 - March 2024
Optimized for RTX 3050 on Linux/WSL2
