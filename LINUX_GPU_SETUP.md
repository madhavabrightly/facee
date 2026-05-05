# Linux GPU Setup Guide - RTX 3050

This guide explains how to set up GPU acceleration for your plant disease detection project on Linux with an NVIDIA RTX 3050.

## Quick Start

If you already have NVIDIA drivers, CUDA, and cuDNN installed, skip to the **Python Setup** section.

## Prerequisites Check

First, verify what you have installed:

```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA
nvcc --version

# Check cuDNN (if installed in default location)
ls /usr/local/cuda/include/cudnn.h
```

---

## Step 1: Install NVIDIA Driver

Your RTX 3050 requires the NVIDIA driver.

### On Ubuntu/Debian WSL2:

```bash
# Update package list
sudo apt update

# Install NVIDIA driver
sudo apt install nvidia-driver-535  # or newer version

# Reboot WSL2 (exit and run 'wsl --shutdown' from Windows PowerShell, then restart)
```

### On native Ubuntu/Debian:

```bash
# Add NVIDIA driver PPA
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update

# Install latest driver
sudo apt install nvidia-driver-latest-dkms

# Reboot
sudo reboot
```

### Verify installation:

```bash
nvidia-smi
```

You should see your RTX 3050 listed with VRAM info.

---

## Step 2: Install CUDA Toolkit 12.4

TensorFlow 2.15+ requires CUDA 12.3+. The RTX 3050 supports CUDA 12.x.

### On Ubuntu WSL2:

```bash
# Download CUDA repository key
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-wsl-ubuntu

# Add CUDA repository
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-repo-wsl-ubuntu-12-4-local_12.4.0-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-12-4-local_12.4.0-1_amd64.deb

# Add signing key
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/3bf863cc.pub

# Update and install
sudo apt-get update
sudo apt-get install cuda-toolkit-12-4

# Set CUDA paths (add to ~/.bashrc)
echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

# Apply changes
source ~/.bashrc
```

### On native Ubuntu/Debian:

```bash
# Download from https://developer.nvidia.com/cuda-downloads
# Select: Linux > x86_64 > Ubuntu > 22.04 > deb (network) or deb (local)

# For network installer:
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-ubuntu2204
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-repo-ubuntu2204-12-4-local_12.4.0-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-4-local_12.4.0-1_amd64.deb
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
sudo apt-get update
sudo apt-get -y install cuda

# Set CUDA paths
echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### Verify CUDA:

```bash
nvcc --version
```

You should see "Cuda compilation tools, release 12.4" or similar.

---

## Step 3: Install cuDNN

cuDNN is the CUDA Deep Neural Network library required by TensorFlow.

### Download cuDNN:

1. Go to https://developer.nvidia.com/cudnn
2. Sign in (create account if needed - it's free)
3. Download **cuDNN 9.x for CUDA 12.x**
4. Extract and copy to CUDA directory:

```bash
# Navigate to download folder
cd ~/Downloads

# Extract cuDNN
tar -xzvf cudnn-linux-*.tar.xz

# Copy to CUDA
sudo cp cudnn-linux-x86_64-*/include/* /usr/local/cuda/include/
sudo cp cudnn-linux-x86_64-*/lib/* /usr/local/cuda/lib64/
sudo chmod a+r /usr/local/cuda/include/cudnn.h
sudo chmod a+r /usr/local/cuda/lib64/libcudnn*
```

### Verify cuDNN:

```bash
ls /usr/local/cuda/include/cudnn.h
```

---

## Step 4: Python Environment Setup

### Create Virtual Environment:

```bash
# Navigate to your project
cd ~/plant/plant_disease_detection-main

# Create virtual environment
python3 -m venv venv_gpu

# Activate it
source venv_gpu/bin/activate
```

### Install Dependencies:

```bash
# Upgrade pip
pip install --upgrade pip

# Install with GPU support
pip install -r requirements_gpu_linux.txt

# Or manually:
pip install tensorflow[and-cuda]==2.15.0 torch torchvision torchaudio ultralytics
pip install flask pillow opencv-python numpy scipy
```

### Verify GPU Installation:

```bash
# Run verification script
python gpu_setup_verification.py
```

Expected output: All components should be ✓

---

## Step 5: Test GPU

```bash
# Run comprehensive GPU tests
python gpu_test.py

# This will test:
# - TensorFlow GPU
# - Inference speed (GPU vs CPU)
# - GPU memory usage
# - Ultralytics GPU support
```

---

## Step 6: Train Your Model with GPU

Now you can train with full GPU acceleration:

```bash
# Activate environment
source venv_gpu/bin/activate

# Train on GPU (adjust as needed for your classes)
python train_model.py \
    --dataset Dataset \
    --classes Tomato_healthy Tomato_Leaf_Mold Tomato_Septoria_leaf_spot \
    --epochs 20 \
    --batch-size 32

# For more classes
python train_model.py \
    --classes Tomato_healthy Tomato_Leaf_Mold Tomato_Septoria_leaf_spot Tomato_Early_blight Tomato_Late_blight \
    --epochs 20
```

Training should now show `✓ GPU(s) found: 1` at startup.

---

## Step 7: Run Inference with GPU

```bash
# Start Flask app (uses GPU automatically)
python app.py

# Open http://127.0.0.1:5000/ in your browser
```

---

## Troubleshooting

### "No GPU found" error

```bash
# Check if drivers are installed
nvidia-smi

# If nvidia-smi fails, reinstall driver (see Step 1)

# Check CUDA
nvcc --version

# If nvcc not found, add to PATH:
export PATH=/usr/local/cuda/bin:$PATH

# Check LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Out of memory during training

Reduce batch size:
```bash
python train_model.py --batch-size 16  # Instead of 32
```

The RTX 3050 has 8GB VRAM, which supports batch sizes 32-64 for normal training.

### CUDA version mismatch

Ensure TensorFlow version matches CUDA version:
```bash
python -c "import tensorflow as tf; print(tf.sysconfig.get_build_info()['cuda_version'])"
```

Should show 12.4 or compatible version.

### WSL2 specific issues

If GPU not visible in WSL2:

```bash
# Check Windows version (requires Windows 11 build 22000+)
# Install NVIDIA GPU driver from Windows (not WSL)
# Install CUDA in WSL as shown above
# Restart WSL: exit terminal, then from Windows PowerShell run:
wsl --shutdown
```

---

## Performance Expectations

**RTX 3050 Performance:**
- Single 224x224 image inference: ~10-20ms
- Training batch size 32: ~2-5s per batch
- Full training (20 epochs, ~1000 images): ~15-30 minutes

If getting slow results, check:
1. `nvidia-smi` shows GPU memory usage
2. CPU/GPU temperature doesn't throttle
3. Batch size is appropriate (16-64 for RTX 3050)

---

## Advanced Configuration

### Enable TensorRT for faster inference

```bash
pip install tensorrt

# In your code:
import tensorflow as tf
import tensorrt as trt

# Convert model to TensorRT format
# (This requires additional setup, see TensorFlow docs)
```

### Use mixed precision training (faster, lower memory)

Already enabled in `train_model.py` when CUDA is available. The RTX 3050 has Tensor Cores that benefit from mixed_float16.

---

## Next Steps

1. Verify GPU: `python gpu_setup_verification.py`
2. Test performance: `python gpu_test.py`
3. Download dataset (if needed): `python download_dataset.py`
4. Start training: `python train_model.py --epochs 20`
5. Run inference: `python app.py`

---

## Support & Errors

If you encounter issues:

1. Run diagnostic: `python gpu_setup_verification.py`
2. Check NVIDIA system reports: `nvidia-smi -q`
3. For WSL2 specific issues, check Windows NVIDIA driver version

---

**Last Updated:** 2024
**Target:** RTX 3050 on Ubuntu/WSL2
**TensorFlow:** 2.15+ with CUDA 12.4
