#!/usr/bin/env python3
"""
GPU Setup Verification Script for RTX 3050 on Linux
This script checks CUDA, cuDNN, and all ML framework GPU support
"""

import os
import sys
import subprocess
from pathlib import Path

def print_header(text):
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def check_cuda():
    """Check NVIDIA CUDA toolkit installation"""
    print_header("CHECKING CUDA TOOLKIT")
    
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ CUDA Compiler (nvcc) found:")
            print(result.stdout)
            return True
        else:
            print("✗ CUDA Compiler (nvcc) not in PATH")
            return False
    except FileNotFoundError:
        print("✗ CUDA Compiler (nvcc) not found")
        print("  Install CUDA Toolkit: https://developer.nvidia.com/cuda-downloads")
        return False

def check_nvidia_driver():
    """Check NVIDIA GPU driver"""
    print_header("CHECKING NVIDIA DRIVER & GPU")
    
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(result.stdout)
            return True
        else:
            print("✗ nvidia-smi failed")
            return False
    except FileNotFoundError:
        print("✗ nvidia-smi not found - GPU driver not installed")
        print("  Install NVIDIA driver: https://www.nvidia.com/en-us/geforce/drivers/")
        return False
    except Exception as e:
        print(f"✗ Error running nvidia-smi: {e}")
        return False

def check_cudnn():
    """Check cuDNN installation"""
    print_header("CHECKING cuDNN")
    
    common_paths = [
        "/usr/local/cuda/include/cudnn.h",
        "/usr/include/cudnn.h",
        "/opt/cuda/include/cudnn.h",
        os.path.expanduser("~/.local/cuda/include/cudnn.h"),
    ]
    
    found = False
    for path in common_paths:
        if os.path.exists(path):
            print(f"✓ cuDNN header found at: {path}")
            found = True
            break
    
    if not found:
        # Try pkg-config
        try:
            result = subprocess.run(['pkg-config', '--cflags', 'cudnn'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✓ cuDNN found via pkg-config: {result.stdout.strip()}")
                found = True
        except:
            pass
    
    if not found:
        print("✗ cuDNN not found in standard locations")
        print("  Install cuDNN: https://developer.nvidia.com/cudnn")
        print("  After install, ensure it's in PATH or LD_LIBRARY_PATH")
    
    return found

def check_tensorflow_gpu():
    """Check TensorFlow GPU support"""
    print_header("CHECKING TENSORFLOW GPU SUPPORT")
    
    try:
        import tensorflow as tf
        
        print(f"TensorFlow version: {tf.__version__}")
        
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"\n✓ GPU(s) detected: {len(gpus)}")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu.name}")
                try:
                    details = tf.config.experimental.get_device_details(gpu)
                    if details:
                        print(f"    Details: {details}")
                except:
                    pass
            return True
        else:
            print("✗ No GPUs detected by TensorFlow")
            print("  Possible causes:")
            print("  1. CUDA/cuDNN not properly installed")
            print("  2. GPU driver not installed")
            print("  3. TensorFlow-CPU installed instead of GPU version")
            return False
            
    except ImportError:
        print("✗ TensorFlow not installed")
        return False
    except Exception as e:
        print(f"✗ Error checking TensorFlow: {e}")
        return False

def check_torch_gpu():
    """Check PyTorch GPU support"""
    print_header("CHECKING PYTORCH GPU SUPPORT")
    
    try:
        import torch
        
        print(f"PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.is_available()}")
            print(f"✓ GPU device: {torch.cuda.get_device_name(0)}")
            print(f"✓ GPU count: {torch.cuda.device_count()}")
            return True
        else:
            print("✗ CUDA not available in PyTorch")
            return False
            
    except ImportError:
        print("⚠ PyTorch not installed (optional for this project)")
        return True  # Not an error since it's optional
    except Exception as e:
        print(f"✗ Error checking PyTorch: {e}")
        return False

def check_ultralytics_gpu():
    """Check Ultralytics (YOLO) GPU support"""
    print_header("CHECKING ULTRALYTICS GPU SUPPORT")
    
    try:
        from ultralytics import YOLO
        import torch
        
        if torch.cuda.is_available():
            print(f"✓ Ultralytics can use GPU: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("✗ GPU not available for Ultralytics")
            return False
            
    except ImportError:
        print("⚠ Ultralytics not installed (optional)")
        return True
    except Exception as e:
        print(f"✗ Error checking Ultralytics: {e}")
        return False

def check_environment_variables():
    """Check important environment variables"""
    print_header("ENVIRONMENT VARIABLES")
    
    important_vars = [
        'CUDA_HOME',
        'CUDA_PATH',
        'CUDA_VISIBLE_DEVICES',
        'LD_LIBRARY_PATH',
        'PATH',
    ]
    
    for var in important_vars:
        value = os.environ.get(var, "NOT SET")
        if value != "NOT SET":
            print(f"✓ {var}: {value}")
        else:
            print(f"  {var}: NOT SET")

def check_ld_library_path():
    """Check if CUDA libraries are in LD_LIBRARY_PATH"""
    print_header("CUDA LIBRARY PATH CHECK")
    
    ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    cuda_paths = ['/usr/local/cuda/lib64', '/usr/local/cuda/lib']
    
    print("Current LD_LIBRARY_PATH:")
    if ld_path:
        for path in ld_path.split(':'):
            print(f"  {path}")
    else:
        print("  (empty)")
    
    found_cuda = False
    for cuda_path in cuda_paths:
        if cuda_path in ld_path:
            print(f"\n✓ CUDA lib path found: {cuda_path}")
            found_cuda = True
            break
    
    if not found_cuda:
        print(f"\n⚠ CUDA library path not in LD_LIBRARY_PATH")
        print(f"  Consider adding to ~/.bashrc or ~/.zshrc:")
        print(f"  export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH")

def main():
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "GPU SETUP VERIFICATION FOR RTX 3050" + " "*18 + "║")
    print("╚" + "="*68 + "╝")
    
    results = {}
    
    # Run checks
    results['NVIDIA Driver'] = check_nvidia_driver()
    results['CUDA Toolkit'] = check_cuda()
    results['cuDNN'] = check_cudnn()
    results['TensorFlow GPU'] = check_tensorflow_gpu()
    results['PyTorch GPU'] = check_torch_gpu()
    results['Ultralytics GPU'] = check_ultralytics_gpu()
    check_environment_variables()
    check_ld_library_path()
    
    # Summary
    print_header("SUMMARY")
    print("\nComponent Status:")
    for component, status in results.items():
        symbol = "✓" if status else "✗"
        print(f"  {symbol} {component}")
    
    all_critical_ok = all([
        results['NVIDIA Driver'],
        results['CUDA Toolkit'],
        results['TensorFlow GPU'],
    ])
    
    if all_critical_ok:
        print("\n✓ All critical components are set up correctly!")
        print("  Your RTX 3050 should work with TensorFlow and Ultralytics")
        return 0
    else:
        print("\n✗ Some critical components are missing")
        print("  Please refer to the installation instructions above")
        return 1

if __name__ == '__main__':
    sys.exit(main())
