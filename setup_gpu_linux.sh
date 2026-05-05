#!/bin/bash
# Automated GPU Setup Script for Plant Disease Detection
# For Ubuntu/WSL2 with RTX 3050
# Usage: bash setup_gpu_linux.sh [--help]

set -e  # Exit on error

VERSION="1.0"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

echo_success() {
    echo -e "${GREEN}✓${NC} $1"
}

echo_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

echo_error() {
    echo -e "${RED}✗${NC} $1"
}

print_header() {
    echo ""
    echo "═══════════════════════════════════════════════════════════════════"
    echo "  $1"
    echo "═══════════════════════════════════════════════════════════════════"
    echo ""
}

check_command() {
    if command -v $1 &> /dev/null; then
        return 0
    else
        return 1
    fi
}

show_help() {
    cat << EOF
GPU Setup Script for Plant Disease Detection - RTX 3050

Usage: bash setup_gpu_linux.sh [OPTIONS]

Options:
    --help              Show this help message
    --skip-driver       Skip NVIDIA driver check (assumes installed)
    --skip-cuda         Skip CUDA installation check
    --skip-cudnn        Skip cuDNN installation check
    --python-only       Only set up Python environment (assumes CUDA/cuDNN ready)
    --verify-only       Only verify existing installation

Examples:
    # Full setup
    bash setup_gpu_linux.sh

    # Only set up Python environment
    bash setup_gpu_linux.sh --python-only

    # Verify installation
    bash setup_gpu_linux.sh --verify-only

EOF
}

check_nvidia_driver() {
    print_header "Checking NVIDIA Driver"
    
    if ! check_command nvidia-smi; then
        echo_error "nvidia-smi not found"
        echo_info "Install NVIDIA drivers from https://www.nvidia.com/en-us/geforce/drivers/"
        return 1
    fi
    
    nvidia-smi
    echo_success "NVIDIA driver found"
    return 0
}

check_cuda() {
    print_header "Checking CUDA Toolkit"
    
    if ! check_command nvcc; then
        echo_error "CUDA compiler (nvcc) not found"
        echo_info "Install CUDA from https://developer.nvidia.com/cuda-downloads"
        return 1
    fi
    
    nvcc --version
    echo_success "CUDA found"
    return 0
}

check_cudnn() {
    print_header "Checking cuDNN"
    
    if [ -f "/usr/local/cuda/include/cudnn.h" ]; then
        echo_success "cuDNN header found at /usr/local/cuda/include/cudnn.h"
        return 0
    elif [ -f "/usr/include/cudnn.h" ]; then
        echo_success "cuDNN header found at /usr/include/cudnn.h"
        return 0
    else
        echo_error "cuDNN header not found"
        echo_info "Download from https://developer.nvidia.com/cudnn and extract to /usr/local/cuda"
        return 1
    fi
}

setup_python_env() {
    print_header "Setting Up Python Environment"
    
    # Check Python version
    if ! check_command python3; then
        echo_error "Python 3 not found"
        return 1
    fi
    
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    echo_info "Python version: $PYTHON_VERSION"
    
    # Create virtual environment
    if [ -d "$SCRIPT_DIR/venv_gpu" ]; then
        echo_warning "Virtual environment already exists"
        read -p "Remove and recreate? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$SCRIPT_DIR/venv_gpu"
            python3 -m venv "$SCRIPT_DIR/venv_gpu"
            echo_success "Virtual environment created"
        fi
    else
        python3 -m venv "$SCRIPT_DIR/venv_gpu"
        echo_success "Virtual environment created at $SCRIPT_DIR/venv_gpu"
    fi
    
    # Activate virtual environment
    echo_info "Activating virtual environment..."
    source "$SCRIPT_DIR/venv_gpu/bin/activate"
    
    # Upgrade pip
    echo_info "Upgrading pip..."
    pip install --upgrade pip setuptools wheel
    
    # Install dependencies
    echo_info "Installing ML packages with GPU support..."
    if [ -f "$SCRIPT_DIR/requirements_gpu_linux.txt" ]; then
        pip install -r "$SCRIPT_DIR/requirements_gpu_linux.txt"
    else
        echo_warning "requirements_gpu_linux.txt not found, installing manually..."
        pip install tensorflow[and-cuda]==2.15.0
        pip install torch torchvision torchaudio
        pip install ultralytics
        pip install flask pillow opencv-python numpy scipy pandas matplotlib seaborn
        pip install jupyter ipython
    fi
    
    echo_success "Python environment set up"
    return 0
}

set_cuda_paths() {
    print_header "Setting CUDA Environment Variables"
    
    # Check if already set
    if [ -z "$CUDA_HOME" ]; then
        echo_info "Setting CUDA_HOME..."
        export CUDA_HOME=/usr/local/cuda
        export PATH=$CUDA_HOME/bin:$PATH
        export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
        
        # Add to bashrc if not already there
        if ! grep -q "export CUDA_HOME" ~/.bashrc; then
            echo "" >> ~/.bashrc
            echo "# CUDA paths" >> ~/.bashrc
            echo "export CUDA_HOME=/usr/local/cuda" >> ~/.bashrc
            echo "export PATH=\$CUDA_HOME/bin:\$PATH" >> ~/.bashrc
            echo "export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc
            echo_success "CUDA paths added to ~/.bashrc"
        fi
        
        source ~/.bashrc
    else
        echo_success "CUDA_HOME already set to $CUDA_HOME"
    fi
    
    return 0
}

verify_installation() {
    print_header "Verifying Installation"
    
    if [ -f "$SCRIPT_DIR/gpu_setup_verification.py" ]; then
        python3 "$SCRIPT_DIR/gpu_setup_verification.py"
    else
        echo_warning "gpu_setup_verification.py not found"
    fi
    
    return 0
}

main() {
    echo ""
    echo "╔═══════════════════════════════════════════════════════════════════╗"
    echo "║   GPU Setup Script - Plant Disease Detection (RTX 3050)           ║"
    echo "║   Version $VERSION                                                    ║"
    echo "╚═══════════════════════════════════════════════════════════════════╝"
    
    # Parse arguments
    SKIP_DRIVER=false
    SKIP_CUDA=false
    SKIP_CUDNN=false
    PYTHON_ONLY=false
    VERIFY_ONLY=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --help)
                show_help
                exit 0
                ;;
            --skip-driver)
                SKIP_DRIVER=true
                shift
                ;;
            --skip-cuda)
                SKIP_CUDA=true
                shift
                ;;
            --skip-cudnn)
                SKIP_CUDNN=true
                shift
                ;;
            --python-only)
                PYTHON_ONLY=true
                shift
                ;;
            --verify-only)
                VERIFY_ONLY=true
                shift
                ;;
            *)
                echo_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # Verify-only mode
    if [ "$VERIFY_ONLY" = true ]; then
        check_nvidia_driver
        check_cuda
        check_cudnn
        verify_installation
        exit $?
    fi
    
    # Python-only mode (skip hardware checks)
    if [ "$PYTHON_ONLY" = true ]; then
        set_cuda_paths
        setup_python_env
        verify_installation
        exit 0
    fi
    
    # Full setup
    if [ "$SKIP_DRIVER" = false ]; then
        check_nvidia_driver || exit 1
    fi
    
    if [ "$SKIP_CUDA" = false ]; then
        check_cuda || exit 1
    fi
    
    if [ "$SKIP_CUDNN" = false ]; then
        check_cudnn || exit 1
    fi
    
    set_cuda_paths
    setup_python_env
    verify_installation
    
    echo ""
    echo "═══════════════════════════════════════════════════════════════════"
    echo_success "GPU setup complete!"
    echo "═══════════════════════════════════════════════════════════════════"
    echo ""
    echo "Next steps:"
    echo "1. Activate environment: source $SCRIPT_DIR/venv_gpu/bin/activate"
    echo "2. Download dataset: python download_dataset.py"
    echo "3. Train model: python train_model.py --epochs 20"
    echo "4. Run inference: python app.py"
    echo ""
}

main "$@"
