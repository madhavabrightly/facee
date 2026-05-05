# PowerShell helper to initialize and start the plant_disease_detection app
#
# Usage: open a PowerShell prompt in the project root and run
#
#   ./start_project.ps1
#
# This script performs the following actions:
#   * activates the local virtual environment (.venv)
#   * creates the uploads directory if it doesn't exist
#   * (optionally) downloads the dataset via kagglehub
#   * trains the model if model.h5 is missing
#   * launches the Flask server
#
# You can edit the `TRAIN` flag below to automatically re-train on startup.

$ErrorActionPreference = 'Stop'

# path to Python executable inside venv
$venvPython = Join-Path -Path $PSScriptRoot -ChildPath ".venv\Scripts\python.exe"
$activateScript = Join-Path -Path $PSScriptRoot -ChildPath ".venv\Scripts\Activate.ps1"

if (-not (Test-Path $venvPython)) {
    Write-Host "Virtual environment not found. Creating .venv with Python 3.11..." -ForegroundColor Yellow
    py -3.11 -m venv .venv
}

Write-Host "Activating virtual environment..." -ForegroundColor Cyan
& $activateScript

# ensure required packages are installed
Write-Host "Installing/updating required Python packages..." -ForegroundColor Cyan
pip install --upgrade pip setuptools wheel

# Install TensorFlow with GPU support (DirectML on Windows)
Write-Host "Installing TensorFlow-DirectML (GPU support for RTX 3050)..." -ForegroundColor Cyan
pip uninstall -y tensorflow tensorflow-gpu tensorflow-cuda
pip install tensorflow-directml

# Install other required packages
Write-Host "Installing other dependencies..." -ForegroundColor Cyan
pip install keras flask pillow opencv-python numpy scipy scikit-image matplotlib kagglehub

Write-Host "Package installation complete!" -ForegroundColor Green

# make uploads directory
$uploads = Join-Path $PSScriptRoot "uploads"
if (-not (Test-Path $uploads)) {
    New-Item -Path $uploads -ItemType Directory | Out-Null
    Write-Host "Created uploads/ directory" -ForegroundColor Green
}

# download dataset if not present
if (-not (Test-Path "Dataset")) {
    Write-Host "Dataset not found; downloading via kagglehub..." -ForegroundColor Yellow
    python download_dataset.py
}

# optionally train model if missing
$TRAIN = $true
if ($TRAIN -or -not (Test-Path "model.h5")) {
    Write-Host "Training model on all dataset classes (this may take a while)..." -ForegroundColor Yellow
    Write-Host "GPU training enabled - monitor Task Manager for GPU usage" -ForegroundColor Cyan
    python train_model.py --dataset Dataset --epochs 15 --batch-size 32
}

Write-Host "Starting Flask server..." -ForegroundColor Cyan
python app.py
