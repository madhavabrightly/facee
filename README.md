# Plant Disease Detection

Project for classifying plant leaf images into three categories: Healthy, Powdery (mildew), and Rust. This repository contains a small convolutional neural network trained with Keras/TensorFlow, a saved model (`model.h5`), and a Flask web app (`app.py`) to serve predictions via a simple UI.

**Key artifacts**
- `app.py` — Flask application that loads `model.h5` and exposes a `/predict` endpoint.
- `Model_Training.ipynb` — Jupyter notebook showing data preparation, model architecture, training, evaluation, and `model.save("model.h5")`.
- `model.h5` — Trained Keras model (not included by default; place in repo root to run the app).
- `Dataset/` — Local dataset folder used for training, validation and testing (see structure below).

**Model (used in this project)**

- Type: Keras Sequential CNN (custom)
- Input size: 225x225x3
- Architecture (as implemented in `Model_Training.ipynb`):
  - `Conv2D(32, (3,3), activation='relu', input_shape=(225,225,3))`
  - `Conv2D(64, (3,3), activation='relu')`
  - `MaxPooling2D(pool_size=(2,2))`
  - `Flatten()`
  - `Dense(64, activation='relu')`
  - `Dense(3, activation='softmax')`  (3 classes)
- Loss: `categorical_crossentropy`
- Optimizer: `adam`
- Augmentation: `ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)`
- Training hyperparameters (from notebook): `batch_size=16`, `epochs=5`
- Saved model file: `model.h5`

**Dataset (used in this project)**

The notebook uses a local dataset organized under `Dataset/` with this expected structure:

> **DirectML note:** if you have a compatible GPU and installed
> `tensorflow-directml` (the default in `start_project.ps1`), TensorFlow will
> automatically pick up the DirectML device at startup.  You can verify the
> hardware by running `python train_model.py --epochs 1` which prints
> `tf.config.list_physical_devices()` before training.

The notebook uses a local dataset organized under `Dataset/` with this expected structure:

```
Dataset/
  Train/Train/
    Healthy/
    Powdery/
    Rust/
  Validation/Validation/
    Healthy/
    Powdery/
    Rust/
  Test/Test/
    Healthy/
    Powdery/
    Rust/
```

Notes:
- Each class folder contains the images used for that split. The notebook references these paths directly (e.g. `Dataset/Train/Train/Healthy`).
- This dataset appears to be a curated set of leaf images (three classes). If you sourced these images from a public dataset (e.g., PlantVillage), please cite the original dataset in your project documentation and include license/attribution.

**How to run (local)**

1. Install dependencies (recommended in a virtual environment):

```bash
# on Windows you can install the DirectML-enabled build for GPU support
# (use `tensorflow` only if you explicitly need the CPU-only build)
```bash
pip install tensorflow-directml keras flask pillow opencv-python numpy scipy kagglehub
```
```

2. If you don't yet have the training dataset, you can use the helper script
   below to pull the PlantVillage `emmarex/plantdisease` archive from Kaggle.

   ```bash
   pip install kagglehub            # also required by download_dataset.py
   python download_dataset.py       # downloads & unpacks into ./Dataset
   ```

   The script assumes your Kaggle API credentials are configured for
   `kagglehub` (see https://pypi.org/project/kagglehub/).

3. (Optional) train your own model from the downloaded data; the
   training script prints device information so you can confirm it’s using
   DirectML/CUDA/etc.  Epochs default to **5**:

   ```bash
   python train_model.py \
       --dataset Dataset \
       --classes Tomato_healthy Tomato_Leaf_Mold Tomato_Septoria_leaf_spot \
       --epochs 5
   ```

   Typical output begins with something like:

   ```text
   TensorFlow version 2.21.0
   Physical devices: [PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU')]
   ```

   or, on a GPU system with DirectML, you’ll see a `DirectML` device listed.

   The default class names may not exactly match your dataset; list the
   directories under `Dataset` and pass any three you like.  The resulting
   `model.h5` will be written in the repository root.

4. Place your trained `model.h5` (either created above or obtained otherwise)
   in the repository root (next to `app.py`).

3. (Optional) Create an `uploads/` folder at repo root for incoming images:

```bash
mkdir uploads
```

4. Start the Flask app:

```bash
python app.py
```

5. Open the UI in your browser at: http://127.0.0.1:5000/

You can also POST an image to `/predict` using the form in the UI. The endpoint returns one of the labels: `Healthy`, `Powdery`, or `Rust`.

**Quick inference example (curl)**

```bash
curl -F "file=@/path/to/leaf.jpg" http://127.0.0.1:5000/predict
```

You can also bypass the web server and run a quick Python check once you have
`tensorflow-directml` installed and a `model.h5` file (or plain `tensorflow` if
using CPU):

```bash
# replace the path with your Python interpreter if it's not on PATH
python - <<'PYCODE'
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

model = load_model('model.h5')

# prepare a single image
img = load_img('some_leaf.jpg', target_size=(225,225))
x = img_to_array(img).astype('float32') / 255.
x = np.expand_dims(x, axis=0)

pred = model.predict(x)[0]
print('raw softmax:', pred)
print('predicted class index:', np.argmax(pred))
PYCODE
```

**Project structure**

- `app.py` — Flask application and inference helper (`getResult` uses `target_size=(225,225)` and returns the softmax predictions).
- `Model_Training.ipynb` — Training notebook; builds the Sequential model, trains with `ImageDataGenerator`, and saves `model.h5`.
- `templates/` — HTML templates used by Flask (contains `index.html` and `import.html`).
- `static/` — CSS/JS assets for the frontend.

**Results & evaluation**

The notebook includes training/validation accuracy plots and basic evaluation. Re-run `Model_Training.ipynb` to reproduce experiments; adjust `epochs` and augmentation as needed for improved accuracy.

**Notes, tips and next steps**

- If you need better accuracy, consider transfer learning (MobileNet, ResNet, EfficientNet) and training for more epochs.
- Standardize image preprocessing between training and inference: both use rescaling to `[0,1]` and target size `225x225`.
- Add a `requirements.txt` if you want reproducible installs.
- If `model.h5` is large, consider storing it in a releases area or an external bucket rather than committing it to version control.

**License & attribution**

Include any dataset licenses or third-party attributions here if you used external data. If this is your own dataset, add a short license (e.g., MIT) if you want to open-source the code.

---

### Quick‑start helper script

A PowerShell starter script is included (`start_project.ps1`).  From the
project root you can run:

```powershell
./start_project.ps1
```

It will set up/activate the `.venv` environment, install packages,
create an `uploads/` folder, download the dataset (if missing), optionally
train `model.h5`, and finally start the Flask server.  Edit the `$TRAIN` flag
inside the script to retrain automatically.

### Prerequisites & common issues

1. **Python version** – TensorFlow 2.21 (used here) requires Python 3.11 on
   Windows; `py -3.11` is the safest way to create the venv.  `pip install
   tensorflow` will fail on 3.14/3.15 with "no matching distribution".
2. **Package installation** – the venv must have
   `tensorflow, keras, flask, pillow, opencv-python, numpy, scipy,
   kagglehub`.  `start_project.ps1` installs them automatically.
3. **Dataset download problems** – you need valid Kaggle API credentials for
   the `kagglehub` tool.  If the automated download fails, manually retrieve
   `emmarex/plantdisease` from Kaggle and unzip it under `Dataset/` using the
   structure documented earlier.
4. **`scipy` error while training** – when first running
   `train_model.py`, you may see `ImportError: This requires the scipy module.`
   Install it manually with `pip install scipy` or rerun via the start script.
5. **Virtual environment locked by `uv`** – do not try to install packages
   into the system Python (`uv`); always activate the `.venv` before
   installing or running Python commands.
6. **Missing model** – the web app expects `model.h5` in the repo root.  If
   it's absent, either train a new one (`train_model.py`) or copy it back.

Following these notes and using the provided script should make starting the
project seamless; refer to the earlier conversation in this README if you
need deeper context about problems faced.

---

If you'd like, I can also:
- add a `requirements.txt` with pinned package versions,
- add a brief `USAGE.md` with screenshots and curl examples, or
- convert the notebook training steps into a runnable `train.py` script.

