import os
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

model = load_model('model.h5')

# Load class labels
class_names = None
if os.path.exists('classes.json'):
    with open('classes.json', 'r', encoding='utf-8') as f:
        class_names = json.load(f)

img = load_img(r'Dataset\Tomato_healthy\000146ff-92a4-4db6-90ad-8fce2ae4fddd___GH_HL Leaf 259.1.JPG', target_size=(225,225))
x = img_to_array(img).astype('float32')/255.0
x = np.expand_dims(x,0)
pred = model.predict(x)[0]
idx = np.argmax(pred)
print('softmax:', pred)
print('class index:', idx)
print('class name:', class_names[idx] if class_names else 'unknown')
