import numpy as np
from PIL import Image
import tifffile as tiff

def load_tif_image(path_or_file):
    if isinstance(path_or_file, str):
        image = tiff.imread(path_or_file)
    else:
        image = tiff.imread(path_or_file)  # file-like object
    return image

def preprocess_image(img, target_size=(256, 256)):
    img_resized = Image.fromarray(img).resize(target_size)
    img_array = np.array(img_resized) / 255.0
    if img_array.ndim == 2:
        img_array = np.expand_dims(img_array, axis=-1)
    return np.expand_dims(img_array, axis=0)
