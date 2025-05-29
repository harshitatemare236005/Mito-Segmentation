import numpy as np

def extract_patches(mask, patch_size=(64, 64), threshold=0.5):
    patches = []
    h, w = mask.shape
    for i in range(0, h - patch_size[0], patch_size[0]):
        for j in range(0, w - patch_size[1], patch_size[1]):
            patch = mask[i:i+patch_size[0], j:j+patch_size[1]]
            if np.mean(patch) > threshold:  # only keep relevant patches
                patches.append(patch)
    return patches
