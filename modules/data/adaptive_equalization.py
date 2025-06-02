from skimage import exposure
from PIL import Image
import numpy as np

class AdaptiveEqualization:
    def __init__(self, clip_limit=0.03):
        self.clip_limit = clip_limit

    def __call__(self, img):
        img_np = np.array(img).astype(np.float32) / 255.0
        eq = exposure.equalize_adapthist(img_np, clip_limit=self.clip_limit)
        eq = (eq * 255).astype(np.uint8)
        return Image.fromarray(eq)
