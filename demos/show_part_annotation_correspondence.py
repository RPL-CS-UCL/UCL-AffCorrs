"""
Show part query correspondence in single object images with AffCorrs
"""
# Standard imports
import os
import sys
from PIL import Image
import yaml
import numpy as np
import matplotlib.pyplot as plt
# Vision imports
import torch
import cv2
sys.path.append("..")
from models.correspondence_functions import (overlay_segment, resize)
from models.aff_corrs import AffCorrs_V1

# User-defined constants
SUPPORT_DIR = "../affordance_database/usb/"
TARGET_IMAGE_PATH = "./images/demo_affordance/eth.jpg"

# Other constants
PATH_TO_CONFIG  = "../config/default_config.yaml"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
COLORS = [[255,0,0],[255,255,0],[255,0,255],
          [0,255,0],[0,0,255],[0,255,255]]

# Load arguments
with open(PATH_TO_CONFIG) as f:
    args = yaml.load(f, Loader=yaml.CLoader)
args['low_res_saliency_maps'] = False
args['load_size'] = 256

# Helper functions
def load_rgb(path):
    """ Loading RGB image with OpenCV
    : param path: string, image path name. Must point to a file.
    """
    assert os.path.isfile(path), f"Path {path} doesn't exist"
    return cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB)

def viz_correspondence(im_a, im_b, parts_a, parts_b):
    """ Visualizes the correspondences
    : param im_a: np.ndarray, RGB image a
    : param im_b: np.ndarray, RGB image b
    : param parts_a: List[np.ndarray], list of part masks in a
    : param parts_b: List[np.ndarray], list of part masks in b
    """
    quer_img = im_a.astype(np.uint8)
    corr_img = im_b.astype(np.uint8)
    for i, part_i in enumerate(parts_a):
        quer_img = overlay_segment(quer_img, part_i,
                                  COLORS[i], alpha=0.3)
        part_out_i = resize(parts_b[i],corr_img.shape[:2]) > 0
        corr_img = overlay_segment(corr_img, part_out_i,
                                  COLORS[i], alpha=0.3)

    _fig, ax = plt.subplots(1,2)
    ax[0].imshow(quer_img)
    ax[1].imshow(corr_img)
    plt.show()

if __name__ == "__main__":
    # The models are ran with no_grad since they are 
    # unsupervised. This preserves GPU memory
    with torch.no_grad():
        model = AffCorrs_V1(args)

        # Prepare inputs
        img1_path = f"{SUPPORT_DIR}/prototype.png"
        aff1_path = f"{SUPPORT_DIR}/affordance.npy"
        rgb_a = load_rgb(img1_path)
        parts = np.load(aff1_path, allow_pickle=True).item()['masks']
        affordances = [None for _ in parts]
        rgb_b = load_rgb(TARGET_IMAGE_PATH)

        ## Produce correspondence
        model.set_source(Image.fromarray(rgb_a), parts, affordances)
        model.generate_source_clusters()
        model.set_target(Image.fromarray(rgb_b))
        model.generate_target_clusters()
        parts_out, aff_out = model.find_correspondences()

        ## Display correspondence
        viz_correspondence(rgb_a, rgb_b, parts, parts_out)
