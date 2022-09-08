""" Code for manual annotation of affordances. """
# Standard imports
import sys
import os
from itertools import cycle
from pathlib import Path
import yaml
from PIL import Image
import numpy as np
# Vision imports
import cv2
import torch
from torchvision import transforms as tf
from models.extractor import ViTExtractor
from models.correspondence_functions import (
                                              find_correspondences_v3,
                                              uv_im_to_desc,
                                              uv_desc_to_im
                                            )

with open("./config/default_config.yaml") as f:
    args = yaml.load(f, Loader=yaml.CLoader)

EXAMPLE_TARGET = "./demos/images/demo_parts/example1/hammer_02_00000001_rgb.jpg"
IMDIR = Path("./demos/images/demo_parts/example1")

LABEL_COLORS = [(255,0,0), (0,255,0), (0,0,255), (255,0,255),
                (0,125,125), (125,125,0), (200,255,50),
                (255, 125, 220), (10, 125, 255)]
COLOR_RED = np.array([0, 0, 255]).tolist()
COLOR_GREEN = np.array([0,255,0]).tolist()

def draw_reticle(img, x, y, label_color):
    """ Draws reticle directly on image
    :param img: np.ndarray, [H,W,3]
    :param x: int, x position
    :param y: int, y position
    :param label_color: vec3 of uint8's, color of reticle
    """
    white = (255,255,255)
    cv2.circle(img,(x,y),10,label_color,1)
    cv2.circle(img,(x,y),11,white,1)
    cv2.circle(img,(x,y),12,label_color,1)
    cv2.line(img,(x,y+1),(x,y+3),white,1)
    cv2.line(img,(x+1,y),(x+3,y),white,1)
    cv2.line(img,(x,y-1),(x,y-3),white,1)
    cv2.line(img,(x-1,y),(x-3,y),white,1)

def torch_resize(image, load_size):
    """ resizes torch tensor image """
    return tf.Resize(load_size,
            interpolation=tf.InterpolationMode.LANCZOS)(image)

class HeatmapVisualization(object):
    """ Visualizer for the annotation """
    def __init__(self, config):
        self.config = config
        self._paused = False
        self._reticle_color = COLOR_GREEN
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.extractor = ViTExtractor(config['model_type'], 
                            config['stride'], device=device)
        cv2.namedWindow('target')
        cv2.namedWindow('sim')
        cv2.namedWindow('source')
        self.img1_saved_recticles = []
        self.img2_saved_recticles = []
        self.masks = []
        self.kps   = []
        self.ims = cycle([p for p in IMDIR.iterdir() \
                    if p.suffix.lower() in [".png",".jpeg", ".jpg"]])
        self._get_new_images()

    def show_image(self, pil_image, win_name):
        """ Shows an RGB image on an openCV window
        :param pil_image: PIL.Image
        :param win_name: Name of a OpenCV window
        """
        cv_image = cv2.cvtColor(np.asarray(pil_image),
                                cv2.COLOR_RGB2BGR)
        cv2.imshow(win_name, cv_image)

    def _get_new_images(self):
        """ Gets new images """
        self.img1 = Image.open(next(self.ims).__str__()).convert('RGB')
        self.img2 = Image.open(EXAMPLE_TARGET).convert('RGB')

        self.img1 = torch_resize(self.img1, self.config['load_size'])
        self.img2 = torch_resize(self.img2, self.config['load_size'])

        img_1_w_annot = np.copy(self.img1)
        draw_reticle(img_1_w_annot, 0, 0, self._reticle_color)
        self.show_image(img_1_w_annot, "source")
        img_2_w_annot = np.copy(self.img2)
        draw_reticle(img_2_w_annot, 0, 0, self._reticle_color)
        self.show_image(img_2_w_annot, "target")

        self.mask1 = np.zeros(img_1_w_annot.shape[:2]).astype(np.uint8)
        self.left_draw = False

        self.img1_saved_recticles = []
        self.img2_saved_recticles = []
        self.masks = []
        self.kps   = []

        with torch.no_grad():
            self.sims, self.metainfo = find_correspondences_v3(
                                          self.img1, self.img2,
                                          load_size=self.config['load_size'],
                                          extractor = self.extractor)
            self.sims = self.sims.reshape(self.metainfo['num_patches1'][0],
                                          self.metainfo['num_patches1'][1],
                                          self.metainfo['num_patches2'][0],
                                          self.metainfo['num_patches2'][1])

    def draw_saved_recticles(self, img1, img2):
        """ Draws reticle on np.ndarray image """
        for Kp in self.kps:
            for i,(u,v) in enumerate(Kp):
                color = LABEL_COLORS[min(i,len(LABEL_COLORS)-1)]
                draw_reticle(img1, u, v, color)

        for i,(u,v) in enumerate(self.img1_saved_recticles):
            color = LABEL_COLORS[min(i,len(LABEL_COLORS)-1)]
            draw_reticle(img1, u, v, color)
        for i,(u,v) in enumerate(self.img2_saved_recticles):
            color = LABEL_COLORS[min(i,len(LABEL_COLORS)-1)]
            draw_reticle(img2, u, v, color)

    def save_annotation(self, location = "./affordance_database/temp/"):
        """ Saves the annotations to a folder """
        os.makedirs(location, exist_ok = True)
        self.img1.save(f"{location}/prototype.png")
        cv2.imwrite(f"{location}/annotation.png", self.image_annot[:,:,::-1])
        affordance = {}
        affordance['masks'] = [m  == 255 for m in self.masks ]
        affordance['kps']  = self.kps
        np.save(f"{location}/affordance.npy",affordance, allow_pickle=True)

    def find_best_match(self, event,u,v,flags,param):
        """
        For each network, find the best match in the target image to 
        point highlighted with reticle in the source image. Displays 
        the result
        :return:
        :rtype:
        """
        if self._paused:
            return

        img_1_w_annot = np.copy(self.img1)
        img_2_w_annot = np.copy(self.img2)

        np1 = self.metainfo['num_patches1']
        np2 = self.metainfo['num_patches2']
        patch, stride = self.extractor.p, self.extractor.stride

        u_d,v_d = uv_im_to_desc(u,v, patch, stride)
        u_im,v_im = uv_desc_to_im(u_d,v_d, patch, stride)
        if u_d > np1[1]-1:
            u_d = np1[1]-1
        if v_d > np1[0]-1:
            v_d = np1[0]-1

        sim_d = self.sims[v_d, u_d]
        # Flatten puts full rows first.
        # Thus % gives horizontal and / - vertical
        val, xy = sim_d.flatten().max(dim=-1)
        v_d2,u_d2 = xy / np2[0], xy % np2[1]
        u_im2,v_im2 = uv_desc_to_im(u_d2,v_d2, patch, stride)

        sim_d = (((sim_d + 1)/2)*255)
        sim_d = sim_d.int().cpu().numpy()
        sim_d = sim_d.astype(np.uint8)
        sim_d = cv2.resize(sim_d, (128,128))
        sim_d = np.stack([sim_d]*3, axis=-1)
        cv2.putText(sim_d,"sim=%f"%val,(20,20), 
                  fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                  fontScale = 0.5,
                  color=(int(((val+1)/2)*255), 0, 0),
                  thickness=2)


        if event == cv2.EVENT_LBUTTONDBLCLK:
            self.img1_saved_recticles.append( (u_im,v_im) )
            self.img2_saved_recticles.append( (u_im2,v_im2) )

        if event == cv2.EVENT_LBUTTONDOWN:
            self.left_draw = True
        if event == cv2.EVENT_LBUTTONUP:
            self.left_draw = False
        if event == cv2.EVENT_MBUTTONDOWN:
            self.masks.append(self.mask1)
            self.kps.append(self.img1_saved_recticles)
            self.mask1 = np.zeros(self.mask1.shape).astype(self.mask1.dtype)
            self.img1_saved_recticles = []

        if self.left_draw:
            self.mask1 = cv2.circle(self.mask1, (u,v), 4, (255,255,255), -1)

        label_colors = torch.Tensor(LABEL_COLORS)
        for m_i, mask_i in enumerate(self.masks):
            mask = torch.stack([torch.Tensor(mask_i).cuda()/255.]*3, dim=-1)
            overlay = torch.ones( img_1_w_annot.shape ).cuda() \
                    * label_colors[m_i].unsqueeze(0).unsqueeze(0).cuda()
            img_1_w_annot = torch.Tensor(img_1_w_annot).cuda()
            img_1_w_annot = (  img_1_w_annot * (1-mask) \
                          + (img_1_w_annot*0.6+overlay*0.4) * (mask) ) 
            img_1_w_annot = img_1_w_annot.cpu().numpy().astype(np.uint8)

        mask = torch.stack([torch.Tensor(self.mask1).cuda()/255.]*3, dim=-1)
        overlay = torch.ones( img_1_w_annot.shape ).cuda() * torch.Tensor([[ [255,0,0] ]]).cuda()
        img_1_w_annot = torch.Tensor(img_1_w_annot).cuda()
        img_1_w_annot = (  img_1_w_annot * (1-mask) \
                      + (img_1_w_annot*0.6+overlay*0.4) * (mask) ) 
        img_1_w_annot = img_1_w_annot.cpu().numpy().astype(np.uint8)

        self.draw_saved_recticles(img_1_w_annot, img_2_w_annot)

        self.image_annot = img_1_w_annot.copy()

        draw_reticle(img_1_w_annot, u_im, v_im, self._reticle_color)
        draw_reticle(img_2_w_annot, u_im2, v_im2, self._reticle_color)

        cv2.imshow("sim", sim_d)
        self.show_image(img_1_w_annot, "source")
        self.show_image(img_2_w_annot, "target")
        self.show_image(self.mask1, "source mask")

    def run(self):
        """ Runs code loop """
        cv2.setMouseCallback('source', self.find_best_match)
        print("Useful keys: \n"\
              + " - q: quit\n"\
              + " - n: next image in dir\n"\
              + " - s: save annotation")
        while True:
            k = cv2.waitKey(20) & 0xFF
            if k == 27 or k == ord('q'):
                break
            if k == ord('n'):
                self._get_new_images()
            elif k == ord('s'):
                self.masks.append(self.mask1)
                self.kps.append(self.img1_saved_recticles)
                self.img1_saved_recticles = []
                self.save_annotation()
        cv2.destroyAllWindows()
        cv2.waitKey(1)


if __name__ == "__main__":
    heatmap_vis = HeatmapVisualization(args)
    print ("starting annotator")
    heatmap_vis.run()
