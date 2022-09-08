import torch
import torchvision.transforms
from torchvision import transforms

import numpy as np
import cv2
import matplotlib.pyplot as plt

from PIL import Image
import sys, os, glob, yaml
sys.path.append("..")

from models.extractor import ViTExtractor
from models.correspondence_functions import *

with open("../config/default_config.yaml") as f:
  args = yaml.load(f, Loader=yaml.CLoader)

img1_path = "./images/demo_points/001.jpg"
img2_path = "./images/demo_points/002.jpg"

# Color definitions
white = (255,255,255)
black = (0,0,0)
label_colors = [(255,0,0), (0,255,0), (0,0,255), (255,0,255), 
                (0,125,125), (125,125,0), (200,255,50), 
                (255, 125, 220), (10, 125, 255)]
COLOR_RED   = np.array([0, 0, 255]).tolist()
COLOR_GREEN = np.array([0,255,0]).tolist()

def draw_reticle(img, x, y, label_color):
    cv2.circle(img,(x,y),10,label_color,1)
    cv2.circle(img,(x,y),11,white,1)
    cv2.circle(img,(x,y),12,label_color,1)
    cv2.line(img,(x,y+1),(x,y+3),white,1)
    cv2.line(img,(x+1,y),(x+3,y),white,1)
    cv2.line(img,(x,y-1),(x,y-3),white,1)
    cv2.line(img,(x-1,y),(x-3,y),white,1)

class HeatmapVisualization(object):
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
    return
  def show_image(self, pil_image, win_name):
    cv_image = cv2.cvtColor(np.asarray(pil_image), 
                           cv2.COLOR_RGB2BGR)
    cv2.imshow(win_name, cv_image)
    return
  def _resize(self,image, load_size):
    return transforms.Resize(load_size, 
            interpolation=transforms.InterpolationMode.LANCZOS)(image)
    
  def _get_new_images(self):
    self.img1 = Image.open(img1_path).convert('RGB')
    self.img2 = Image.open(img2_path).convert('RGB')

    self.img1 = self._resize(self.img1, self.config['load_size'])
    self.img2 = self._resize(self.img2, self.config['load_size'])

    img_1_with_reticle = np.copy(self.img1)
    draw_reticle(img_1_with_reticle, 0, 0, self._reticle_color)
    self.show_image(img_1_with_reticle, "source")
    img_2_with_reticle = np.copy(self.img2)
    draw_reticle(img_2_with_reticle, 0, 0, self._reticle_color)
    self.show_image(img_2_with_reticle, "target")
    
    with torch.no_grad():
      self.sims, self.metainfo = find_correspondences_v3(
                                    self.img1, self.img2, 
                                    load_size=self.config['load_size'],
                                    extractor = self.extractor)
      self.sims = self.sims.reshape(self.metainfo['num_patches1'][0],
                                    self.metainfo['num_patches1'][1],
                                    self.metainfo['num_patches2'][0],
                                    self.metainfo['num_patches2'][1])
    return

  def draw_saved_recticles(self, img1, img2, reticle_color):
      for i,(u,v) in enumerate(self.img1_saved_recticles):
          color = label_colors[min(i,len(label_colors)-1)]
          draw_reticle(img1, u, v, color)
      for i,(u,v) in enumerate(self.img2_saved_recticles):
          color = label_colors[min(i,len(label_colors)-1)]
          draw_reticle(img2, u, v, color)
      return
  
  def find_best_match(self, event,u,v,flags,param):
    """
    For each network, find the best match in the target image to 
    point highlighted with reticle in the source image. Displays 
    the result
    :return:
    :rtype:
    """

    if self._paused: return

    img_1_with_reticle = np.copy(self.img1)
    img_2_with_reticle = np.copy(self.img2)
    
    np1 = self.metainfo['num_patches1']
    np2 = self.metainfo['num_patches2']
    patch, stride = self.extractor.p, self.extractor.stride

    u_d,v_d = uv_im_to_desc(u,v, patch, stride)
    u_im,v_im = uv_desc_to_im(u_d,v_d, patch, stride )
    if u_d >= self.sims.shape[1]: u_d = self.sims.shape[1]-1
    if v_d >= self.sims.shape[0]: v_d = self.sims.shape[0]-1
    
    sim_d = self.sims[v_d, u_d]
    # Flatten puts full rows first. 
    # Thus % gives horizontal and / - vertical
    val, xy = sim_d.flatten().max(dim=-1)
    v_d2,u_d2 = xy / np2[0], xy % np2[1]
    u_im2,v_im2 = uv_desc_to_im(u_d2,v_d2, patch, stride )

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


    draw_reticle(img_1_with_reticle, u_im, v_im, self._reticle_color)
    draw_reticle(img_2_with_reticle, u_im2, v_im2, self._reticle_color)

    if event == cv2.EVENT_LBUTTONDBLCLK:
      self.img1_saved_recticles.append( (u_im,v_im) )
      self.img2_saved_recticles.append( (u_im2,v_im2) )

    self.draw_saved_recticles(img_1_with_reticle, img_2_with_reticle, self._reticle_color)

    cv2.imshow("sim", sim_d)
    self.show_image(img_1_with_reticle, "source")
    self.show_image(img_2_with_reticle, "target")
    
  def run(self):
    self._get_new_images()
    cv2.setMouseCallback('source', self.find_best_match)

    #self._get_new_images()

    while True:
      k = cv2.waitKey(20) & 0xFF
      if k == 27 or k == ord('q'):
          break
      elif k == ord('n'):
          self._get_new_images()
    cv2.destroyAllWindows()
    cv2.waitKey(1)


if __name__ == "__main__":
  heatmap_vis = HeatmapVisualization(args)
  print ("starting heatmap vis")
  heatmap_vis.run()