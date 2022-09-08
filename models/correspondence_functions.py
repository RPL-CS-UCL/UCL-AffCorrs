"""
Code for correspondence functions and utils related to them.

Some correspondence functions included in this code are from
https://github.com/ShirAmir/dino-vit-features
@article{amir2021deep,
  author    = {Shir Amir and Yossi Gandelsman and Shai Bagon and Tali Dekel},
  title     = {Deep ViT Features as Dense Visual Descriptors},
  journal   = {arXiv preprint arXiv:2112.05814},
  year      = {2021}
}
"""

# Standard imports
from pathlib import Path
from PIL import Image
from typing import List, Tuple
import time
import numpy as np

# Vision imports
#import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from skimage.transform import resize
from skimage import img_as_bool
import faiss
import cv2
import torch
import torchvision.transforms as tf
from pytorch_metric_learning import distances

# Custom imports
from .extractor import ViTExtractor
from .crf import *
from .clustering import *
from .augmentation_functions import *

### Helper functions

def get_normal_descriptors(descriptors_list):
    """ Returns normalized descriptor array
      :param descriptor list: list of descriptors [1,1,N,D]
      :return: normalized descriptor array [N,D]
    """
    all_descriptors = np.ascontiguousarray(np.concatenate(descriptors_list, axis=2)[0, 0])
    normalized_all_descriptors = all_descriptors.astype(np.float32)
    faiss.normalize_L2(normalized_all_descriptors)  # in-place operation
    return normalized_all_descriptors



def get_saliency_maps(extractor, image_batch, image_path, curr_num_patches, 
                      curr_load_size, low_res_saliency_maps, save_dir, device):
    """ Returns a saliency map extracted from DINO ViT.
      :param extractor: model network
      :param image_batch: image input [B,C,H,W]
      :param image_path: image path used to extract features if using low resolution
                        and to save images if saving
      :param curr_num_patches: Number of patches in each dimension
      :param curr_load_size: loading size
      :param low_res_saliency_maps: bool, flag for low resolution
      :param save_dir: os.Path, path to saving directory
      :param device: str, 'cpu' or 'gpu' flag
      :return: saliency map, [H,W]
    """
    if low_res_saliency_maps:
        if load_size is not None:
            low_res_load_size = (curr_load_size[0] // 2, 
                                 curr_load_size[1] // 2)
        else:
            low_res_load_size = curr_load_size
        image_batch, _ = extractor.preprocess(image_path, low_res_load_size).to(device)
    saliency_map = extractor.extract_saliency_maps(image_batch.to(device)).cpu().numpy()
    curr_sal_num_patches, curr_sal_load_size = extractor.num_patches, extractor.load_size
    if low_res_saliency_maps:
        reshape_op = tf.Resize(curr_num_patches, tf.InterpolationMode.NEAREST)
        saliency_map_reshaped = saliency_map.reshape(curr_sal_num_patches)
        saliency_map_reshaped = reshape_op(Image.fromarray(saliency_map_reshaped))
        saliency_map = np.array(saliency_map_reshaped)
    saliency_map = saliency_map.flatten()

    # save saliency maps and resized images if needed (not for augmentations)
    if save_dir is not None and not '_aug_' in Path(image_path).stem:
        fig, ax = plt.subplots()
        ax.axis('off')
        ax.imshow(saliency_map.reshape(curr_num_patches), 
                  vmin=0, vmax=1, cmap='jet')
        savepath = save_dir / f'{Path(image_path).stem}_saliency_map.png'
        fig.savefig(savepath, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    return saliency_map


def get_saliency_maps_V2(saliency_extractor, image_batch, 
                      image, num_patches, 
                      load_size, low_res_saliency_maps, device, thresh):
  if low_res_saliency_maps and load_size is not None:
    load_size = (load_size[0] // 2, load_size[1] // 2)
    image_batch, _ = saliency_extractor.preprocess_image(image, load_size)
  saliency_map = saliency_extractor.extract_saliency_maps(image_batch.to(device)).cpu().numpy()
  sal_num_patches = saliency_extractor.num_patches
  
  reshape_op = tf.Resize(num_patches, tf.InterpolationMode.NEAREST)
  saliency_map = np.array(reshape_op(Image.fromarray(saliency_map.reshape(sal_num_patches))))
  saliency_map = saliency_map > thresh
  return saliency_map

def rescale_pts(u,v, size_in, size_out):
    """ Rescale Keypoints based on reshaped sizes """
    return (np.array(u*size_out[1]*1.0/size_in[1],dtype=np.int32),
            np.array(v*size_out[0]*1.0/size_in[0],dtype=np.int32) )

def uv_desc_to_im(u,v, patch, stride):
    """ Translates UV locations from descriptor to image size """
    u_im = (np.array(u, dtype=np.uint16)-1) * stride[1] \
        + stride[1] + patch//2
    v_im = (np.array(v, dtype=np.uint16)-1) * stride[0] \
        + stride[0] + patch//2
    return u_im, v_im

def to_int(x):
    """ Returns an integer type object """
    if torch.is_tensor(x): return x.long()
    if isinstance(x, np.ndarray): return x.astype(np.int32)
    return int(x)

def uv_im_to_desc(u,v, patch, stride):
    """ Translates UV locations from image to descriptor size """
    u_d = to_int((u -  patch//2)//stride[1] +1)
    v_d = to_int((v -  patch//2)//stride[0] +1)
    return u_d, v_d

def get_load_shape(im, load_size):
    """ Computes the load shape given a minimum size """
    shape = np.asarray(im).shape[:2]
    ratio = load_size/min(shape)
    return (int(shape[0]*ratio),int(shape[1]*ratio))

def image_resize(im, s=None):
    """ Returns a resized image im with side(s) s
    :param im: array-like
    :param s: int or list(2)
    """
    if s is None:
        return im
    if isinstance(s, int):
        im = im.resize((s,s), Image.ANTIALIAS)
    if isinstance(s, List):
        if len(s)!=2: raise ValueError()
        if None in s:
            if s[0] is None: raise ValueError()
            shape = np.asarray(im).shape[:2]
            ratio = s[0]/min(shape)
            im = im.resize( tuple(reversed(
                              get_load_shape(im, s[0]))),
                          Image.ANTIALIAS)
        else:
            im = im.resize(s, Image.ANTIALIAS)
    return im

def map_descriptors_to_clusters(desc, fg_centroids, bg_centroids):
    """ 
    Takes descriptors and maps them to the closest centroids
    Background centroids are accumulated into one.
    :param desc        : [N0,D], torch.tensor or np.ndarray
    :param fg_centroids: [N1,D], torch.tensor or np.ndarray
    :param bg_centroids: [N2,D], torch.tensor or np.ndarray
    :return: [N0,1], torch.Tensor or mapping where values 
              correspond to clusters. Last value is background
    """
    from pytorch_metric_learning import distances
    d_fn = distances.CosineSimilarity()

    D_fg = d_fn(torch.Tensor(desc), torch.Tensor(fg_centroids))
    D_bg = d_fn(torch.Tensor(desc), torch.Tensor(bg_centroids)).max(dim=-1,keepdim=True).values
    C = torch.cat([D_fg,D_bg],dim=-1).argmax(dim=-1)
    return C

def map_values_to_segments(seg_im, values):
    """ Returns an image where each pixel is assigned the
    corresponding cluster score.
    """    
    P = torch.zeros(seg_im.shape)
    for i,val_i in enumerate(values):
        P [seg_im==i] = val_i
    return P

def get_best_corr_at_loc(sims_AB, u_d_A, v_d_A, np_B):
    """ Returns the best correspondence at given UV locations 
    :param sims_AB: torch.Tensor, similarity matrix from A to B, 
                    with shape [Ha,Wa,Hb,Wb]
    :param u_d_A: vector of U values in a
    :param v_d_A: vector of V values in a
    :param np_B: List of number of patches in each dimension of B
    """
    # Select row/col to get sim scores in second image
    sim_AB = sims_AB[to_int(v_d_A),
                    to_int(u_d_A)]
    
    # Flatten puts full rows first. 
    # Thus % gives horizontal and / - vertical
    val, xy = sim_AB.flatten(1).max(dim=-1)
    v_d_B, u_d_B = torch.floor(xy / np_B[0]) , (xy % np_B[1])
    return u_d_B, v_d_B

def get_part_correspondence(query_mask, p_sim, C, tgt_image,
                            curr_load_size, load_shape):
  """ Returns the final part correspondence after passing through the CRF """
  # Transform into FB/BG probability where FG is selected area.
  # Sum HW into FG and BG (2,K)
  mask_ = resize(img_as_bool(query_mask), p_sim.shape[:2]).astype(np.bool)
  fg = p_sim[mask_].sum(0) 

  # Remap to Image 2 (2, HW)
  # no bg is passed, so it defaults to 0
  P = map_values_to_segments(C, fg)

  # Set as Unary to CRF
  # Lower energy = bigger distance
  # Higher probability = lower distance.
  P = torch.stack([P,1-P],dim=-1).numpy()
  P = P.reshape(*C.shape,-1)
  
  final = CRF( tgt_image, P, C.shape, None, curr_load_size )
  final = final.reshape(*load_shape)
  return final

def overlay_segment(img, seg, color, alpha=0.4):
    """ Returns the original image with a semi-transparent overlay 
    over a segment
    :param img: np.ndarray, RGB image
    :param seg: np.ndarray, a single segment
    :param color: np.ndarray, vec3 color
    :return: overlaid image.
    """
    seg = seg.astype(np.bool)
    img[seg] = (img[seg]*alpha).astype(np.uint8) \
                + (1-alpha)*np.asarray(color).astype(np.uint8)
    return img          

def overlay_all_segments(image_pil, segm_out, colors):
  """ Returns the original image with a semi-transparent overlay
  over each segment
  :param image_pil: PIL.Image
  :param segm_out: List of segment masks
  :param colors: List of vec3 colors
  """
  img_out = np.asarray(image_pil).astype(np.uint8)
  for i, seg_i in enumerate(segm_out):
      img_out = overlay_segment(img_out, seg_i, colors[i])
  return img_out



def chunk_cosine_sim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """ Computes cosine similarity between all possible pairs in two sets of vectors.
    Operates on chunks so no large amount of GPU RAM is required.
    :param x: an tensor of descriptors of shape Bx1x(t_x)xd' where d' is the dimensionality of the descriptors and t_x
    is the number of tokens in x.
    :param y: a tensor of descriptors of shape Bx1x(t_y)xd' where d' is the dimensionality of the descriptors and t_y
    is the number of tokens in y.
    :return: cosine similarity between all descriptors in x and all descriptors in y. Has shape of Bx1x(t_x)x(t_y) """
    result_list = []
    num_token_x = x.shape[2]
    for token_idx in range(num_token_x):
        token = x[:, :, token_idx, :].unsqueeze(dim=2)  # Bx1x1xd'
        result_list.append(torch.nn.CosineSimilarity(dim=3)(token, y))  # Bx1xt
    return torch.stack(result_list, dim=2)  # Bx1x(t_x)x(t_y)



################### Drawing functions. ###################

def draw_part_cosegmentation(num_parts: int, segmentation_parts: List[np.ndarray], 
                            pil_images: List[Image.Image]):
    """
    Visualizes part cosegmentation results on chessboard background.
    :param num_parts: number of object parts in all part cosegmentations.
    :param segmentation_parts: list of binary segmentation masks
    :param pil_images: list of corresponding images.
    :return: list of figures with fg segment of each image and chessboard bg.
    """
    figures = []
    for parts_seg, pil_image in zip(segmentation_parts, pil_images):
        current_mask = ~np.isnan(parts_seg)  # np.isin(segmentation, segment_indexes)
        stacked_mask = np.dstack(3 * [current_mask])
        masked_image = np.array(pil_image)
        masked_image[~stacked_mask] = 0
        masked_image_transparent = np.concatenate((masked_image, 255. * current_mask.astype(np.uint8)[..., None]),
                                                  axis=-1)
        # create chessboard bg
        checkerboard_bg = np.zeros(masked_image.shape[:2])
        checkerboard_edge = 10
        checkerboard_bg[[x // checkerboard_edge % 2 == 0 for x in range(checkerboard_bg.shape[0])], :] = 1
        checkerboard_bg[:, [x // checkerboard_edge % 2 == 1 for x in range(checkerboard_bg.shape[1])]] = \
            1 - checkerboard_bg[:, [x // checkerboard_edge % 2 == 1 for x in range(checkerboard_bg.shape[1])]]
        checkerboard_bg[checkerboard_bg == 0] = 0.75
        checkerboard_bg = 255. * checkerboard_bg

        # show
        fig, ax = plt.subplots()
        ax.axis('off')
        color_list = ["red", "yellow", "blue", "lime", "darkviolet", "magenta", "cyan", "brown", "yellow"]
        cmap = 'jet' if num_parts > 10 else ListedColormap(color_list[:num_parts])
        ax.imshow(checkerboard_bg, cmap='gray', vmin=0, vmax=255)
        ax.imshow(masked_image_transparent.astype(np.int32), vmin=0, vmax=255)
        ax.imshow(parts_seg, cmap=cmap, vmin=0, vmax=num_parts - 1, alpha=0.5, interpolation='nearest')
        figures.append(fig)
    return figures


def draw_correspondences(points1: List[Tuple[float, float]], points2: List[Tuple[float, float]],
                         image1: Image.Image, image2: Image.Image):
    """
    draw point correspondences on images.
    :param points1: a list of (y, x) coordinates of image1, corresponding to points2.
    :param points2: a list of (y, x) coordinates of image2, corresponding to points1.
    :param image1: a PIL image.
    :param image2: a PIL image.
    :return: two figures of images with marked points.
    """
    assert len(points1) == len(points2), f"points lengths are incompatible: {len(points1)} != {len(points2)}."
    num_points = len(points1)
    fig1, ax1 = plt.subplots()
    ax1.axis('off')
    fig2, ax2 = plt.subplots()
    ax2.axis('off')
    ax1.imshow(image1)
    ax2.imshow(image2)
    if num_points > 15:
        cmap = plt.get_cmap('tab10')
    else:
        cmap = ListedColormap(["red", "yellow", "blue", "lime", "magenta", "indigo", "orange", "cyan", "darkgreen",
                               "maroon", "black", "white", "chocolate", "gray", "blueviolet"])
    colors = np.array([cmap(x) for x in range(num_points)])
    radius1, radius2 = 8, 1
    for point1, point2, color in zip(points1, points2, colors):
        y1, x1 = point1
        circ1_1 = plt.Circle((x1, y1), radius1, facecolor=color, edgecolor='white', alpha=0.5)
        circ1_2 = plt.Circle((x1, y1), radius2, facecolor=color, edgecolor='white')
        ax1.add_patch(circ1_1)
        ax1.add_patch(circ1_2)
        y2, x2 = point2
        circ2_1 = plt.Circle((x2, y2), radius1, facecolor=color, edgecolor='white', alpha=0.5)
        circ2_2 = plt.Circle((x2, y2), radius2, facecolor=color, edgecolor='white')
        ax2.add_patch(circ2_1)
        ax2.add_patch(circ2_2)
    return fig1, fig2





#####################################################################
################### Old correspondence functions. ###################
#####################################################################
# Kept for compatibility, but not used in final version of AffCorrs #

def find_correspondences(image_path1: str, image_path2: str, num_pairs: int = 10, load_size: int = 224, layer: int = 9,
                         facet: str = 'key', bin: bool = True, thresh: float = 0.05, model_type: str = 'dino_vits8',
                         stride: int = 4) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]],
                                                                              Image.Image, Image.Image]:
    """
    finding point correspondences between two images.
    :param image_path1: path to the first image.
    :param image_path2: path to the second image.
    :param num_pairs: number of outputted corresponding pairs.
    :param load_size: size of the smaller edge of loaded images. If None, does not resize.
    :param layer: layer to extract descriptors from.
    :param facet: facet to extract descriptors from.
    :param bin: if True use a log-binning descriptor.
    :param thresh: threshold of saliency maps to distinguish fg and bg.
    :param model_type: type of model to extract descriptors from.
    :param stride: stride of the model.
    :return: list of points from image_path1, list of corresponding points from image_path2, the processed pil image of
    image_path1, and the processed pil image of image_path2.
    """
    # extracting descriptors for each image
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    extractor = ViTExtractor(model_type, stride, device=device)
    image1_batch, image1_pil = extractor.preprocess(image_path1, load_size)
    descriptors1 = extractor.extract_descriptors(image1_batch.to(device), layer, facet, bin)
    num_patches1, load_size1 = extractor.num_patches, extractor.load_size
    image2_batch, image2_pil = extractor.preprocess(image_path2, load_size)
    descriptors2 = extractor.extract_descriptors(image2_batch.to(device), layer, facet, bin)
    num_patches2, load_size2 = extractor.num_patches, extractor.load_size

    # extracting saliency maps for each image
    saliency_map1 = extractor.extract_saliency_maps(image1_batch.to(device))[0]
    saliency_map2 = extractor.extract_saliency_maps(image2_batch.to(device))[0]
    # save saliency maps and resized images if needed (not for augmentations)
    
    # threshold saliency maps to get fg / bg masks
    fg_mask1 = saliency_map1 > thresh
    fg_mask2 = saliency_map2 > thresh

    # calculate similarity between image1 and image2 descriptors
    similarities = chunk_cosine_sim(descriptors1, descriptors2)
    
    # calculate best buddies
    image_idxs = torch.arange(num_patches1[0] * num_patches1[1], device=device)
    sim_1, nn_1 = torch.max(similarities, dim=-1)  # nn_1 - indices of block2 closest to block1
    sim_2, nn_2 = torch.max(similarities, dim=-2)  # nn_2 - indices of block1 closest to block2
    sim_1, nn_1 = sim_1[0, 0], nn_1[0, 0]
    sim_2, nn_2 = sim_2[0, 0], nn_2[0, 0]
    bbs_mask = nn_2[nn_1] == image_idxs

    # remove best buddies where at least one descriptor is marked bg by saliency mask.
    fg_mask2_new_coors = nn_2[fg_mask2] # matches in 1
    fg_mask2_mask_new_coors = torch.zeros(num_patches1[0] * num_patches1[1], dtype=torch.bool, device=device)
    fg_mask2_mask_new_coors[fg_mask2_new_coors] = True
    bbs_mask = torch.bitwise_and(bbs_mask, fg_mask1)
    bbs_mask = torch.bitwise_and(bbs_mask, fg_mask2_mask_new_coors)
    
    # applying k-means to extract k high quality well distributed correspondence pairs
    bb_descs1 = descriptors1[0, 0, bbs_mask, :].cpu().numpy()
    bb_descs2 = descriptors2[0, 0, nn_1[bbs_mask], :].cpu().numpy()
    # apply k-means on a concatenation of a pairs descriptors.
    all_keys_together = np.concatenate((bb_descs1, bb_descs2), axis=1)
    n_clusters = min(num_pairs, len(all_keys_together))  # if not enough pairs, show all found pairs.
    length = np.sqrt((all_keys_together ** 2).sum(axis=1))[:, None]
    normalized = all_keys_together / length
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(normalized)
    bb_topk_sims = np.full((n_clusters), -np.inf)
    bb_indices_to_show = np.full((n_clusters), -np.inf)

    # rank pairs by their mean saliency value
    bb_cls_attn1 = saliency_map1[bbs_mask]
    bb_cls_attn2 = saliency_map2[nn_1[bbs_mask]]
    bb_cls_attn = (bb_cls_attn1 + bb_cls_attn2) / 2
    ranks = bb_cls_attn

    for k in range(n_clusters):
        for i, (label, rank) in enumerate(zip(kmeans.labels_, ranks)):
            if rank > bb_topk_sims[label]:
                bb_topk_sims[label] = rank
                bb_indices_to_show[label] = i

    # get coordinates to show
    indices_to_show = torch.nonzero(bbs_mask, as_tuple=False).squeeze(dim=1)[
        bb_indices_to_show]  # close bbs
    img1_indices_to_show = torch.arange(num_patches1[0] * num_patches1[1], device=device)[indices_to_show]
    img2_indices_to_show = nn_1[indices_to_show]
    # coordinates in descriptor map's dimensions
    img1_y_to_show = (img1_indices_to_show / num_patches1[1]).cpu().numpy()
    img1_x_to_show = (img1_indices_to_show % num_patches1[1]).cpu().numpy()
    img2_y_to_show = (img2_indices_to_show / num_patches2[1]).cpu().numpy()
    img2_x_to_show = (img2_indices_to_show % num_patches2[1]).cpu().numpy()
    points1, points2 = [], []
    for y1, x1, y2, x2 in zip(img1_y_to_show, img1_x_to_show, img2_y_to_show, img2_x_to_show):
        x1_show = (int(x1) - 1) * extractor.stride[1] + extractor.stride[1] + extractor.p // 2
        y1_show = (int(y1) - 1) * extractor.stride[0] + extractor.stride[0] + extractor.p // 2
        x2_show = (int(x2) - 1) * extractor.stride[1] + extractor.stride[1] + extractor.p // 2
        y2_show = (int(y2) - 1) * extractor.stride[0] + extractor.stride[0] + extractor.p // 2
        points1.append((y1_show, x1_show))
        points2.append((y2_show, x2_show))
    return points1, points2, image1_pil, image2_pil

def find_correspondences_v2(image_path1: str, image_path2: str, num_pairs: int = 10, load_size: int = 224, layer: int = 9,
                         facet: str = 'key', bin: bool = True, thresh: float = 0.05, model_type: str = 'dino_vits8',
                         stride: int = 4) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]],
                                                                              Image.Image, Image.Image]:
    """
    finding point correspondences between two images.
    :param image_path1: path to the first image.
    :param image_path2: path to the second image.
    :param num_pairs: number of outputted corresponding pairs.
    :param load_size: size of the smaller edge of loaded images. If None, does not resize.
    :param layer: layer to extract descriptors from.
    :param facet: facet to extract descriptors from.
    :param bin: if True use a log-binning descriptor.
    :param thresh: threshold of saliency maps to distinguish fg and bg.
    :param model_type: type of model to extract descriptors from.
    :param stride: stride of the model.
    :return: list of points from image_path1, list of corresponding points from image_path2, the processed pil image of
    image_path1, and the processed pil image of image_path2.
    """
    # extracting descriptors for each image
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    extractor = ViTExtractor(model_type, stride, device=device)
    image1_batch, image1_pil = extractor.preprocess(image_path1, load_size)
    descriptors1 = extractor.extract_descriptors(image1_batch.to(device), layer, facet, bin)
    num_patches1, load_size1 = extractor.num_patches, extractor.load_size
    image2_batch, image2_pil = extractor.preprocess(image_path2, load_size)
    descriptors2 = extractor.extract_descriptors(image2_batch.to(device), layer, facet, bin)
    num_patches2, load_size2 = extractor.num_patches, extractor.load_size

    # extracting saliency maps for each image
    saliency_map1 = extractor.extract_saliency_maps(image1_batch.to(device))[0]
    saliency_map2 = extractor.extract_saliency_maps(image2_batch.to(device))[0]

    save_dir = Path("./output_c")
    if save_dir is not None:
      fig, ax = plt.subplots()
      ax.axis('off')
      ax.imshow(saliency_map1.cpu().detach().numpy().reshape(num_patches1), vmin=0, vmax=1, cmap='jet')
      fig.savefig(save_dir / f'{Path(image_path1).stem}_saliency_map1.png', bbox_inches='tight', pad_inches=0)
      plt.close(fig)
    # threshold saliency maps to get fg / bg masks
    fg_mask1 = saliency_map1 > thresh
    fg_mask2 = saliency_map2 > thresh

    if save_dir is not None:
        fig, ax = plt.subplots()
        ax.axis('off')
        ax.imshow(fg_mask1.cpu().detach().numpy().reshape(num_patches1), 
                  vmin=0, vmax=1, cmap='jet')
        fig.savefig(save_dir / f'{Path(image_path1).stem}_saliency_map1fg.png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    # calculate similarity between image1 and image2 descriptors
    similarities = chunk_cosine_sim(descriptors1, descriptors2)

    # calculate best buddies
    image_idxs = torch.arange(num_patches1[0] * num_patches1[1], device=device)
    sim_1, nn_1 = torch.max(similarities, dim=-1)  # nn_1 - indices of block2 closest to block1
    sim_2, nn_2 = torch.max(similarities, dim=-2)  # nn_2 - indices of block1 closest to block2
    sim_1, nn_1 = sim_1[0, 0], nn_1[0, 0]
    sim_2, nn_2 = sim_2[0, 0], nn_2[0, 0]
    # best buddies only where the best match of the best match of (1) is (1)
    # I.e cyclic
    bbs_mask = nn_2[nn_1] == image_idxs

    # remove best buddies where at least one descriptor is marked bg by saliency mask.
    fg_mask2_new_coors = nn_2[fg_mask2] # matches in 1
    fg_mask2_mask_new_coors = torch.zeros(num_patches1[0] * num_patches1[1],
                                          dtype=torch.bool, device=device)
    fg_mask2_mask_new_coors[fg_mask2_new_coors] = True
    bbs_mask = torch.bitwise_and(bbs_mask, fg_mask1)

    # applying k-means to extract k high quality well distributed correspondence pairs
    bb_descs1 = descriptors1[0, 0, bbs_mask, :].cpu().numpy()
    bb_descs2 = descriptors2[0, 0, nn_1[bbs_mask], :].cpu().numpy()

    # get coordinates to show
    indices_to_show = torch.nonzero(bbs_mask, as_tuple=False).squeeze(dim=1)[:]  # close bbs
    img1_indices_to_show = torch.arange(num_patches1[0] * num_patches1[1], device=device)[indices_to_show]
    img2_indices_to_show = nn_1[indices_to_show]
    # coordinates in descriptor map's dimensions
    img1_y_to_show = (img1_indices_to_show / num_patches1[1]).cpu().numpy()
    img1_x_to_show = (img1_indices_to_show % num_patches1[1]).cpu().numpy()
    img2_y_to_show = (img2_indices_to_show / num_patches2[1]).cpu().numpy()
    img2_x_to_show = (img2_indices_to_show % num_patches2[1]).cpu().numpy()
    points1, points2 = [], []
    for y1, x1, y2, x2 in zip(img1_y_to_show, img1_x_to_show, img2_y_to_show, img2_x_to_show):
        x1_show = (int(x1) - 1) * extractor.stride[1] + extractor.stride[1] + extractor.p // 2
        y1_show = (int(y1) - 1) * extractor.stride[0] + extractor.stride[0] + extractor.p // 2
        x2_show = (int(x2) - 1) * extractor.stride[1] + extractor.stride[1] + extractor.p // 2
        y2_show = (int(y2) - 1) * extractor.stride[0] + extractor.stride[0] + extractor.p // 2
        points1.append((y1_show, x1_show))
        points2.append((y2_show, x2_show))
    return points1, points2, image1_pil, image2_pil

def find_correspondences_v3(image1: Image.Image, image2: Image.Image,
      load_size: int = 224, extractor=None,
      layer: int = 9, facet: str = 'key',
      bin: bool = True, model_type: str = 'dino_vits8',
      stride: int = 4, load_size2 = None):
    """
    finding point correspondences between two images.
    V3 Changes:
      - Image inputs instead of image paths
      - Optional mask inputs instead of computing saliency

    :param image_path1: path to the first image.
    :param image_path2: path to the second image.
    :param num_pairs: number of outputted corresponding pairs.
    :param load_size: size of the smaller edge of loaded images. If None, does not resize.
    :param layer: layer to extract descriptors from.
    :param facet: facet to extract descriptors from.
    :param bin: if True use a log-binning descriptor.
    :param thresh: threshold of saliency maps to distinguish fg and bg.
    :param model_type: type of model to extract descriptors from.
    :param stride: stride of the model.
    :return: list of points from image_path1, list of corresponding points
             from image_path2, the processed pil image of image_path1, and
             the processed pil image of image_path2.
    """

    if load_size2 is None:
        load_size2 = load_size
    # extracting descriptors for each image
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if extractor is None:
        extractor = ViTExtractor(model_type, stride, device=device)
    image1_batch, image1_pil = extractor.preprocess_image(image1, load_size)
    descriptors1 = extractor.extract_descriptors(image1_batch.to(device), layer, facet, bin)
    num_patches1, load_size1 = extractor.num_patches, extractor.load_size
    image2_batch, image2_pil = extractor.preprocess_image(image2, load_size2)
    descriptors2 = extractor.extract_descriptors(image2_batch.to(device), layer, facet, bin)
    num_patches2, load_size2 = extractor.num_patches, extractor.load_size

    # calculate similarity between image1 and image2 descriptors
    similarities = chunk_cosine_sim(descriptors1.detach(), descriptors2.detach()).cpu()
    metainfo = dict(num_patches1=num_patches1, num_patches2=num_patches2)
    return similarities, metainfo


def find_part_cosegmentation(image_paths: List[str], elbow: float = 0.975,
                             load_size: int = 224, layer: int = 11,
                             facet: str = 'key', bin: bool = False,
                             thresh: float = 0.065, model_type: str = 'dino_vits8',
                             stride: int = 4, votes_percentage: int = 75,
                             sample_interval: int = 100, low_res_saliency_maps: bool = True,
                             num_parts: int = 4, num_crop_augmentations: int = 0,
                             three_stages: bool = False, elbow_second_stage: float = 0.94,
                             save_dir: str = None, extractor=None
                             ) -> Tuple[List[Image.Image], List[Image.Image]]:
    """
    finding cosegmentation of a set of images.
    This is the original co-segmentation function from Amir et al. 

    :param image_paths: a list of paths of all the images.
    :param elbow: elbow coefficient to set number of clusters.
    :param load_size: size of the smaller edge of loaded images. If None, does not resize.
    :param layer: layer to extract descriptors from.
    :param facet: facet to extract descriptors from.
    :param bin: if True use a log-binning descriptor.
    :param thresh: threshold of saliency maps to distinguish fg and bg.
    :param model_type: type of model to extract descriptors from.
    :param stride: stride of the model.
    :param votes_percentage: the percentage of positive votes so a cluster will be considered salient.
    :param sample_interval: sample every ith descriptor before applying clustering.
    :param low_res_saliency_maps: Use saliency maps with lower resolution (dramatically reduces GPU RAM needs,
    doesn't deteriorate performance).
    :param num_parts: Number of parts of final output.
    :param num_crop_augmentations: number of crop augmentations to apply on input images. Increases performance for
    small sets with high variations.
    :param three_stages: If true, uses three clustering stages - fg/bg, non-common objects, and parts. Increases
    performance for small sets with high variations.
    :param elbow_second_stage: elbow coefficient for clustering in the second stage.
    :param save_dir: optional. if not None save intermediate results in this directory.
    :return: a list of segmentation masks and a list of processed pil images.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if extractor is None:
        extractor = ViTExtractor(model_type, stride, device=device)
    if low_res_saliency_maps:
        saliency_extractor = ViTExtractor(model_type, stride=8, device=device)
    else:
        saliency_extractor = extractor
    num_images = len(image_paths)
    assert num_images>=2
    if save_dir is not None:
        save_dir = Path(save_dir)
      
    # create augmentations if needed
    if num_crop_augmentations > 0:
        augmentations_image_paths = augment_images(extractor, image_paths, save_dir, num_crop_augmentations, load_size)
        image_paths = image_paths + augmentations_image_paths
    
    # extract descriptors and saliency maps for each image
    num_patches_list = []
    load_size_list = []
    image_pil_list = []
    saliency_maps_list = []
    descriptors_list = []
    for image_path in image_paths:
        image_batch, image_pil = extractor.preprocess(image_path, load_size)
        image_pil_list.append(image_pil)

        descs = extractor.extract_descriptors(image_batch.to(device), 
                                              layer, facet, bin).cpu().numpy()
        curr_num_patches, curr_load_size = extractor.num_patches, extractor.load_size

        descriptors_list.append(descs)
        num_patches_list.append(curr_num_patches)
        load_size_list.append(curr_load_size)

        saliency_map = get_saliency_maps(saliency_extractor, image_batch,
                                         image_path, curr_num_patches,
                                         curr_load_size, low_res_saliency_maps,
                                         save_dir, device)
        saliency_maps_list.append(saliency_map)

        if save_dir is not None and not '_aug_' in Path(image_path).stem:
            image_pil.save(save_dir / f'{Path(image_path).stem}_resized.png')
    # Descriptors with key facet have shape [B,H,T,D] - H is #heads T is #tokens. 
    # D is #dimensions. Tokens are the image patches.

    # cluster all images using k-means:
    algorithm, labels = get_K_means(descriptors_list, sample_interval,
                                    elbow, n_cluster_range=list(range(6,15)))
    n_clusters = algorithm.centroids.shape[0]

    num_labels = n_clusters + 1
    num_descriptors_per_image = [num_patches[0]*num_patches[1] for num_patches in num_patches_list]
    labels_per_image = np.split(labels, np.cumsum(num_descriptors_per_image)[:-1])

    if save_dir is not None:
        cmap = 'jet' if num_labels > 10 else 'tab10'
        for path, np, lpi in zip(image_paths, num_patches_list, labels_per_image):
            stem = Path(path).stem
            if not '_aug_' in stem:
                fig, ax = plt.subplots()
                ax.axis('off')
                ax.imshow(lpi.reshape(np), vmin=0, vmax=num_labels-1, cmap=cmap)
                fig.savefig(save_dir / f'{stem}_clustering.png', 
                            bbox_inches='tight', pad_inches=0)
                plt.close(fig)

    # use saliency maps to vote for salient clusters (only original images vote, not augmentations)
    # Each image 'votes' for a label if the mean saliency for that labelled region is above a
    # threshold. Any label which has more than votes_percentage images voting for it, is considered
    # salient.
    votes = np.zeros(num_labels)
    for path, labels, saliency_map in zip(image_paths, labels_per_image, saliency_maps_list):
        if not ('_aug_' in Path(path).stem):
            for label in range(num_labels):
                label_saliency = saliency_map[labels[:, 0] == label].mean()
                if label_saliency > thresh:
                    votes[label] += 1

    salient_labels = np.where(votes >= np.ceil(num_images * votes_percentage / 100))[0]
    # cluster all parts using k-means:
    fg_masks = [np.isin(labels, salient_labels) for labels in labels_per_image]  # get only foreground descriptors
    fg_descriptor_list = [desc[:, :, fg_mask[:, 0], :] for fg_mask, desc in zip(fg_masks, descriptors_list)]

    if len(fg_descriptor_list)==0:
        return [np.zeros(np.asarray(x).shape) for x in image_pil_list], image_pil_list


    normalized_all_fg_descriptors = get_normal_descriptors(fg_descriptor_list)
    sampled_fg_descriptors_list = [x[:, :, ::sample_interval, :] for x in fg_descriptor_list]    
    normalized_all_fg_sampled_descriptors = get_normal_descriptors(sampled_fg_descriptors_list)

    sum_of_squared_dists = []
    # if applying three stages, use elbow to determine number of clusters
    # in second stage, otherwise use the specified number of parts.

    n_cluster_range = list(range(1, 15)) if three_stages else [num_parts]
    part_algorithm, part_labels = get_K_means(fg_descriptor_list, sample_interval,
                                              elbow_second_stage, n_cluster_range)

    part_num_labels = np.max(part_labels) + 1
    parts_num_descriptors_per_image = [np.count_nonzero(mask) for mask in fg_masks]
    part_labels_per_image = np.split(part_labels, np.cumsum(parts_num_descriptors_per_image))

    # get smoothed parts using crf
    part_segmentations = []
    for image_path, img, num_patches, load_size, descs in zip(image_paths, image_pil_list, num_patches_list, load_size_list, descriptors_list):
        if ('_aug_' in Path(image_path).stem) and not three_stages:
            part_segmentations.append(None)
            continue
        bg_centroids = tuple(i for i in range(algorithm.centroids.shape[0]) if not i in salient_labels)
        
        curr_normalized_descs = descs[0, 0].astype(np.float32)
        faiss.normalize_L2(curr_normalized_descs.copy(order='C'))  # in-place operation
        # distance to parts
        dist_to_parts = ((curr_normalized_descs[:, None, :] - part_algorithm.centroids[None, ...]) ** 2
                         ).sum(axis=2)
        # dist to BG
        dist_to_bg = ((curr_normalized_descs[:, None, :] - algorithm.centroids[None, bg_centroids, :]) ** 2
                      ).sum(axis=2)
        min_dist_to_bg = np.min(dist_to_bg, axis=1)[:, None]
        
        final = CRF_v1(img, dist_to_parts, min_dist_to_bg, num_patches, part_num_labels, load_size)
        parts_float = final.astype(np.float32)
        parts_float[parts_float == part_num_labels] = np.nan
        part_segmentations.append(parts_float)

    if three_stages:  # if needed, apply third stage

        # visualize second stage
        if num_crop_augmentations > 0:
            curr_part_segmentations, curr_image_pil_list = [], []
            for image_path, part_seg, pil_image in zip(image_paths, part_segmentations, image_pil_list):
                if not ('_aug_' in Path(image_path).stem):
                    curr_part_segmentations.append(part_seg)
                    curr_image_pil_list.append(pil_image)
        else:
            curr_part_segmentations, curr_image_pil_list = part_segmentations, image_pil_list

        if save_dir is not None:
            part_figs = draw_part_cosegmentation(part_num_labels, curr_part_segmentations, curr_image_pil_list)
            for image, part_fig in zip(image_paths, part_figs):
                part_fig.savefig(save_dir / f'{Path(image).stem}_vis_sec_stage.png', bbox_inches='tight', pad_inches=0)
            plt.close('all')

        # get labels after crf for each descriptor
        smoothed_part_labels_per_image = []
        for part_segment, num_patches in zip(part_segmentations, num_patches_list):
            resized_part_segment = np.array(torch.nn.functional.interpolate(torch.from_numpy(part_segment)
                                                                            [None, None, ...], size=num_patches,
                                                                            mode='nearest')[0, 0])
            smoothed_part_labels_per_image.append(resized_part_segment.flatten())

        # take only parts that appear in all original images (otherwise they belong to non-common objects)
        votes = np.zeros(part_num_labels)
        for image_path, image_labels in zip(image_paths, smoothed_part_labels_per_image):
            if not ('_aug_' in Path(image_path).stem):
                unique_labels = np.unique(image_labels[~np.isnan(image_labels)]).astype(np.int32)
                votes[unique_labels] += 1
        common_labels = np.where(votes == num_images)[0]

        # get labels after crf for each descriptor
        common_parts_masks = []
        for part_segment in smoothed_part_labels_per_image:
            common_parts_masks.append(np.isin(part_segment, common_labels).flatten())

        # cluster all final parts using k-means:
        common_descriptor_list = [desc[:, :, mask, :] for mask, desc in zip(common_parts_masks, descriptors_list)]
        all_common_descriptors = np.ascontiguousarray(np.concatenate(common_descriptor_list, axis=2)[0, 0])
        normalized_all_common_descriptors = all_common_descriptors.astype(np.float32)
        faiss.normalize_L2(normalized_all_common_descriptors)  # in-place operation
        sampled_common_descriptors_list = [x[:, :, ::sample_interval, :] for x in common_descriptor_list]
        all_common_sampled_descriptors = np.ascontiguousarray(np.concatenate(sampled_common_descriptors_list,
                                                                             axis=2)[0, 0])
        normalized_all_common_sampled_descriptors = all_common_sampled_descriptors.astype(np.float32)
        faiss.normalize_L2(normalized_all_common_sampled_descriptors)  # in-place operation

        common_part_algorithm = faiss.Kmeans(d=normalized_all_common_sampled_descriptors.shape[1], k=num_parts,
                                             niter=300, nredo=10)
        common_part_algorithm.train(normalized_all_common_sampled_descriptors.astype(np.float32))
        
        d_fn = distances.LpDistance(p=2, power=1,normalize_embeddings=False)
        d = d_fn(torch.Tensor(normalized_all_common_descriptors), 
                torch.Tensor(part_algorithm.centroids))
        common_part_labels = d.argmax(dim=1, keepdim=True).cpu().detach().numpy() # N,1
        #_, common_part_labels = part_algorithm.index.search(normalized_all_common_descriptors.astype(np.float32), 1)

        common_part_num_labels = np.max(common_part_labels) + 1
        parts_num_descriptors_per_image = [np.count_nonzero(mask) for mask in common_parts_masks]
        common_part_labels_per_image = np.split(common_part_labels, np.cumsum(parts_num_descriptors_per_image))

        # get smoothed parts using crf
        common_part_segmentations = []
        for img, num_patches, load_size, descs in zip(image_pil_list, num_patches_list, load_size_list, descriptors_list):
            bg_centroids_1 = tuple(i for i in range(algorithm.centroids.shape[0]) if not i in salient_labels)
            bg_centroids_2 = tuple(i for i in range(part_algorithm.centroids.shape[0]) if not i in common_labels)
            curr_normalized_descs = descs[0, 0].astype(np.float32)
            curr_normalized_descs = np.ascontiguousarray(curr_normalized_descs)
            faiss.normalize_L2(curr_normalized_descs)  # in-place operation

            # distance to parts
            dist_to_parts = ((curr_normalized_descs[:, None, :] - common_part_algorithm.centroids[None, ...]) ** 2).sum(
                axis=2)
            # dist to BG
            dist_to_bg_1 = ((curr_normalized_descs[:, None, :] -
                             algorithm.centroids[None, bg_centroids_1, :]) ** 2).sum(axis=2)
            dist_to_bg_2 = ((curr_normalized_descs[:, None, :] -
                             part_algorithm.centroids[None, bg_centroids_2, :]) ** 2).sum(axis=2)
            dist_to_bg = np.concatenate((dist_to_bg_1, dist_to_bg_2), axis=1)
            min_dist_to_bg = np.min(dist_to_bg, axis=1)[:, None]
            

            final = CRF_v1(img, dist_to_parts, min_dist_to_bg, num_patches, num_parts, load_size)
            common_parts_float = final.astype(np.float32)
            common_parts_float[common_parts_float == num_parts] = np.nan
            common_part_segmentations.append(common_parts_float)

        # reassign third stage results as final results
        part_segmentations = common_part_segmentations

    # remove augmentation results if existing
    if num_crop_augmentations > 0:
        no_aug_part_segmentations, no_aug_image_pil_list = [], []
        for image_path, part_seg, pil_image in zip(image_paths, part_segmentations, image_pil_list):
            if not ('_aug_' in Path(image_path).stem):
                no_aug_part_segmentations.append(part_seg)
                no_aug_image_pil_list.append(pil_image)
        part_segmentations = no_aug_part_segmentations
        image_pil_list = no_aug_image_pil_list

    return part_segmentations, image_pil_list
