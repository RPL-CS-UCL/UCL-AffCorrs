import torch
import torchvision.transforms as tf
import faiss

import numpy as np
from skimage.transform import resize
import pydensecrf.densecrf as dcrf

from pytorch_metric_learning import distances
d_fn = distances.CosineSimilarity()

def get_normal_descriptors(descriptors_list):
    """ Returns normalized descriptor array
      :param descriptor list: list of descriptors [1,1,N,D]
      :return: normalized descriptor array [N,D]
    """
    all_descriptors = np.ascontiguousarray(np.concatenate(descriptors_list, axis=2)[0, 0])
    normalized_all_descriptors = all_descriptors.astype(np.float32)
    faiss.normalize_L2(normalized_all_descriptors)  # in-place operation
    return normalized_all_descriptors

def CRF( img, d_to_cent, num_patches, num_parts, load_size ):
    """ Returns CRF output
      :param img: PIL.Image, input image [H,W,C]
      :param d_to_cent: Distance to each part centroid [N,K]
      :param num_patches: Number of patches in each dimension
      :param num_parts: number of parts K
      :param load_size: load size of the image
    """  
    # upsample to load size
    upsample = torch.nn.Upsample(size=load_size)
    u = np.array(upsample(torch.from_numpy(d_to_cent).permute(2, 0, 1)[None, ...])[0].permute(1, 2, 0))
    d = dcrf.DenseCRF2D(u.shape[1], u.shape[0], u.shape[2])
    # Set as unary of DenseCRF
    d.setUnaryEnergy(np.ascontiguousarray(u.reshape(-1, u.shape[-1]).T))
    compat = [40, 25] # 50,15
    # Add Gaussian for XY
    d.addPairwiseGaussian(sxy=(3, 3), compat=compat[0], kernel=dcrf.DIAG_KERNEL,
                          normalization=dcrf.NORMALIZE_SYMMETRIC)
    # Add Bilateral for XYRGB
    d.addPairwiseBilateral(sxy=5, srgb=13, rgbim=np.array(img), compat=compat[1], kernel=dcrf.DIAG_KERNEL,
                          normalization=dcrf.NORMALIZE_SYMMETRIC)
    # Infer
    Q = d.inference(10)
    # Assign class
    final = np.argmax(Q, axis=0)
    return final

def CRF_v1( img, dist_to_parts, min_dist_to_bg, num_patches, num_parts, load_size ):
    """ Returns CRF output
      :param img: PIL.Image, input image [H,W,C]
      :param dist_to_parts: Distance to each part centroid [N,K]
      :param min_dist_to_bg: Minimum distance to a background centroid
      :param num_patches: Number of patches in each dimension
      :param num_parts: number of parts K
      :param load_size: load size of the image
    """  
    print ("Using standard CRF")

    d_to_cent = np.concatenate((dist_to_parts, min_dist_to_bg), axis=1)
    d_to_cent = d_to_cent.reshape(num_patches[0], num_patches[1], d_to_cent.shape[-1])
    # Invert distance
    # Subtract the maximum for each descriptor? make negative energy
    d_to_cent = d_to_cent - np.max(d_to_cent, axis=-1)[..., None]
    return CRF( img, d_to_cent, 
                num_patches, num_parts, load_size ).reshape(load_size)


def CRF_v2(img, descs, fg_centroids, bg_centroids,
          load_size, load_shape, num_patches):
    """ Returns CRF output
      :param img: PIL.Image, input image [H,W,C]
      :param dist_to_parts: Distance to each part centroid [N,K]
      :param min_dist_to_bg: Minimum distance to a background centroid
      :param num_patches: Number of patches in each dimension
      :param num_parts: number of parts K
      :param load_size: load size of the image
    """  
    print ("Using CRF - Cosine Similarity. Expecting normalized centroids")

    # Get normalized descriptors - All
    normalized_all_descriptors = get_normal_descriptors([descs])
    # # Get distance to centroids 
    # (HW, D) (K, D)
    dist_to_parts = 1 - d_fn(torch.Tensor(normalized_all_descriptors), 
              torch.Tensor(fg_centroids)).numpy()
    dist_to_bg = 1 - d_fn(torch.Tensor(normalized_all_descriptors), 
              torch.Tensor(bg_centroids)).numpy()
    min_dist_to_bg = np.min(dist_to_bg, axis=1)[:, None]

    d_to_cent = np.concatenate((dist_to_parts, min_dist_to_bg), axis=1)
    d_to_cent = d_to_cent.reshape(num_patches[0], num_patches[1], d_to_cent.shape[-1])
    # Invert distance
    # Subtract the maximum for each descriptor? make negative energy
    d_to_cent = d_to_cent - np.max(d_to_cent, axis=-1)[..., None]

    final = CRF( img, d_to_cent, num_patches, None, load_size )
    final = final.reshape(*load_shape)
    return final


def CRF_with_depth(img, depth_image, descs, fg_centroids, bg_centroids,
          load_size, load_shape, num_patches):
    """ Returns CRF output with depth
    """  
    raise NotImplementedError()