import torch
import torchvision.transforms as tf
from pathlib import Path
from PIL import Image
import numpy as np
from skimage.transform import resize
from skimage import img_as_bool

def augment_images(extractor, image_paths, save_dir, num_crop_augmentations, load_size):
    """Augments images """
    augmentations_image_paths = []
    augmentations_dir = save_dir / 'augs' if save_dir is not None else Path('augs')
    augmentations_dir.mkdir(exist_ok=True, parents=True)
    for image_path in image_paths:
        image_batch, image_pil = extractor.preprocess(image_path, load_size)
        flipped_image_pil = image_pil.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = (int(image_batch.shape[2] * 0.95), int(image_batch.shape[3] * 0.95))
        random_crop = tf.RandomCrop(size=crop_size)
        for i in range(num_crop_augmentations):
            random_crop_file_name = augmentations_dir / f'{Path(image_path).stem}_resized_aug_{i}.png'
            random_crop(image_pil).save(random_crop_file_name)
            random_crop_flipped_file_name = augmentations_dir / f'{Path(image_path).stem}_resized_flip_aug_{i}.png'
            random_crop(flipped_image_pil).save(random_crop_flipped_file_name)
            augmentations_image_paths.append(random_crop_file_name)
            augmentations_image_paths.append(random_crop_flipped_file_name)
    return augmentations_image_paths

def swap_background(image, mask):
    """ Swaps mask-background """
    def get_gradient_2d(start, stop, width, height, is_horizontal):
        if is_horizontal:
          return np.tile(np.linspace(start, stop, width), (height, 1))
        else:
          return np.tile(np.linspace(start, stop, height), (width, 1)).T

    def get_gradient_3d(width, height, start_list, stop_list, is_horizontal_list):
        result = np.zeros((height, width, len(start_list)), dtype=np.float)
        for i, (start, stop, is_horizontal) in enumerate(zip(start_list, stop_list, is_horizontal_list)):
          result[:, :, i] = get_gradient_2d(start, stop, width, height, is_horizontal)
        result = torch.Tensor(result).permute(2,0,1)
        return result

    def add_gaussian_noise(image):
        ch,row,col= image.shape
        mean = 0
        var = 1e-3
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(ch, row,col)
        noisy = image + torch.Tensor(gauss).to(image.device)
        noisy [noisy>1] = 1
        noisy [noisy<0] = 0
        return noisy

    rgb_augmentation = tf.Compose([
        tf.ToTensor(),
        tf.ColorJitter(hue=0.04,saturation=0.04,brightness=0.04,contrast=0.04)
    ])
    def rand(m_a=1, m_i=0):
      return np.random.random()*(m_a-m_i)+m_i
    image = rgb_augmentation(image)
    aug_im =  get_gradient_3d(image.shape[-1], image.shape[-2], 
                          (int(rand(255)),int(rand(255)),int(rand(255))), 
                          (int(rand(255)),int(rand(255)),int(rand(255))), 
                          (True,False,False))/255.
    image[:, mask==False] = aug_im[:, mask==False]
    image = add_gaussian_noise(image)
    image = tf.ToPILImage()(image)
    return image

def augment_images_v2(extractor, images, num_crop_augmentations, 
                      load_size, masks=None):
    """ Augments images """
    augmentations_image_paths = ['_aug_'] * len(images)
    augmentations = []
    for i in range (len(images)):
        image_batch,image_pil = extractor.preprocess_image(images[i], load_size)
        im_aug = image_pil
        for j in range (num_crop_augmentations):
            if masks is not None: 
                if masks[i] is not None: 
                    masks[i] = img_as_bool(resize(masks[i],  
                                                  np.asarray(im_aug).shape[0:2] ))
                    im_aug = swap_background(im_aug, masks[i])
            augmentations.append(im_aug)
    return augmentations, augmentations_image_paths


def get_mask_crop_lims(mask, bound_scale, min_side):
    """ Returns mask cropping indices """
    H,W = mask.shape
    if torch.is_tensor(mask):
        uv = torch.nonzero(mask)
        # get mean of mask
        umean = uv[:,0].float().mean(axis=-1)
        vmean = uv[:,1].float().mean(axis=-1)
        # get bounds of mask
        umax = uv[:,0].max()
        vmax = uv[:,1].max()
        umin = uv[:,0].min()
        vmin = uv[:,1].min()
    else:
        u,v = np.nonzero(mask)
        # get mean of mask
        umean = u.mean(axis=-1)
        vmean = v.mean(axis=-1)
        # get bounds of mask
        umax = u.max()
        vmax = v.max()
        umin = u.min()
        vmin = v.min()
    h = vmax - vmin 
    w = umax - umin
    side = max(int(max(h,w)*bound_scale), min_side)
    # make square crop 
    crop_ulims = np.array([umean - side//2, umean + side//2]).astype(np.int32)
    crop_vlims = np.array([vmean - side//2, vmean + side//2]).astype(np.int32)
    # if padding is within limits, crop as normal
    # shift crop to cover the area without any zero padding.
    if crop_ulims.min() < 0: crop_ulims -= crop_ulims.min()
    if crop_vlims.min() < 0: crop_vlims -= crop_vlims.min()
    if crop_ulims.max() >= H: crop_ulims += H-crop_ulims.max()
    if crop_vlims.max() >= W: crop_vlims += W-crop_vlims.max()
    return crop_ulims, crop_vlims

def crop_mask(image, mask, 
              bound_scale : float =1.25, min_side : int =50):
    """
    Returns image and mask crops at mask centroid.
    Inputs:
      image : np.ndarray of shape CHW
      mask  : np.ndarray of shape HW
      bound_scale : float, bound pad percentage
      min_side : int, minimum side of crop
    Returns:
      image_crop : np.ndarray of shape CHW
      mask_crop  : np.ndarray of shape HW
    """
    u_crop, v_crop = get_mask_crop_lims(mask, bound_scale, min_side)
    mask_crop = mask [u_crop[0]:u_crop[1], v_crop[0]:v_crop[1]] 
    image_crop = image [...,u_crop[0]:u_crop[1], v_crop[0]:v_crop[1]] 
    return image_crop, mask_crop, (u_crop,v_crop)