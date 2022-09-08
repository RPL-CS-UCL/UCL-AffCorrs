# One-Shot Affordance Region Correspondences with AffCorrs

This repository contains code for annotation generation, and affordance correspondence described in the paper  :exclamation:reference:exclamation:

One-shot transfer example outputs from our model shown in pairs of refernce image and annotated areas (left) and the target image with estimated correspondences (right) without any training on these classes.

![image](https://user-images.githubusercontent.com/30011340/178766579-5d488cb8-646e-4349-9829-864e77e05c0b.png)


## Dependencies

The code has been tested with the following packages:

```python
pydensecrf=1.0
torch=1.10+cu113
faiss=1.5
pytorch_metric_learning=0.9.99
fast_pytorch_kmeans=0.1.6
timm=0.6.7
cv2=4.6
scikit-image=0.17.2
```

However, other versions of these packages will likely be sufficient as well.

```
pip3 install pydensecrf torch torchvision timm cv2 scikit-image\
faiss pytorch_metric_learning fast_pytorch_kmeans
```

## Demo
To run a demo of the model `cd demos; python3 show_part_annotation_correspondence.py`.

To try on your own images, (i) change `SUPPORT_DIR` to point to a directory containing the `affordance.npy` and `prototype.png` and (ii) change `TARGET_IMAGE_PATH` to point toward an image file which contains the target scene.

## Annotation

Annotation can be generated via `annotate_parts.py`. Four named windows will appear. Drawing on the source image with the mouse will create the affordance mask. Middle-click
will save the current affordance and proceed to the next affordance mask. Pressing `s` will save all the affordance masks generated for this source image into a folder called `affordance_database/temp`. Pressing the `n`-key will change the source image to the next in the folder. The target image and similarity windows are there just for reference while annotating, showing the best point correspondences.

