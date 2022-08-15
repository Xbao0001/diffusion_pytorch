import numpy as np
from einops import rearrange
from PIL import Image


def normalize_to_neg_one_to_one(img):
    return img * 2 - 1


def unnormalize_to_zero_to_one(img):
    return (img + 1) * 0.5


def make_grid(images:np.ndarray, n_cols=4):
    """ 
    convert a batch of numpy images to one image which has n_cols columns.

    Args:
        img: numpy.ndarray or jt.Var of shape (b c h w) or (b h w c)
        n_cols: number of images per row

    Returns:
        single image of shape (h*n_rows, w*n_cols, c), where n_rows * n_cols 
        = batch_size, if not, it will add some black images
    """
    assert len(images.shape) == 4, 'image must be 4d array'

    if images.shape[0] % n_cols != 0:
        B, C, H, W = images.shape
        padding_images = np.zeros((n_cols - (B % n_cols), C, H, W))
        images = np.vstack((images, padding_images))

    if images.shape[1] == 3 or images.shape[1] == 1:
        grid = rearrange(images,
                         '(n_rows n_cols) c h w -> (n_rows h) (n_cols w) c',
                         n_cols=n_cols)
    elif images.shape[3] == 3 or images.shape[3] == 1:
        grid = rearrange(images,
                         '(n_rows n_cols) h w c -> (n_rows h) (n_cols w) c',
                         n_cols=n_cols)
    else:
        raise ValueError(f"Can not process images of shape: {images.shape}")

    if grid.shape[-1] == 1:  # for wandb to log segmentation
        return grid[:, :, 0]
    return grid


def to_pil_image(x: np.ndarray):
    assert x.ndim == 3, "only support C H W or H W C, where C = 3"
    if x.shape[0] == 3: x = x.transpose(1, 2, 0) # C H W -> H W C
    return Image.fromarray(np.uint8(x * 255), 'RGB')
