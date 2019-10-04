import numpy as np
from PIL import Image
from matplotlib import cm
from PIL import Image, ImageOps, ImageEnhance, PILLOW_VERSION
import cv2
import PIL
from torchvision import transforms as T
from torchvision.transforms import functional as F
import numpy as np
import random
import torch

def pad_if_smaller(img, size, fill=0):
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img

class RandomResize(object):
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        size = random.randint(self.min_size, self.max_size)
        image = F.resize(image, size)
        target = F.resize(target, size, interpolation=Image.NEAREST)
        return image, target


def recrop_image(image, color_threshold=180, x_padding=5, y_padding=2):
    """ Crop image from path to content (e.g. darkest pixels, alpha channel)

    Args:
        image_path (array-like): Numpy array of image
        color_threshold: pixels darker than this will be considered border

        : padding to add after choosing crop

    Returns:
        np-array: A BGR numpy picture representation
    """
    image = np.asarray(image)
    ## Define crop max
    if len(image.shape) == 3:
        average_color = np.mean(image, axis=2)
        mask_coords = np.where(average_color < color_threshold)
    else:
        mask_coords = np.where(image < color_threshold)

    qualified_coords = np.sum(np.array(mask_coords).shape)

    if qualified_coords > 1000:
        # crop to picture
        x_min = np.min(mask_coords[0])
        x_max = np.max(mask_coords[0])
        y_min = np.min(mask_coords[1])
        y_max = np.max(mask_coords[1])

        x_min = max(0, x_min - x_padding)
        x_max = min(image.shape[0], x_max + x_padding)
        y_min = max(0, y_min - y_padding)
        y_max = min(image.shape[1], y_max + y_padding)

        # if x_max+y_max-x_min-y_min + maximum_crop > np.sum(image.shape):
        # crop to image
        image = image[x_min:x_max, y_min:y_max]

    return Image.fromarray(image)


class ResizePad(object):
    def __init__(self, target_size, pad_color=(255, 255, 255), padding_placement="random"):
        self.target_size = target_size
        self.pad_color = pad_color
        self.padding_placement = padding_placement

    def __call__(self, image, target=None):
        image = np.asarray(image)
        x_target, y_target = self.target_size
        y, x = image.shape
        new_x = int(y_target * x / y)
        new_shape = (new_x, y_target)
        image = cv2.resize(image, dsize=new_shape)[:, :]

        total_padding = x_target - new_x
        if self.padding_placement=="random":
            left = random.randint(0,total_padding)
        elif self.padding_placement=="equal":
            left = int(total_padding / 2)
        right = total_padding - left
        image = cv2.copyMakeBorder(image, top=0, bottom=0, left=left, right=right, borderType=cv2.BORDER_CONSTANT,
                                   value=self.pad_color)

        image = Image.fromarray(image)
        if target:
            target = np.asarray(target)
            target = cv2.copyMakeBorder(target, top=0, bottom=0, left=left, right=right, borderType=cv2.BORDER_CONSTANT,
                                       value=self.pad_color)
            image = Image.fromarray(target)
            return image, target
        else:
            return image

class MinimalCrop(object):
    def __init__(self,  color_threshold=180, x_padding=5, y_padding=2):
        self.color_threshold = color_threshold
        self.x_padding = x_padding
        self.y_padding = y_padding

    def __call__(self, image):
        """ Crop image from path to content (e.g. darkest pixels, alpha channel)

        Args:
            image_path (array-like): Numpy array of image
            color_threshold: pixels darker than this will be considered border

            : padding to add after choosing crop

        Returns:
            np-array: A BGR numpy picture representation
        """
        image = np.asarray(image)
        ## Define crop max
        if len(image.shape) == 3:
            average_color = np.mean(image, axis=2)
            mask_coords = np.where(average_color < self.color_threshold)
        else:
            mask_coords = np.where(image < self.color_threshold)

        qualified_coords = np.sum(np.array(mask_coords).shape)

        if qualified_coords > 1000:
            # crop to picture
            x_min = np.min(mask_coords[0])
            x_max = np.max(mask_coords[0])
            y_min = np.min(mask_coords[1])
            y_max = np.max(mask_coords[1])

            x_min = max(0, x_min - self.x_padding)
            x_max = min(image.shape[0], x_max + self.x_padding)
            y_min = max(0, y_min - self.y_padding)
            y_max = min(image.shape[1], y_max + self.y_padding)

            # if x_max+y_max-x_min-y_min + maximum_crop > np.sum(image.shape):
            # crop to image
            image = image[x_min:x_max, y_min:y_max]

        return Image.fromarray(image)