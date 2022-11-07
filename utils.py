import cv2
import numpy as np


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    """Resize image with original size ratio if one dimension is not specified

    Args:
        image: RGB image with shape of (img_h, img_w, 3)
        width: desired width of resized image (int)
        height: desire height of resized image (int)
        inter: interpolation method of cv2

    Returns:
        resized_img: resized image with new shape disired width/height
    """
    dim = None
    img_h, img_w = image.shape[:2]

    # if both the width and height are None, return the original image
    if width == None and height == None:
        return image

    dim = (width, height)

    if width == None:
        # calculate the ratio of the height and construct the dimensions
        r = height / img_h
        dim = (int(img_w * r), height)
    
    if height == None:
        # calculate the ratio of the width and construct the dimensions
        r = width / img_w
        dim = (width, int(img_h * r))
        
    # resize the image
    resized_img = cv2.resize(image, dim, interpolation = inter)

    return resized_img


def crop_object(obj_img, obj_mask, obj_id):
    """Crop the object from the obj_img based on the obj_mask

    Args:
        obj_img: RGB image of object with shape of (obj_height, obj_width, 3)
        obj_mask: boolean mask of object with shape of (obj_height, obj_width), 
            True is object area and False is background area
        obj_id: id value of object in obj_mask (int)

    Returns:
        cropped_obj: cropped image of object only (crop_height, crop_width, 3)
        cropped_mask: boolean mask of the corresponding cropped image (crop_height, crop_width)
    """
    boolean_mask = obj_mask == obj_id
    segmentation = np.where(boolean_mask == True)
    x_min = np.min(segmentation[1])
    x_max = np.max(segmentation[1])
    y_min = np.min(segmentation[0])
    y_max = np.max(segmentation[0])

    # Crop image from just the portion of the image that fits the object
    cropped_obj = obj_img[y_min:y_max, x_min:x_max, :]
    # Crop the mask to match the cropped image
    cropped_mask = obj_mask[y_min:y_max, x_min:x_max] 

    return cropped_obj, cropped_mask


def paste_object(bg_img, bg_mask, obj_img, obj_mask, obj_id, paste_pos):
    '''Paste an object on an image

    Args:
        bg_img: RGB background image with shape of (bg_height, bg_width, 3) 
        bg_mask: mask of background image with shape of (bg_height, bg_width)
        obj_img: RGB image of object with shape of (obj_height, obj_width, 3)
        obj_mask: mask of object with shape of (obj_height, obj_width), 
        obj_id: id value of object in obj_mask (int)
        paste_pos: tuple of (x, y) - coordinates of paste position 
            (x, y) is also the center of the object image
            0 <= x <= width of background image
            0 <= y <= height of background image
    
    Returns:
        res_img: RGB background image with added object 
            with shape of (bg_height, bg_width, 3) 
        res_mask: corresponding mask with added object mask 
            with shape of (bg_height, bg_width)
    '''
    assert 0 <= paste_pos[0] <= bg_img.shape[1] \
        and 0 <= paste_pos[1] <= bg_img.shape[0], "paste position out of range"

    res_img = bg_img.copy()
    res_mask = bg_mask.copy()
    bg_height, bg_width = res_img.shape[:2]
    obj_height, obj_width = obj_img.shape[:2]
    
    # Calculate the top left corner from paste position according to obj_img size
    x, y = paste_pos[0] - obj_width // 2, paste_pos[1] - obj_height // 2

    # Separate object from obj_img
    boolean_mask = obj_mask == obj_id # True is object area and False is background area
    mask_rgb = np.stack([boolean_mask, boolean_mask, boolean_mask], axis=2)
    obj = obj_img * mask_rgb
    
    if x >= 0:
        # width_part - part of the image which overlaps background along x-axis
        width_part = obj_width - max(0, x + obj_width - bg_width) 
        if y >= 0:
            # height_part - part of the image which overlaps background along y-axis
            height_part = obj_height - max(0, y + obj_height - bg_height) 
            paste_area_mask = ~mask_rgb[:height_part, :width_part, :]
            obj_part = obj[:height_part, :width_part, :]
            mask_part = obj_mask[:height_part, :width_part]
        else:
            height_part = obj_height + y
            paste_area_mask = ~mask_rgb[obj_height - height_part:, :width_part, :]
            obj_part = obj[obj_height - height_part:, :width_part, :]
            mask_part = obj_mask[obj_height - height_part:, :width_part]
            y = 0
    else:
        width_part = obj_width + x
        x = 0
        if y >= 0:
            height_part = obj_height - max(0, y + obj_height - bg_height) 
            paste_area_mask = ~mask_rgb[:height_part, obj_width -width_part:, :]
            obj_part = obj[:height_part, obj_width - width_part:, :]
            mask_part = obj_mask[:height_part, obj_width - width_part:]
        else:
            height_part = obj_height + y
            paste_area_mask = ~mask_rgb[obj_height - height_part:, obj_width - width_part:, :]
            obj_part = obj[obj_height - height_part:, obj_width - width_part:, :]
            mask_part = obj_mask[obj_height - height_part:, obj_width - width_part:]
            y = 0
        
    res_img[y:y + height_part, x:x + width_part, :] \
        = res_img[y:y + height_part, x:x + width_part, :] * paste_area_mask + obj_part
    res_mask[y:y + height_part, x:x + width_part] \
        = res_mask[y:y + height_part, x:x + width_part] * paste_area_mask[:, :, 0] + mask_part
    return res_img, res_mask