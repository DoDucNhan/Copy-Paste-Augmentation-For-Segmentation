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


def contour_area(contours):
    """Calculate and create a descending sorted list of contour by its area
    Args:
        contours: result of cv2.findContours

    Returns:
        sorted_area: descending array of contour by area
    """
    cnt_area = []
    for cnt in contours:
        # Calculate the area of the contour
        cnt_area.append([cv2.contourArea(cnt), cnt])

    cnt_area = np.array(cnt_area)
    sorted_area = cnt_area[cnt_area[:, 0].argsort()][::-1]

    return sorted_area[:, 1]
  

def crop_object(obj_img, obj_mask, obj_id, max_obj=1):
    """Crop the object from the obj_img based on the obj_mask

    Args:
        obj_img: RGB image of object with shape of (obj_height, obj_width, 3)
        obj_mask: image mask of object with shape of (obj_height, obj_width), 
            True is object area and False is background area
        obj_id: id value of object in obj_mask (int)

    Returns:
        cropped_obj: list of cropped image of object only 
        cropped_mask: list of mask of the corresponding cropped image 
    """
    cropped_obj = []
    cropped_mask = []
    item_mask = obj_mask == obj_id
    contours, _ = cv2.findContours(item_mask.astype('u1'), 
                                           cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Get the list of sorted contour by area
    sorted_area = contour_area(contours)
    # If max_obj exceed number of contours, max_obj = number of contours - 1
    ## Last contour area is likely noise
    max_obj = len(sorted_area) - 1 if max_obj >= len(sorted_area) else max_obj 
    for cnt in sorted_area[:max_obj]:
        # Get the details of the bounding rectangle of contour
        x, y, w, h = cv2.boundingRect(cnt)
        cropped_obj.append(obj_img[y:y + h, x:x + w])
        cropped_mask.append(obj_mask[y:y + h, x:x + w])
 
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