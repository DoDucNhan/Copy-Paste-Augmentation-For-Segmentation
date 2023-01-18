import cv2
import mmcv
import numpy as np
from mmcv.parallel import collate, scatter
from mmseg.datasets.pipelines import Compose

#-----------------------PREPROCESSING--------------------------

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


def sort_contour(contours):
    """Calculate and create a descending sorted list of contour by its area
    Args:
        contours: result of cv2.findContours

    Returns:
        sorted_contours: descending array of contour by area
    """
    cnt_area = []
    for cnt in contours:
        # Calculate the area of the contour
        cnt_area.append(cv2.contourArea(cnt))

    cnt_area = np.array(cnt_area, dtype='float')
    sorted_contours = np.array(contours)[np.argsort(cnt_area)[::-1]]

    return sorted_contours
  

def crop_object(obj_img, obj_mask, obj_id, max_obj=1):
    """Crop the object from the obj_img based on the obj_mask

    Args:
        obj_img: RGB image of object with shape of (obj_height, obj_width, 3)
        obj_mask: image mask of object with shape of (obj_height, obj_width), 
            obj_id value is object area and 0 is background area
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
    sorted_contours = sort_contour(contours)
    # If max_obj exceed number of contours, max_obj = number of contours - 1
    ## Last contour area is likely noise
    if max_obj >= len(sorted_contours):
        if len(sorted_contours) == 1:
            max_obj = 1
        else:
            max_obj = len(sorted_contours) - 1
    for cnt in sorted_contours[:max_obj]:
        # Get the details of the bounding rectangle of contour
        x, y, w, h = cv2.boundingRect(cnt)
        cropped_obj.append(obj_img[y:y + h, x:x + w])
        cropped_mask.append(obj_mask[y:y + h, x:x + w])
 
    return cropped_obj, cropped_mask



#-----------------------UTILS FOR SEGMENTATION PHASE------------------------

class LoadImage:
    """A simple pipeline to load image."""

    def __call__(self, results):
        """Call function to load images into results.

        Args:
            results (dict): A result dict contains the file name
                of the image to be read.

        Returns:
            dict: ``results`` will be returned containing loaded image.
        """

        if isinstance(results['img'], str):
            results['filename'] = results['img']
            results['ori_filename'] = results['img']
        else:
            results['filename'] = None
            results['ori_filename'] = None
        img = mmcv.imread(results['img'])
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results


def get_image_data(model, img_path):
    cfg = model.cfg
    # build the data pipeline
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)

    # prepare data
    data = []
    imgs = img_path if isinstance(img_path, list) else [img_path]
    for img in imgs:
        img_data = dict(img=img)
        img_data = test_pipeline(img_data)
        data.append(img_data)
    data = collate(data, samples_per_gpu=len(imgs))
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, ['cuda'])[0]
    else:
        data['img_metas'] = [i.data[0] for i in data['img_metas']]
    
    return data


#-----------------------COPY-PASTE-METHOD--------------------------


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

    boolean_mask = obj_mask == obj_id # True is object area and False is background area
    # All false means no object in image, return original image and mask
    if (~boolean_mask).all():
        return bg_img, bg_mask

    res_img = bg_img.copy()
    res_mask = bg_mask.copy()
    bg_height, bg_width = res_img.shape[:2]
    obj_height, obj_width = obj_img.shape[:2]
    
    # Calculate the top left corner from paste position according to obj_img size
    x, y = paste_pos[0] - obj_width // 2, paste_pos[1] - obj_height // 2

    # Separate object from obj_img
    mask_rgb = np.stack([boolean_mask, boolean_mask, boolean_mask], axis=2)
    obj = obj_img * mask_rgb
    mask = obj_mask * boolean_mask
    
    if x >= 0:
        # width_part - part of the image which overlaps background along x-axis
        width_part = obj_width - max(0, x + obj_width - bg_width) 
        if y >= 0:
            # height_part - part of the image which overlaps background along y-axis
            height_part = obj_height - max(0, y + obj_height - bg_height) 
            paste_area_mask = ~mask_rgb[:height_part, :width_part, :]
            obj_part = obj[:height_part, :width_part, :]
            mask_part = mask[:height_part, :width_part]
        else:
            height_part = obj_height + y
            paste_area_mask = ~mask_rgb[obj_height - height_part:, :width_part, :]
            obj_part = obj[obj_height - height_part:, :width_part, :]
            mask_part = mask[obj_height - height_part:, :width_part]
            y = 0
    else:
        width_part = obj_width + x
        x = 0
        if y >= 0:
            height_part = obj_height - max(0, y + obj_height - bg_height) 
            paste_area_mask = ~mask_rgb[:height_part, obj_width - width_part:, :]
            obj_part = obj[:height_part, obj_width - width_part:, :]
            mask_part = mask[:height_part, obj_width - width_part:]
        else:
            height_part = obj_height + y
            paste_area_mask = ~mask_rgb[obj_height - height_part:, obj_width - width_part:, :]
            obj_part = obj[obj_height - height_part:, obj_width - width_part:, :]
            mask_part = mask[obj_height - height_part:, obj_width - width_part:]
            y = 0
    

    # Check the max size of background image and adjust the size of object to paste 
    max_h, max_w = res_img[y:y + height_part, x:x + width_part, :].shape[:2]
    paste_area_mask =  paste_area_mask[:max_h, :max_w, :]
    obj_part = obj_part[:max_h, :max_w, :]
    mask_part = mask_part[:max_h, :max_w]

    res_img[y:y + height_part, x:x + width_part, :] \
        = res_img[y:y + height_part, x:x + width_part, :] * paste_area_mask + obj_part
    res_mask[y:y + height_part, x:x + width_part] \
        = res_mask[y:y + height_part, x:x + width_part] * paste_area_mask[:, :, 0] + mask_part
    return res_img, res_mask


#-----------------------MONTAGE-METHOD--------------------------


def build_montage(img_square_s, img_square_l, img_tall, img_wide, mask=False):
    """Generate a montage image with size of 224x224

    Args:
        img_square_s: input image will be resize to 
            small square image with size 64x64
        img_square_l: input image will be resize to 
            large square image with size 64x64160x160
        img_tall: input image will be resize to 
            tall image with size 160x64 
        img_wide: input image will be resize to 
            wide square image with size 64x160
        mask: False if inputs are images, 
            True if inputs are segmentation masks

    Returns:
        montage_image: the montage image with channel=3 if mask=False,
            channel=1 of mask=True
    """
    # Start with black canvas to draw images onto
    if mask:
        montage_image = np.zeros(shape=(224, 224), dtype=np.uint8)
    else:
        montage_image = np.zeros(shape=(224, 224, 3), dtype=np.uint8)

    # Resize images for montage
    top_left_img = cv2.resize(img_square_s, (64, 64))
    top_right_img = cv2.resize(img_wide, (160, 64))
    bottom_left_img = cv2.resize(img_tall, (64, 160))
    bottom_right_img = cv2.resize(img_square_l, (160, 160))

    montage_image[0:64, 0:64] = top_left_img
    montage_image[64:, 0:64] = bottom_left_img
    montage_image[0:64:, 64:] = top_right_img
    montage_image[64:, 64:] = bottom_right_img

    return montage_image