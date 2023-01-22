from utils import build_montage
import os.path as osp
import argparse
import logging
import random
import json
import cv2
import os


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str, default='augmented_images',
    help="path to the ADE20K directory")
parser.add_argument("-o", "--out", type=str, default='montage_images',
	help="path to ADE20K directory to apply copy-paste")

args = vars(parser.parse_args())



logging.basicConfig(filename="montage.log",
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    if not osp.exists(args['out']):
        os.makedirs(f"{args['out']}/images")
        os.makedirs(f"{args['out']}/annotations")

    # Path for saving new images and masks
    out_img_path = f"{args['out']}/images"
    out_ann_path = f"{args['out']}/masks"

    # Get all image names from augmentation directory
    aug_img_path = osp.join(args['input'], "images")
    aug_ann_path = osp.join(args['input'], "masks")
    image_names = os.listdir(aug_img_path)
    
    # Get counter for naming images
    first_name = image_names[0].split('.')[0]
    count = int(first_name.split('_')[-1])

        
    # Choose random 4 images to create montage images
    ### Need adjustment of size of array not divided by 4
    random.shuffle(image_names)
    quad_name_list = [image_names[i: i+4] for i in range(0, len(image_names), 4)]
    for img_name in quad_name_list:
        imgs = []
        masks = []
        # Get image name
        for name in img_name:
            image_name = osp.join(aug_img_path, img_name)
            mask_name = osp.join(aug_ann_path, img_name.split('.')[0] + ".png")

            logger.info(f"image name: {image_name}")

            imgs.append(cv2.imread(image_name))
            masks.append(cv2.imread(mask_name, cv2.IMREAD_ANYDEPTH))

        img1, img2, img3, img4 = imgs
        mask1, mask2, mask3, mask4 = masks
        # Check size and sort images to order: tall, wide, square large, square small
        sorted_imgs = []
        sorted_masks = []
        for t in range(3):
            idx = 0
            max_val = 0
            for i in range(len(imgs)):
                img_h, img_w = imgs[i].shape[:2]
                diff = abs(img_h - img_w)
                if t == 0:
                    if img_h > img_w:
                        if diff > max_val:
                            idx = i
                            max_val = diff
                elif t == 1:
                    if img_w > img_h:
                        if diff > max_val:
                            idx = i
                            max_val = diff
                else:
                    area = img_h * img_w
                    if area > max_val:
                        idx = i
                        max_val = area
            
            sorted_imgs.append(imgs.pop(idx))
            sorted_masks.append(masks.pop(idx))
                
        # Append last image and mask
        sorted_imgs.append(imgs.pop(0))
        sorted_masks.append(masks.pop(0))
        

        new_img = build_montage(img1, img2, img3, img4)
        new_mask = build_montage(mask1, mask2, mask3, mask4, True)

        count += 1
        filename = f"ADE_train_{count:0>8}"
        logger.info(f"save image name: {filename}")
        logger.info("--------------------------------")
        save_img_path = osp.join(out_img_path, filename + ".jpg")
        save_mask_path = osp.join(out_ann_path, filename + ".png")
        cv2.imwrite(save_img_path, new_img)
        cv2.imwrite(save_mask_path, new_mask)
        



