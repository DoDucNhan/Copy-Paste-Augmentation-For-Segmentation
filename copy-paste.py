from utils import paste_object
import os.path as osp
import argparse
import logging
import random
import json
import cv2
import os


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str, default='ADE20K',
    help="path to the ADE20K directory")
parser.add_argument("-s", "--segmentation", type=str, default='segmented_objects',
    help="path to the segmented object directory")
parser.add_argument("-o", "--out", type=str, default='augmented_images',
	help="path to ADE20K directory to apply copy-paste")
parser.add_argument("-n", "--num", type=int, default='100',
	help="number of object instances of each class want to augmentation")

args = vars(parser.parse_args())


def get_class_id():
    with open("class2label.json") as f:
        return json.load(f)


logging.basicConfig(filename="augmentation.log",
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    if not osp.exists(args['out']):
        os.makedirs(f"{args['out']}/images")
        os.makedirs(f"{args['out']}/masks")

    # Path for saving new images and masks
    out_img_path = f"{args['out']}/images"
    out_ann_path = f"{args['out']}/masks"

    class2label = get_class_id()
    # Get all image names from training set of ADE20K
    train_img_path = osp.join(args['input'], "images", "training")
    train_ann_path = osp.join(args['input'], "annotations", "training")
    image_names = os.listdir(train_img_path)
    count = len(image_names)

    # Get all image names from segmented objects directory
    for obj_dir in os.listdir(args['segmentation']):
        # print("\n**********\n" + obj_dir + "\n**********\n")
        obj_path = osp.join(args['segmentation'], obj_dir)
        obj_img_path = osp.join(obj_path, "images")
        obj_mask_path = osp.join(obj_path, "masks")
        obj_names = os.listdir(obj_img_path)
        num_files = len(obj_names)
        
        # Choose random image to be pasted
        random.shuffle(image_names)
        for img_name in image_names[:args['num']]:
            # Get image and mask name
            image_name = osp.join(train_img_path, img_name)
            mask_name = osp.join(train_ann_path, img_name.split('.')[0] + ".png")

            # Choose random object to paste
            idx = random.randint(0, num_files - 1)
            obj_img_name = osp.join(obj_img_path, obj_names[idx])
            obj_mask_name = osp.join(obj_mask_path, obj_names[idx].split('.')[0] + ".png")

            # Load background image and mask
            org_img = cv2.imread(image_name)
            org_mask = cv2.imread(mask_name, cv2.IMREAD_ANYDEPTH)
            # Load object image and mask
            obj_img = cv2.imread(obj_img_name)
            obj_mask = cv2.imread(obj_mask_name, cv2.IMREAD_ANYDEPTH)
            
            logger.info(f"image name: {image_name}")
            logger.info(f"object name: {obj_img_name}")
            
            # print(f"image name: {image_name}")
            # print(f"mask name: {mask_name}")
            # print(f"object name: {obj_img_name}")
            # print(f"object mask name: {obj_mask_name}")
            # print("--------------------------------")

            # Annotation masks range [0, 150], where 0 refers to "other objects". 
            ## Those pixels are not considered in the official evaluation.
            # Choose random position to paste
            x = random.randint(0, org_img.shape[1] - 1)
            y = random.randint(0, org_img.shape[0] - 1)
            new_img, new_mask = paste_object(org_img, org_mask, \
                obj_img, obj_mask, class2label[obj_dir] + 1, (x, y))

            count += 1
            filename = f"ADE_train_{count:0>8}"
            logger.info(f"save image name: {filename}")
            logger.info("--------------------------------")
            
            save_img_path = osp.join(out_img_path, filename + ".jpg")
            save_mask_path = osp.join(out_ann_path, filename + ".png")
            cv2.imwrite(save_img_path, new_img)
            cv2.imwrite(save_mask_path, new_mask)
        



