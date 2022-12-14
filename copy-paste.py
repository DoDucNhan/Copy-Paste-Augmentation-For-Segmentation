from utils import paste_object
import logging
import argparse
import random
import json
import cv2
import os


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str, default='segmented_objects',
    help="path to the segmented object directory")
parser.add_argument("-o", "--out", type=str, default='ADE20K-custom',
	help="path to ADE20K directory to apply copy-paste")
parser.add_argument("-n", "--num", type=int, default='100',
	help="number of object instaces want to augmentation")

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
    class2label = get_class_id()
    # Get all image names from training set of ADE20K
    train_img_path = os.path.join(args['out'], "images", "training")
    train_ann_path = os.path.join(args['out'], "annotations", "training")
    image_names = os.listdir(train_img_path)

    # Get all image names from segmented objects directory
    for obj_dir in os.listdir(args['input']):
        # print("\n**********\n" + obj_dir + "\n**********\n")
        obj_path = os.path.join(args['input'], obj_dir)
        obj_img_path = os.path.join(obj_path, "images")
        obj_mask_path = os.path.join(obj_path, "masks")
        obj_names = os.listdir(obj_img_path)
        num_files = len(obj_names)
        
        # Choose random image to be pasted
        random.shuffle(image_names)
        for img_name in image_names[:args['num']]:
            # Get image name
            image_name = os.path.join(train_img_path, img_name)
            mask_name = os.path.join(train_ann_path, img_name.split('.')[0] + ".png")

            # Choose random object to paste
            idx = random.randint(0, num_files - 1)
            obj_img_name = os.path.join(obj_img_path, obj_names[idx])
            obj_mask_name = os.path.join(obj_mask_path, obj_names[idx])

            # Load background image and mask
            org_img = cv2.imread(image_name)
            org_mask = cv2.imread(mask_name, cv2.IMREAD_ANYDEPTH)
            # Load object image and mask
            obj_img = cv2.imread(obj_img_name)
            obj_mask = cv2.imread(obj_mask_name, cv2.IMREAD_ANYDEPTH)
            
            logger.info(f"image name: {image_name}")
            logger.info(f"object name: {obj_img_name}")
            logger.info("--------------------------------")
            
            # print(f"image name: {image_name}")
            # print(f"mask name: {mask_name}")
            # print(f"object name: {obj_img_name}")
            # print(f"object mask name: {obj_mask_name}")
            # print("--------------------------------")

            # Choose random position to paste
            x = random.randint(0, org_img.shape[1] - 1)
            y = random.randint(0, org_img.shape[0] - 1)
            new_img, new_mask = paste_object(org_img, org_mask, \
                obj_img, obj_mask, class2label[obj_dir], (x, y))

            cv2.imwrite(image_name, new_img)
            cv2.imwrite(mask_name, new_mask)
        



