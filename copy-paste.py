from utils import paste_object
import argparse
import random
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


if __name__ == "__main__":
    # Get all image names from training set of ADE20K
    train_img_path = os.path.join(args['out'], "images", "training")
    train_ann_path = os.path.join(args['out'], "annotations", "training")
    image_names = os.listdir(train_img_path)

    # Get all image names from segmented objects directory
    for obj_dir in os.listdir(args['input']):
        print("\n**********\n" + obj_dir + "\n**********\n")
        obj_path = os.path.join(args['input'], obj_dir)
        obj_img_path = os.path.join(obj_path, "images")
        obj_mask_path = os.path.join(obj_path, "masks")
        obj_names = os.listdir(obj_img_path)
        num_files = len(obj_names)
        
        random.shuffle(image_names)
        for img_name in image_names[:args['num']]:
            # Get image name
            image_name = os.path.join(train_img_path, img_name)
            mask_name = os.path.join(train_ann_path, img_name)

            # Get object image name
            idx = random.randint(0, num_files - 1)
            obj_img_name = os.path.join(obj_img_path, obj_names[idx])
            obj_mask_name = os.path.join(obj_mask_path, obj_names[idx])

            print(f"image name: {image_name}")
            print(f"mask name: {mask_name}")
            print(f"object name: {obj_img_name}")
            print(f"object mask name: {obj_mask_name}")

            print("--------------------------------")




