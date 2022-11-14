from mmseg.apis import init_segmentor, inference_segmentor
from mmseg.core.evaluation import get_classes
from utils import crop_object
import argparse
import cv2
import os


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dir", type=str, default='crawl_images',
    help="path to the downloaded images")
parser.add_argument("-o", "--out", type=str, default='segmented_objects',
	help="path to output directory for segmented objects")

args = vars(parser.parse_args())

config_file = 'configs/beit/upernet_beit-large_fp16_8x1_640x640_160k_ade20k.py'
checkpoint_file = 'checkpoints/beit/upernet_beit-large_fp16_8x1_640x640_160k_ade20k-8fc0dd5d.pth'



if __name__ == "__main__":
    model = init_segmentor(config_file, checkpoint_file, device='cuda:0')
    classes = get_classes('ade20k')
    class2label = {cls: i for i, cls in enumerate(classes)}

    if not os.path.exists(args['out']):
            os.makedirs(args['out'])

    # Get all image filenames in every object folder
    for obj_dir in os.listdir(args['dir']):
        obj_path = os.path.join(args['dir'], obj_dir)
        save_path = os.path.join(args['out'], obj_dir)

        # Create save directory for segmented object
        out_image_path = os.path.join(save_path, "images")
        out_mask_path = os.path.join(save_path, "masks")
        if not os.path.exists(out_image_path):
            os.makedirs(out_image_path)
        if not os.path.exists(out_mask_path):
            os.makedirs(out_mask_path)

        count = 0 # counter for image name
        for filename in os.listdir(obj_path):
            img_path = os.path.join(obj_path, filename)
            image = cv2.imread(img_path)
            
            print(f"Image {img_path}")
            # Segmentation
            result = inference_segmentor(model, img_path)

            seg_result = result[0]
            item_mask = seg_result == class2label[obj_dir]
            # Skip if no object found in image
            if (~item_mask).all():
                print(f"Skipping image {img_path}")
                continue
            
            # Crop objects from image
            cropped_images, cropped_masks = crop_object(image, seg_result, class2label[obj_dir], 2)
            for i in range(len(cropped_images)):
                count += 1
                filename = f"{count:0>4}"
                image_name = os.path.join(out_image_path, filename + ".jpg")
                mask_name = os.path.join(out_mask_path, filename + ".png")
                # Save object image and its mask
                cv2.imwrite(image_name, cropped_images[i])
                cv2.imwrite(mask_name, cropped_masks[i])