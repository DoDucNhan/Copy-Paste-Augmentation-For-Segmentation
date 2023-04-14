from mmseg.apis import init_segmentor, inference_segmentor
from mmseg.core.evaluation import get_classes
from utils import crop_object, get_image_data
from scipy.stats import entropy
import os.path as osp
import numpy as np
import argparse
import logging
import torch
import cv2
import os


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dir", type=str, default='crawl_images',
    help="path to the downloaded images")
parser.add_argument("-o", "--out", type=str, default='segmented_objects',
	help="path to output directory for segmented objects")
parser.add_argument("-a", "--alpha", type=float, default=0.2,
	help="percentage of confident pixel prediction")

args = vars(parser.parse_args())

config_file = 'configs/beit/upernet_beit-large_fp16_8x1_640x640_160k_ade20k.py'
checkpoint_file = 'checkpoints/beit/upernet_beit-large_fp16_8x1_640x640_160k_ade20k-8fc0dd5d.pth'


logging.basicConfig(filename="segmentation.log",
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

logger = logging.getLogger(__name__)



if __name__ == "__main__":
    model = init_segmentor(config_file, checkpoint_file, device='cuda:0')
    classes = get_classes('ade20k')
    class2label = {cls: i for i, cls in enumerate(classes)}

    if not osp.exists(args['out']):
            os.makedirs(args['out'])

    # Get all image filenames in every object folder
    for obj_dir in os.listdir(args['dir']):
        obj_path = osp.join(args['dir'], obj_dir)
        save_path = osp.join(args['out'], obj_dir)

        # Create save directory for segmented object
        out_image_path = osp.join(save_path, "images")
        out_mask_path = osp.join(save_path, "masks")
        if not osp.exists(out_image_path):
            os.makedirs(out_image_path)
        if not osp.exists(out_mask_path):
            os.makedirs(out_mask_path)

        object_id = class2label[obj_dir]
        # Adjustments for 2 specific classes "step" and "land"
        is_step, is_land = False, False
        if obj_dir == "step":
            object_id = class2label["stairway"]
            is_step = True

        if obj_dir == "land":
            object_id = class2label["earth"]
            is_land = True

        count = 0 # counter for image name
        for filename in os.listdir(obj_path):
            img_path = osp.join(obj_path, filename)
            image = cv2.imread(img_path)
            
            logger.info(f"Image {img_path}")
            # Segmentation
            alpha = args['alpha']
            data = get_image_data(model, img_path)
            with torch.no_grad():
                class_softmax = model.inference(data['img'][0], data['img_metas'][0], True)
            output = class_softmax.argmax(dim=1)
            # Apply entropy threshold to keep confident pixels
            class_entropy = entropy(class_softmax.cpu(), axis=1)
            threshold = np.percentile(class_entropy.flatten(), 100 * (1 - alpha))
            refine = np.where(class_entropy[0] < threshold, 1, 0)
            refine_output = output.cpu() * refine
            object_mask = np.where(refine_output == object_id, object_id, 0)
            # Skip if object pixels is too little compared to whole image
            obj_pixel_ratio = np.sum(object_mask != 0) / (object_mask.shape[0] * object_mask.shape[1])
            if obj_pixel_ratio < 0.1:
                logger.info(f"Skipping image {img_path}")
                logger.info("--------------------------------")
                continue

            logger.info("--------------------------------")
            # Crop objects from image
            cropped_images, cropped_masks = crop_object(image, object_mask[0], object_id)
            for i in range(len(cropped_images)):
                count += 1
                filename = f"{count:0>4}"
                image_name = osp.join(out_image_path, filename + ".jpg")
                mask_name = osp.join(out_mask_path, filename + ".png")
                # Save object image and its mask
                cropped_image = cropped_images[i]
                cropped_mask = cropped_masks[i]
                # Annotation masks range [0, 150], where 0 refers to "other objects". 
                ## Those pixels are not considered in the official evaluation.
                cropped_mask = np.where(cropped_masks == object_id, object_id + 1, 0)
                # Deal with 2 specific classes: step and land by modifying from stairway and earth
                if is_step:
                    height, width = cropped_masks.shape
                    # Crop 1/3 bottom of stairway object to get step object
                    if height > 150:
                        crop_pos = int(height*2/3)
                        cropped_image = cropped_image[crop_pos:, :, :]
                        cropped_mask = cropped_mask[crop_pos:, :]

                    cropped_mask = np.where(cropped_masks == object_id, class2label['step'] + 1, 0)

                if is_land:
                    cropped_mask = np.where(cropped_masks == object_id, class2label['land'] + 1, 0)

                cv2.imwrite(image_name, cropped_image)
                cv2.imwrite(mask_name, cropped_mask)