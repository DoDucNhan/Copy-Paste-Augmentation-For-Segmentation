from mmseg.apis import init_segmentor, inference_segmentor
from mmseg.core.evaluation import get_classes
from utils import crop_object
import argparse
import cv2
import os


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dir", type=str, default='crawl_images',
    help="path to the downloaded images")
parser.add_argument("-o", "--out", type=str, default='segmented_object',
	help="path to output directory for segmented objects")

args = vars(parser.parse_args())

config_file = 'configs/beit/upernet_beit-large_fp16_8x1_640x640_160k_ade20k.py'
checkpoint_file = 'checkpoints/beit/upernet_beit-large_fp16_8x1_640x640_160k_ade20k-8fc0dd5d.pth'



if __name__ == "__main__":
    model = init_segmentor(config_file, checkpoint_file, device='cuda:0')
    classes = get_classes('ade20k')
    class2label = {cls: i for i, cls in enumerate(classes)}

    # Get all image filenames in every object folder
    for obj_dir in os.listdir(args['dir']):
        img_paths = []
        img_list = []
        save_path = os.path.join(args['out'], obj_dir)
        for filename in os.listdir(obj_dir):
            img_path = os.path.join(args['dir'], obj_dir, filename)
            image = cv2.imread(img_path)
            img_paths.append(img_path)
            img_list.append(image)
            
        
        results = inference_segmentor(model, img_path)
        count = 0
        for i, seg_result in enumerate(results):
            item_mask = seg_result == class2label[obj_dir]
            # Skip if no object found in image
            if (~item_mask).all():
                continue
            
            cropped_images, cropped_masks = crop_object(img_list[i], seg_result, class2label[obj_dir], 2)
            for i in range(cropped_images):
                count += 1
                filename = f"{count:0>4}.jpg"
                img_path = os.path.join(save_path, "image", filename)
                mask_path = os.path.join(save_path, "mask", filename)
                cv2.imwrite(img_path, cropped_images[i])
                cv2.imwrite(mask_path, cropped_masks[i])