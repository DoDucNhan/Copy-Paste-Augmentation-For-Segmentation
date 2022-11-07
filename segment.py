from mmseg.apis import init_segmentor
from mmseg.core.evaluation import get_classes
from mmseg.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter
import torch.nn.functional as F
import matplotlib.pyplot as plt 
import numpy as np
import mmcv


config_file = 'configs/beit/upernet_beit-large_fp16_8x1_640x640_160k_ade20k.py'
checkpoint_file = 'checkpoints/beit/upernet_beit-large_fp16_8x1_640x640_160k_ade20k-8fc0dd5d.pth'

imgs = 'barrel_3.jpg'
device = 'cuda'

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



model = init_segmentor(config_file, checkpoint_file, device='cuda:0')


cfg = model.cfg
# build the data pipeline
test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
test_pipeline = Compose(test_pipeline)

# prepare data
data = []
imgs = imgs if isinstance(imgs, list) else [imgs]
for img in imgs:
    img_data = dict(img=img)
    img_data = test_pipeline(img_data)
    data.append(img_data)
data = collate(data, samples_per_gpu=len(imgs))
if next(model.parameters()).is_cuda:
    # scatter to specified GPU
    data = scatter(data, [device])[0]
else:
    data['img_metas'] = [i.data[0] for i in data['img_metas']]


new_probs = model.inference(data['img'][0], data['img_metas'][0], True)

output = F.softmax(new_probs, dim=1)
barrel = output[0, 111, :, :]
brl_cpu = barrel.cpu().detach().numpy()

threshold = 0.01
new_img = np.where(brl_cpu >= threshold, 1, 0)
plt.imshow(new_img, cmap='gray')