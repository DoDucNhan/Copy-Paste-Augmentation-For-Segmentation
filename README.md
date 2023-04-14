# Copy-Paste-Augmentation-For-Segmentation

## Dependencies

1. Install mmcv and mmsegmentation toolbox

```bash
# Install MMCV
pip install openmim
mim install mmcv-full
pip install git+https://github.com/open-mmlab/mmsegmentation.git
```

2. Install libraries

```bash
pip install -r requirements.txt
```

## Run project

1. Crawl images

```bash
python src/crawl.py --dir <path to saved folder> --num <number of images> --platfomr <google/flickr> --keywords <object keywords>
```

2. Object segmentation

```bash
python src/segment.py --dir <path to crawl images folder> --out <output folder> --alpha <threshold for pseudo label>
```

3. Copy Paste method

```bash
python src/copy-paste.py --input <path to ADE20K folder> --segmentation <path to segmented images> --out <output folder> --num <number of additional object of each class>
```

4. Montage method

```bash
python src/montage.py --input <path to copy-paste images> --out <output folder>
```

***Note:*** The resolution for montage images can be changed by uncommenting the code in the `montage.py` script.

## Model traing

The working environment for model training is entirely on Google Colab. The training process is in this [notebook](https://colab.research.google.com/drive/1bPLZKN3hyWlGc9MC1UFfx-EzGv-4B4LY?usp=sharing). The training instruction for mmsegmentation toolbox can be found in this [link](https://mmsegmentation.readthedocs.io/en/main/)