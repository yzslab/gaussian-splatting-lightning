# Gaussian Splatting PyTorch Lightning Implementation
## Known issues
* Multi-GPU training not work correctly
  * GPUs do not have a consistent gaussian model after densification
  * Slower than single GPU
## Features
* <del>Multi-GPU/Node training</del>
* Dynamic object mask
* Appearance variation support
* Load arbitrary number of images without OOM
* Interactive web viewer
## Installation
```bash
# create virtual environment
conda create -yn gspl python=3.9 pip
conda activate gspl

# install requirements
pip install -r requirements-first.txt
pip install -r requirements-second.txt
```

## Training
### Colmap Dataset
* Base
```bash
python main.py fit \
    --data.path DATASET_PATH \
    -n EXPERIMENT_NAME
```
* With mask 
```
--data.params.colmap.mask_dir MASK_DIR_PATH
```
* Load large dataset without OOM
```
--data.params.train_max_num_images_to_cache 1024
```
* Enable appearance model to train on appearance variation images
```
# 1. Generate appearance groups
python generate_image_apperance_groups.py PATH_TO_DATASET \
    --camera \
    --name appearance_group_by_camera
    
# 2. Enable appearance model
python main.py fit \
    ... \
    --model.renderer AppearanceMLPRenderer \
    --data.params.colmap.appearance_groups appearance_group_by_camera \
    ...
```
### Blender Dataset
<b>[IMPORTANT]</b> Use config file `configs/blender.yaml` when training on blender dataset.
```bash
python main.py fit \
    --config configs/blender.yaml \
    --data.path DATASET_PATH \
    -n EXPERIMENT_NAME
```
### Multi-GPU training
```bash
python main.py fit \
    --config configs/blender.yaml \
    --trainer configs/ddp.yaml \
    --data.path DATASET_PATH
```

## Web Viewer
```bash
# replace CEHCKPOINT_PATH to your ckpt file path
python viewer.py CEHCKPOINT_PATH start
# e.g.: python viewer.py outputs/lego/lightning_logs/version_0/checkpoints/epoch=299-step=30000.ckpt start
```

# License
This repository is licensed under MIT license. Except some thirdparty dependencies (e.g. files in `submodules` directory), files and codes copied from other repositories, which are separately licensed.
```text
MIT License

Copyright (c) 2023 yzslab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```