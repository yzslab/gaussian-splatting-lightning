# Gaussian Splatting PyTorch Lightning Implementation
## Known issues
* Multi-GPU training can only be enabled after densification
## Features
* Multi-GPU/Node training (only after densification)
* Dynamic object mask
* Appearance variation support
* Deformable Gaussians
  * <a href="https://ingra14m.github.io/Deformable-Gaussians/">Deformable 3D Gaussians</a>
  * <a href="https://guanjunwu.github.io/4dgs/index.html">4D Gaussian</a> (Viewer Only)
* Load arbitrary number of images without OOM
* Interactive web viewer
  * Load multiple models
  * Model transform
  * Scene editor
  * Video camera path editor
* Video renderer
## Installation
```bash
# clone repository
git clone --recursive https://github.com/yzslab/gaussian-splatting-lightning.git
cd gaussian-splatting-lightning
# if you forgot the `--recursive` options, you can run below git commands after cloning:
#   git submodule sync --recursive
#   git submodule update --init --recursive --force


# create virtual environment
conda create -yn gspl python=3.9 pip
conda activate gspl

# install the PyTorch first, you must install the one match to the version of your nvcc (nvcc --version)
# for cuda 11.7
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
# for cuda 11.8
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# install other requirements
pip install -r requirements.txt
# optional one, you can skip this one unless you want to train with appearance variation images
pip install ./submodules/tiny-cuda-nn-fp32/bindings/torch
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
```bash
--data.params.colmap.mask_dir MASK_DIR_PATH
```
* Load large dataset without OOM
```bash
--data.params.train_max_num_images_to_cache 1024
```
* Enable appearance model to train on appearance variation images
```bash
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
<b>[NOTE]</b> Multi-GPU training can only be enabled after densification. You can start a single GPU training at the beginning, and save a checkpoint after densification finishing. Then resume from this checkpoint and enable multi-GPU training.

You will get improved PSNR and SSIM with more GPUs:
![image](https://github.com/yzslab/gaussian-splatting-lightning/assets/564361/06e91e71-5068-46ce-b169-524a069609bf)


```bash
# Single GPU at the beginning
python main.py fit \
    --config ... \
    --data.path DATASET_PATH \
    --model.gaussian.optimization.densify_until_iter 15000 \
    --max_steps 15000
# Then resume, and enable multi-GPU
python main.py fit \
    --config ... \
    --trainer configs/ddp.yaml \
    --data.path DATASET_PATH \
    --max_steps 30000 \
    --ckpt_path last  # find latest checkpoint automatically, or provide a path to checkpoint file
```

### <a href="https://ingra14m.github.io/Deformable-Gaussians/">Deformable 3D Gaussians</a>
<video src="https://github.com/yzslab/gaussian-splatting-lightning/assets/564361/177b3fbf-fdd2-490f-b446-433a4d929502"></video>

```bash
python main.py fit \
    --config configs/deformable_blender.yaml \
    --data.path ...
```

## Evaluation

### Evaluate on validation set
```bash
python main.py validate \
    --config outputs/lego/config.yaml
```

### On test set
```bash
python main.py test \
    --config outputs/lego/config.yaml
```

### Save images that rendered during evaluation
```bash
python main.py <validate or test> \
    --config outputs/lego/config.yaml \
    --model.save_val_output true
```
Then you can find the images in `outputs/lego/<val or test>`.

## Web Viewer
| Transform | Camera Path | Edit |
| --- | --- | --- |
| <video src="https://github.com/yzslab/gaussian-splatting-lightning/assets/564361/de1ff3c3-a27a-4600-8c76-ab6551df6fca"></video> | <video src="https://github.com/yzslab/gaussian-splatting-lightning/assets/564361/3f87243d-d9a1-41e2-9d51-225735925db4"></video> | <video src="https://github.com/yzslab/gaussian-splatting-lightning/assets/564361/7cf0ccf2-44e9-4fc9-87cc-740b7bbda488"></video> |


### Base
* Also works for <a href="https://github.com/graphdeco-inria/gaussian-splatting">graphdeco-inria/gaussian-splatting</a>'s ply output
```bash
python viewer.py TRAINING_OUTPUT_PATH
# e.g.: 
#   python viewer.py outputs/lego/
#   python viewer.py outputs/lego/checkpoints/epoch=300-step=30000.ckpt
#   python viewer.py outputs/lego/baseline/point_cloud/iteration_30000/point_cloud.ply  # only works with VanillaRenderer
```
### Load multiple models and enable transform options
```bash
python viewer.py \
    outputs/garden \
    outputs/lego \
    outputs/Synthetic_NSVF/Palace/point_cloud/iteration_30000/point_cloud.ply \
    --enable_transform
```

### Load model trained by other implementations
<b>[NOTE]</b> The commands in this section only design for third-party outputs

* <a href="https://github.com/ingra14m/Deformable-3D-Gaussians">ingra14m/Deformable-3D-Gaussians</a>

```bash
python viewer.py \
    Deformable-3D-Gaussians/outputs/lego \
    --vanilla_deformable \
    --reorient disable  # change to enable when loading real world scene
```

* <a href="https://github.com/hustvl/4DGaussians">hustvl/4DGaussians</a>

```bash
python viewer.py \
    4DGaussians/outputs/lego \
    --vanilla_gs4d
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
