# Gaussian Splatting PyTorch Lightning Implementation
## Features
* Multi-GPU/Node training
* Dynamic object mask
* Appearance variation support
* Load arbitrary number of images without OOM
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
--model.enable_appearance_model True
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
