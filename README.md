# Gaussian Splatting PyTorch Lightning Implementation
* <a href="#1-installation">Installation</a>
* <a href="#2-training">Training</a>
* <a href="#4-web-viewer">Web Viewer</a>
* <a href="https://github.com/yzslab/gaussian-splatting-lightning/releases">Changelog</a>
## Known issues
* Multi-GPU training can only be enabled after densification
## Features
* Multi-GPU/Node training (only after densification)
* Switch between diff-gaussian-rasterization and <a href="https://github.com/nerfstudio-project/gsplat">nerfstudio-project/gsplat</a>
* Multiple dataset types support
  * <a href="https://drive.google.com/drive/folders/1JDdLGDruGNXWnM1eqY1FNL9PlStjaKWi">Blender (nerf_synthetic)</a>
  * Colmap
  * <a href="https://github.com/google/nerfies?tab=readme-ov-file#datasets">Nerfies</a>
  * <a href="https://github.com/facebookresearch/NSVF?tab=readme-ov-file#dataset">NSVF (Synthetic only)</a>
  * <a href="https://city-super.github.io/matrixcity/">MatrixCity</a>
  * <a href="https://www.cs.ubc.ca/~kmyi/imw2020/data.html">PhotoTourism</a>
* <a href="#4-web-viewer">Interactive web viewer</a>
  * Load multiple models
  * Model transform
  * Scene editor
  * Video camera path editor
* Video renderer
* Load a large number of images without OOM
* Dynamic object mask
* Derived algorithms
  * Deformable Gaussians
    * <a href="#25-deformable-3d-gaussians">Deformable 3D Gaussians (2.5.)</a>
    * <a href="#43-load-model-trained-by-other-implementations">4D Gaussian (4.3.)</a> (Viewer Only)
  * <a href="#26-mip-splatting">Mip-Splatting (2.6.)</a>
  * <a href="#27-lightgaussian">LightGaussian (2.7.)</a>
  * <a href="#28-absgs--efficientgs">AbsGS / EfficientGS (2.8.)</a>
  * <a href="#29-2d-gaussian-splatting">2D Gaussian Splatting (2.9.)</a>
  * <a href="#210-segment-any-3d-gaussians">Segment Any 3D Gaussians (2.10.)</a>
  * Reconstruct a large scale scene with the partitioning strategy like <a href="https://vastgaussian.github.io/">VastGaussian</a> (see <a href="#211-reconstruct-a-large-scale-scene-with-the-partitioning-strategy-like-vastgaussian">2.11.</a> below)
  * <a href="#212-appearance-model">New Appearance Model (2.12.)</a>: improve the quality when images have various appearances
  * <a href="#213-3dgs-mcmc">3D Gaussian Splatting as Markov Chain Monte Carlo (2.13.)</a>
## 1. Installation
### 1.1. Clone repository

```bash
# clone repository
git clone --recursive https://github.com/yzslab/gaussian-splatting-lightning.git
cd gaussian-splatting-lightning
```

* If you forgot the `--recursive` options, you can run below git commands after cloning:

  ```bash
   git submodule sync --recursive
   git submodule update --init --recursive --force
  ```

### 1.2. Create virtual environment

```bash
# create virtual environment
conda create -yn gspl python=3.9 pip
conda activate gspl
```

### 1.3. Install PyTorch
* Tested on `PyTorch==2.0.1`
* You must install the one match to the version of your nvcc (nvcc --version)
* For CUDA 11.8

  ```bash
  pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
  ```

### 1.4. Install requirements

```bash
pip install -r requirements.txt
```

### 1.5. Install optional packages
* If you want to use nerfstudio-project/gsplat
  
    ```bash
    pip install git+https://github.com/yzslab/gsplat.git
    ```
  
  This command will install my modified version, which is required by LightGaussian and Mip-Splatting. If you do not need them, you can also install vanilla gsplat <a href="https://github.com/nerfstudio-project/gsplat/tree/v0.1.12">v0.1.12</a>.
  
* If you need SegAnyGaussian
  * gsplat (see command above)
  * `pip install hdbscan scikit-learn==1.3.2 git+https://github.com/facebookresearch/segment-anything.git`
  * <a href="https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md">facebookresearch/pytorch3d</a>
  * Download <a href="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth">ViT-H SAM model</a>, place it to the root dir of this repo.: `wget -O sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth`

## 2. Training
### 2.1. Basic command
```bash
python main.py fit \
    --data.path DATASET_PATH \
    -n EXPERIMENT_NAME
```
It can detect some dataset type automatically. You can also specify type with option `--data.type`. Possible values are: `colmap`, `blender`, `nsvf`, `nerfies`, `matrixcity`, `phototourism`.

<b>[NOTE]</b> By default, only checkpoint files will be produced on training end. If you need ply file in vanilla 3DGS's format (can be loaded by SIBR_viewer or some WebGL/GPU based viewer):
  * [Option 1]: Convert checkpoint file to ply: `python utils/ckpt2ply.py TRAINING_OUTPUT_PATH`, e.g.:
    * `python utils/ckpt2ply.py outputs/lego`
    * `python utils/ckpt2ply.py outputs/lego/checkpoints/epoch=300-step=30000.ckpt`
  * [Option 2]: Start training with option: `--model.save_ply true`
### 2.2. Some useful options
* Run training with web viewer
```bash
python main.py fit \
    --viewer \
    ...
```
* It is recommended to use config file `configs/blender.yaml` when training on blender dataset.
```bash
python main.py fit \
    --config configs/blender.yaml \
    ...
```
* With mask (colmap dataset only)
  * You may need to undistort mask images too: <a href="https://github.com/yzslab/gaussian-splatting-lightning/blob/main/utils/colmap_undistort_mask.py">utils/colmap_undistort_mask.py</a>
```bash
# the requirements of mask
#   * must be single channel
#   * zero(black) represent the masked pixel (won't be used to supervise learning)
#   * the filename of the mask file must be image filename + '.png', 
#     e.g.: the mask of '001.jpg' is '001.jpg.png'
--data.params.colmap.mask_dir MASK_DIR_PATH
```
* Use downsampled images (colmap dataset only)

You can use `utils/image_downsample.py` to downsample your images, e.g. 4x downsample: `python utils/image_downsample.py PATH_TO_DIRECTORY_THAT_STORE_IMAGES --factor 4`
```bash
# it will load images from `images_4` directory
--data.params.colmap.down_sample_factor 4
```
* Load large dataset without OOM
```bash
--data.params.train_max_num_images_to_cache 1024
```
### 2.3. Use <a href="https://github.com/nerfstudio-project/gsplat">nerfstudio-project/gsplat</a>
Make sure that command `which nvcc` can produce output, or gsplat will be disabled automatically.
```bash
python main.py fit \
    --config configs/gsplat.yaml \
    ...
```

### 2.4. Multi-GPU training
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

### 2.5. <a href="https://ingra14m.github.io/Deformable-Gaussians/">Deformable 3D Gaussians</a>
<video src="https://github.com/yzslab/gaussian-splatting-lightning/assets/564361/177b3fbf-fdd2-490f-b446-433a4d929502"></video>

```bash
python main.py fit \
    --config configs/deformable_blender.yaml \
    --data.path ...
```

### 2.6. <a href="https://niujinshuchong.github.io/mip-splatting/">Mip-Splatting</a>
```bash
python main.py fit \
    --config configs/mip_splatting_gsplat.yaml \
    --data.path ...
```

### 2.7. <a href="https://lightgaussian.github.io/">LightGaussian</a>
* Prune & finetune only currently
* Train & densify & prune

  ```bash
  ... fit \
      --config configs/light_gaussian/train_densify_prune-gsplat.yaml \
      --data.path ...
  ```

* Prune & finetune (make sure to use the same hparams as the input model used)

  ```bash
  ... fit \
      --config configs/light_gaussian/prune_finetune-gsplat.yaml \
      --data.path ... \
      ... \
      --ckpt_path YOUR_CHECKPOINT_PATH
  ```
  
### 2.8. <a href="https://ty424.github.io/AbsGS.github.io/">AbsGS</a> / EfficientGS
```bash
... fit \
    --config configs/gsplat-absgrad.yaml \
    --data.path ...
```

### 2.9. <a href="https://surfsplatting.github.io/">2D Gaussian Splatting</a>
* Install `diff-surfel-rasterization` first
  ```bash
  pip install git+https://github.com/hbb1/diff-surfel-rasterization.git@3a9357f6a4b80ba319560be7965ed6a88ec951c6
  ```

* Then start training
  ```bash
  ... fit \
      --config configs/vanilla_2dgs.yaml \
      --data.path ...
  ```
  
### 2.10. <a href="https://jumpat.github.io/SAGA/">Segment Any 3D Gaussians</a>
* First, train a 3DGS scene using gsplat
  ```bash
  python main.py fit \
      --config configs/gsplat.yaml \
      --data.path data/Truck \
      -n Truck -v gsplat  # trained model will save to `outputs/Truck/gsplat`
  ```
* Then generate SAM masks and their scales
  * Masks
    ```bash
    python utils/get_sam_masks.py data/Truck/images
    ```
    You can specify the path to SAM checkpoint via argument `-c PATH_TO_SAM_CKPT`
  
  * Scales
    ```bash
    python utils/get_sam_mask_scales.py outputs/Truck/gsplat
    ```
  
  Both the masks and scales will be saved in `data/Truck/semantics`, the structure of `data/Truck` will like this:
  ```bash
  ├── images  # The images of your dataset
      ├── 000001.jpg
      ├── 000002.jpg
      ...
  ├── semantic  # Generate by `get_sam_masks.py` and `get_sam_mask_scales.py`
      ├── masks
          ├── 000001.jpg.pt
          ├── 000002.jpg.pt
          ...
      └── scales
          ├── 000001.jpg.pt
          ├── 000002.jpg.pt
          ...
  ├── sparse  # colmap sparse database
      ...
  ```

* Train SegAnyGS
  ```bash
  python seganygs.py fit \
      --config configs/segany_splatting.yaml \
      --data.path data/Truck \
      --model.initialize_from outputs/Truck/gsplat \
      -n Truck -v seganygs  # save to `outputs/Truck/seganygs`
  ```
  The value of `--model.initialize_from` is the path to the trained 3DGS model

* Start the web viewer to perform segmentation or cluster
  ```bash
  python viewer.py outputs/Truck/seganygs
  ```
  <video src="https://github.com/yzslab/gaussian-splatting-lightning/assets/564361/0b98a8ed-77d7-436d-b9f8-c5b51af5ba52"></video>

### 2.11. Reconstruct a large scale scene with the partitioning strategy like <a href="https://vastgaussian.github.io/">VastGaussian</a>
| Baseline | Partitioning |
| --- | --- |
| ![image](https://github.com/yzslab/gaussian-splatting-lightning/assets/564361/d3cb7d1a-f319-4315-bfa3-b56e3a98b19e) | ![image](https://github.com/yzslab/gaussian-splatting-lightning/assets/564361/12f930ee-eb5d-41c6-9fb7-6d043122a91c) |
| ![image](https://github.com/yzslab/gaussian-splatting-lightning/assets/564361/cec1bb13-15c0-4c6b-8d33-83bc21f2160e) | ![image](https://github.com/yzslab/gaussian-splatting-lightning/assets/564361/6bfd0130-29be-401f-ac9f-ce07dffe9fdd) |

There is no single script to finish the whole pipeline. Please refer to below contents about how to reconstruct a large scale scene.
* Partitioning
  * MatrixCity: <a href="https://github.com/yzslab/gaussian-splatting-lightning/blob/main/notebooks/matrix_city_aerial_split.ipynb">notebooks/matrix_city_aerial_split.ipynb</a>
  * Colmap: <a href="https://github.com/yzslab/gaussian-splatting-lightning/blob/main/notebooks/colmap_aerial_split.ipynb">notebooks/colmap_aerial_split.ipynb</a>
* Training
  * MatrixCity: Included in its partitioning notebook
  * Colmap: <a href="https://github.com/yzslab/gaussian-splatting-lightning/blob/main/utils/train_colmap_partitions.py">utils/train_colmap_partitions.py</a>
* Optional LightGaussian pruning
  * Pruning: <a href="https://github.com/yzslab/gaussian-splatting-lightning/blob/main/notebooks/partition_light_gaussian_pruning.ipynb">notebooks/partition_light_gaussian_pruning.ipynb</a>
  * Finetune after pruning: <a href="https://github.com/yzslab/gaussian-splatting-lightning/blob/main/utils/finetune_partition.py">utils/finetune_partition.py</a>
* Merging: <a href="https://github.com/yzslab/gaussian-splatting-lightning/blob/main/notebooks/merge_partitions.ipynb">notebooks/merge_partitions.ipynb</a>


### 2.12. Appearance Model
With appearance model, the reconstruction quality can be improved when your images have various appearance, such as different exposure, white balance, contrast and even day and night.

This model assign an extra feature vector $\boldsymbol{\ell}^{(g)}$ to each 3D Gaussian and an appearance embedding vector $\boldsymbol{\ell}^{(a)}$ to each appearance group. Both of them will be used as the input of a lightweight MLP to calculate the color.

$$ \mathbf{C} = f \left ( \boldsymbol{\ell}^{(g)}, \boldsymbol{\ell}^{(a)} \right ) $$

Please refer to <a href="https://github.com/yzslab/gaussian-splatting-lightning/blob/main/internal/renderers/gsplat_appearance_embedding_renderer.py">internal/renderers/gsplat_appearance_embedding_renderer.py</a> for more details.
  
| Baseline | New Model |
| --- | --- |
| <video src="https://github.com/yzslab/gaussian-splatting-lightning/assets/564361/3a990247-b57b-4ba8-8e9d-7346a3bd41e3"></video> | <video src="https://github.com/yzslab/gaussian-splatting-lightning/assets/564361/afeea69f-ed74-4c50-843a-e5d480eb66ef"></video> |
|  | <video src="https://github.com/yzslab/gaussian-splatting-lightning/assets/564361/ab89e4cf-80c0-4e99-88bc-3ec5ca047e19"></video> |
* First generate appearance groups (Colmap or PhotoTourism dataset only)
```bash
python utils/generate_image_apperance_groups.py PATH_TO_DATASET_DIR \
    --image \
    --name appearance_image_dedicated  # the name will be used later
```
The images in a group will share a common appearance embedding. The command above will assign each image a group, which means that will not share any appearance embedding between images.

* Then start training
```bash
python main.py fit \
    --config configs/appearance_embedding_renderer/view_dependent.yaml \
    --data.path PATH_TO_DATASET_DIR \
    --data.params.colmap.appearance_groups appearance_image_dedicated  # value here should be the same as the one provided to `--name` above
```
If you are using PhotoTourism dataset, please replace `--data.params.colmap.` with `--data.params.phototourism.`, and specify the dataset type with `--data.type phototourism`.

### 2.13. <a href="https://ubc-vision.github.io/3dgs-mcmc/">3DGS-MCMC</a>
* Install `submodules/mcmc_relocation` first

```bash
pip install submodules/mcmc_relocation
```

* Then training

```bash
... fit \
    --config configs/gsplat-mcmc.yaml \
    --model.density.cap_max MAX_NUM_GAUSSIANS \
    ...
```
`MAX_NUM_GAUSSIANS` is the maximum number of Gaussians that will be used.
  
Refer to <a href="https://github.com/ubc-vision/3dgs-mcmc">ubc-vision/3dgs-mcmc</a>, <a href="https://github.com/yzslab/gaussian-splatting-lightning/tree/main/internal/density_controllers/mcmc_density_controller.py">internal/density_controllers/mcmc_density_controller.py</a> and <a href="https://github.com/yzslab/gaussian-splatting-lightning/tree/main/internal/metrics/mcmc_metrics.py">internal/metrics/mcmc_metrics.py</a> for more details.

## 3. Evaluation

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

### On train set
```bash
python main.py validate \
    --config outputs/lego/config.yaml \
    --val_train
```

### Save images that rendered during evaluation/test
```bash
python main.py <validate or test> \
    --config outputs/lego/config.yaml \
    --save_val
```
Then you can find the images in `outputs/lego/<val or test>`.

## 4. Web Viewer
| Transform | Camera Path | Edit |
| --- | --- | --- |
| <video src="https://github.com/yzslab/gaussian-splatting-lightning/assets/564361/de1ff3c3-a27a-4600-8c76-ab6551df6fca"></video> | <video src="https://github.com/yzslab/gaussian-splatting-lightning/assets/564361/3f87243d-d9a1-41e2-9d51-225735925db4"></video> | <video src="https://github.com/yzslab/gaussian-splatting-lightning/assets/564361/7cf0ccf2-44e9-4fc9-87cc-740b7bbda488"></video> |


### 4.1 Basic usage
* Also works for <a href="https://github.com/graphdeco-inria/gaussian-splatting">graphdeco-inria/gaussian-splatting</a>'s ply output
```bash
python viewer.py TRAINING_OUTPUT_PATH
# e.g.: 
#   python viewer.py outputs/lego/
#   python viewer.py outputs/lego/checkpoints/epoch=300-step=30000.ckpt
#   python viewer.py outputs/lego/baseline/point_cloud/iteration_30000/point_cloud.ply  # only works with VanillaRenderer
```
### 4.2 Load multiple models and enable transform options
```bash
python viewer.py \
    outputs/garden \
    outputs/lego \
    outputs/Synthetic_NSVF/Palace/point_cloud/iteration_30000/point_cloud.ply \
    --enable_transform
```

### 4.3 Load model trained by other implementations
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

* <a href="https://github.com/hbb1/2d-gaussian-splatting">hbb1/2d-gaussian-splatting</a>
```bash
# Install `diff-surfel-rasterization` first
pip install git+https://github.com/hbb1/diff-surfel-rasterization.git@28c928a36ea19407cd9754d068bd9a9535216979
# Then start viewer
python viewer.py \
    2d-gaussian-splatting/outputs/Truck \
    --vanilla_gs2d
```

* <a href="https://github.com/Jumpat/SegAnyGAussians">Jumpat/SegAnyGAussians</a>
```bash
python viewer.py \
    SegAnyGAussians/outputs/Truck \
    --vanilla_seganygs
```

* <a href="https://github.com/autonomousvision/mip-splatting">autonomousvision/mip-splatting</a>
```bash
python viewer.py \
    mip-splatting/outputs/bicycle \
    --vanilla_mip
```

## 5. F.A.Q.
<b>Q: </b> The viewer shows my scene in unexpected orientation, how to rotate the camera, like the `U` and `O` key in the SIBR_viewer?

<b>A: </b> Check the `Orientation Control` on the right panel, rotate the camera frustum in the scene to the orientation you want, then click `Apply Up Direction`.
<video src="https://github.com/yzslab/gaussian-splatting-lightning/assets/564361/7e9198b5-d853-4800-aac2-1774640a8874"></video>

<br/>

Besides: You can also click the 'Reset up direction' button. Then the viewer will use your current orientation as the reference.
 * First use mouse to rotate your camera to the orientation you want
 * Then click the 'Reset up direction' button


##

<b>Q: </b> The web viewer is slow (or low fps, far from real-time).

<b>A: </b> This is expected because of the overhead of the image transfer over network. You can get around 10fps in 1080P resolution, which is enough for you to view the reconstruction quality.

