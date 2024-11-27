# Usage
## 1. Install libraries
```bash
pip install \
    git+https://github.com/yzslab/taming-3dgs-rasterizer.git \
    git+https://github.com/rahul-goel/fused-ssim.git@84422e0da94c516220eb3acedb907e68809e9e01
```
## 2. Training
```bash
python main.py fit \
    --config configs/taming_3dgs/rasterizer-fused_ssim.yaml \
    ...
```