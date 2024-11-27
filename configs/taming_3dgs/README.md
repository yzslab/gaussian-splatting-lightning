# Usage
## 1. Install libraries
```bash
pip install \
    git+https://github.com/yzslab/diff-gaussian-rasterization.git@b403ab6c5cfb4ed89265a9759bd4766f9c4b56de \
    git+https://github.com/rahul-goel/fused-ssim.git@d99e3d27513fa3563d98f74fcd40fd429e9e9b0e
```
## 2. Training
```bash
python main.py fit \
    --config configs/taming_3dgs/rasterizer-fused_ssim.yaml \
    ...
```