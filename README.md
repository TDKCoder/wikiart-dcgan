# DCGAN on WikiArt

ECE 285 Homework 2: an unconditional GAN trained on the 81,444 paintings of the
[WikiArt dataset](https://www.kaggle.com/datasets/steubk/wikiart) at 64×64 resolution, evaluated
with FID and Inception Score using [PyTorch-Ignite](https://pytorch-ignite.ai/blog/gan-evaluation-with-fid-and-is/).

- **Generator:** DCGAN (transposed conv + BatchNorm + ReLU, `tanh` output), 128-d latent.
- **Discriminator:** DCGAN-style conv net with spectral normalization in every layer and no BatchNorm.
- **Training:** hinge loss, Adam (lr 2e-4, betas 0.5/0.999), batch size 128, bf16 autocast, EMA of generator weights.

## Files

| File | Purpose |
|---|---|
| `preprocess.py` | Decodes JPEGs straight from the Kaggle zip, center-crops, resizes, and saves `data/wikiart_{64,128}.npy` |
| `models.py` | Generator, spectral-norm discriminator, weight init, EMA update |
| `train.py` | Training loop. Logs per-iteration losses to CSV, saves per-epoch sample grids and checkpoints, supports `--resume` |
| `evaluate.py` | FID (2048-d pool features) and Inception Score with `ignite.metrics` in an Ignite `Engine` |
| `plots.py` | Loss, discriminator-output, and FID-vs-epoch figures |
| `sample.py` | Sample grids and latent-space interpolations from a checkpoint |
| `report/` | NeurIPS-format LaTeX report |

## Reproduce

Requires PyTorch ≥ 2.7 with CUDA, plus `pip install pytorch-ignite scipy pandas matplotlib kaggle`.

```bash
kaggle datasets download steubk/wikiart -p data/raw      # 31.4 GB
python preprocess.py                                     # ~1 min on 22 cores
python train.py --name dcgan64 --res 64 --epochs 100     # ~25 min on an RTX 5070 Laptop GPU
python evaluate.py runs/dcgan64/ckpt_*.pt --n 10000 --real_is
python plots.py dcgan64 --out report/figs
python sample.py runs/dcgan64/ckpt_100.pt --out report/figs/samples.png --interp 4
```

On systems whose system `libstdc++` is older than conda's, run `source env.sh` first.
