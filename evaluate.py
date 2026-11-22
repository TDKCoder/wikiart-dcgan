"""FID and Inception Score with PyTorch-Ignite, following
https://pytorch-ignite.ai/blog/gan-evaluation-with-fid-and-is/ (Engine + ignite.metrics.FID/InceptionScore).

Changes from the blog:
- Images are mapped from [-1, 1] to [0, 1], upsampled to 299x299 bilinearly on the GPU, and
  ImageNet-normalized, which is the input torchvision's InceptionV3 expects. The blog's
  per-image PIL resize does the same resize but skips the range fix.
- Two FIDs are reported. `fid` is the standard one on 2048-d pool features. `fid_blog` uses
  Ignite's default FID(), which compares 1000-d softmax outputs. It matches the blog but is
  not comparable to published FID numbers.

Usage: python evaluate.py runs/<name>/ckpt_100.pt [more ckpts...] [--n 10000] [--which ema]
Results are appended to runs/<name>/metrics.jsonl.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from ignite.engine import Engine
from ignite.metrics import FID, InceptionScore
from ignite.metrics.gan.utils import InceptionModel

from models import Generator

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def to_inception(x):
    """[-1, 1] images (any size) -> 299x299 ImageNet-normalized."""
    x = (x.float() * 0.5 + 0.5).clamp(0, 1)
    x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
    return (x - MEAN.to(x.device)) / STD.to(x.device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--n", type=int, default=10000, help="number of fake (and real) images")
    ap.add_argument("--bs", type=int, default=100)
    ap.add_argument("--which", choices=["ema", "g"], default="ema")
    ap.add_argument("--real_is", action="store_true", help="also compute IS of the real images")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = "cuda"

    inc_prob = InceptionModel(return_features=False, device=dev)  # softmax over 1000 classes
    inc_feat = InceptionModel(return_features=True, device=dev)  # 2048-d pool features

    for ckpt_path in args.ckpts:
        ck = torch.load(ckpt_path, map_location=dev)
        a = ck["args"]
        G = Generator(a["nz"], a["ngf"], a["res"]).to(dev)
        G.load_state_dict(ck["G_ema" if args.which == "ema" else "G"])
        G.eval()

        # Same real subset and noise for every checkpoint, so scores are comparable across epochs.
        real_all = np.load(a.get("data") or f"data/wikiart_{a['res']}.npy", mmap_mode="r")
        rng = np.random.default_rng(args.seed)
        real_idx = np.sort(rng.choice(len(real_all), args.n, replace=False))
        gen = torch.Generator(dev).manual_seed(args.seed)
        batches = [real_idx[i:i + args.bs] for i in range(0, args.n, args.bs)]

        def eval_step(engine, idx):
            with torch.no_grad():
                real = torch.from_numpy(real_all[idx]).to(dev).permute(0, 3, 1, 2).float() / 127.5 - 1
                z = torch.randn(len(idx), G.nz, device=dev, generator=gen)
                fake = G(z)
                return to_inception(fake), to_inception(real)

        evaluator = Engine(eval_step)
        FID(num_features=2048, feature_extractor=inc_feat, device=dev).attach(evaluator, "fid")
        FID(num_features=1000, feature_extractor=inc_prob, device=dev).attach(evaluator, "fid_blog")
        InceptionScore(num_features=1000, feature_extractor=inc_prob, device=dev,
                       output_transform=lambda x: x[0]).attach(evaluator, "is")
        if args.real_is:
            InceptionScore(num_features=1000, feature_extractor=inc_prob, device=dev,
                           output_transform=lambda x: x[1]).attach(evaluator, "is_real")

        t0 = time.time()
        state = evaluator.run(batches, max_epochs=1)
        res = {"ckpt": str(ckpt_path), "epoch": ck["epoch"], "which": args.which, "n": args.n,
               **{k: float(v) for k, v in state.metrics.items()}, "seconds": round(time.time() - t0, 1)}
        print(json.dumps(res), flush=True)
        with open(Path(ckpt_path).parent / "metrics.jsonl", "a") as f:
            f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
