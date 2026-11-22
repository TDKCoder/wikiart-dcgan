"""Sample grids and latent interpolations from a checkpoint.

python sample.py runs/<name>/ckpt_100.pt --out report/figs/samples.png [--n 64] [--interp 8]
"""
import argparse

import torch
from torchvision.utils import save_image

from models import Generator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--nrow", type=int, default=8)
    ap.add_argument("--interp", type=int, default=0, help="rows of 10-step latent interpolations")
    ap.add_argument("--which", choices=["ema", "g"], default="ema")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cuda")
    a = ck["args"]
    G = Generator(a["nz"], a["ngf"], a["res"]).cuda().eval()
    G.load_state_dict(ck["G_ema" if args.which == "ema" else "G"])
    g = torch.Generator("cuda").manual_seed(args.seed)
    with torch.no_grad():
        imgs = G(torch.randn(args.n, G.nz, device="cuda", generator=g))
        save_image(imgs * 0.5 + 0.5, args.out, nrow=args.nrow, padding=1)
        if args.interp:
            z0, z1 = (torch.randn(args.interp, 1, G.nz, device="cuda", generator=g) for _ in range(2))
            t = torch.linspace(0, 1, 10, device="cuda").view(1, -1, 1)
            z = (z0 * (1 - t) + z1 * t).view(-1, G.nz)
            save_image(G(z) * 0.5 + 0.5, args.out.replace(".png", "_interp.png"), nrow=10, padding=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
