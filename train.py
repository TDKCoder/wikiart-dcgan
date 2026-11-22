"""Train a DCGAN (spectral-norm D) on preprocessed WikiArt arrays.

Outputs in runs/<name>/: config.json, losses.csv (per iteration), samples/epoch_XXX.png
(fixed noise, EMA generator), ckpt_XXX.pt (G + G_ema), last.pt (full state for --resume).
"""
import argparse
import copy
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from models import Discriminator, Generator, ema_update, init_weights


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--data", default=None, help="default: data/wikiart_{res}.npy")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--nz", type=int, default=128)
    ap.add_argument("--ngf", type=int, default=64)
    ap.add_argument("--ndf", type=int, default=64)
    ap.add_argument("--lr_g", type=float, default=2e-4)
    ap.add_argument("--lr_d", type=float, default=2e-4)
    ap.add_argument("--beta1", type=float, default=0.5)
    ap.add_argument("--beta2", type=float, default=0.999)
    ap.add_argument("--loss", choices=["hinge", "bce"], default="hinge")
    ap.add_argument("--n_dis", type=int, default=1, help="D steps per G step")
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--flip", type=int, default=1, help="random horizontal flips")
    ap.add_argument("--amp", type=int, default=1, help="bf16 autocast")
    ap.add_argument("--ckpt_every", type=int, default=10)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--limit", type=int, default=0, help="use only the first N images (debug)")
    ap.add_argument("--gpu_data_gb", type=float, default=3.0, help="keep dataset on GPU if smaller")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    return ap.parse_args()


def d_loss_fn(kind, real, fake):
    if kind == "hinge":
        return F.relu(1 - real).mean() + F.relu(1 + fake).mean()
    return F.softplus(-real).mean() + F.softplus(fake).mean()


def g_loss_fn(kind, fake):
    return -fake.mean() if kind == "hinge" else F.softplus(-fake).mean()


def main():
    args = get_args()
    dev = "cuda"
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    out = Path("runs") / args.name
    (out / "samples").mkdir(parents=True, exist_ok=True)

    # ---- data: whole dataset as uint8 NHWC, on GPU when it fits ----
    arr = np.load(args.data or f"data/wikiart_{args.res}.npy", mmap_mode="r")
    if args.limit:
        arr = arr[: args.limit]
    data = torch.from_numpy(np.array(arr))  # copy out of the read-only memmap
    data = data.to(dev) if data.nbytes / 1e9 < args.gpu_data_gb else data.pin_memory()
    n = len(data)
    iters_per_epoch = n // args.bs
    print(f"{n} images ({data.nbytes / 1e9:.2f} GB on {data.device}), {iters_per_epoch} iters/epoch")

    def get_batch(idx):
        x = data[idx.to(data.device)].to(dev, non_blocking=True)
        x = x.permute(0, 3, 1, 2).float().div_(127.5).sub_(1)
        if args.flip:
            mask = torch.rand(x.size(0), 1, 1, 1, device=dev) < 0.5
            x = torch.where(mask, x.flip(3), x)
        return x.contiguous(memory_format=torch.channels_last)

    # ---- models ----
    G = Generator(args.nz, args.ngf, args.res).to(dev)
    D = Discriminator(args.ndf, args.res).to(dev)
    G.apply(init_weights)
    D.apply(init_weights)
    G, D = G.to(memory_format=torch.channels_last), D.to(memory_format=torch.channels_last)
    G_ema = copy.deepcopy(G).eval().requires_grad_(False)
    opt_g = torch.optim.Adam(G.parameters(), args.lr_g, betas=(args.beta1, args.beta2))
    opt_d = torch.optim.Adam(D.parameters(), args.lr_d, betas=(args.beta1, args.beta2))
    print(f"G params {sum(p.numel() for p in G.parameters()) / 1e6:.2f}M, "
          f"D params {sum(p.numel() for p in D.parameters()) / 1e6:.2f}M")

    fixed_z = torch.randn(64, args.nz, device=dev, generator=torch.Generator(dev).manual_seed(1234))
    start_epoch, step = 0, 0
    if args.resume and (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=dev)
        G.load_state_dict(ck["G"]), D.load_state_dict(ck["D"]), G_ema.load_state_dict(ck["G_ema"])
        opt_g.load_state_dict(ck["opt_g"]), opt_d.load_state_dict(ck["opt_d"])
        start_epoch, step = ck["epoch"], ck["step"]
        torch.set_rng_state(ck["rng_cpu"].cpu()), torch.cuda.set_rng_state(ck["rng_cuda"].cpu())
        print(f"resumed from epoch {start_epoch}")
    else:
        (out / "config.json").write_text(json.dumps(vars(args), indent=2))

    log_f = open(out / "losses.csv", "a" if start_epoch else "w", newline="")
    log = csv.writer(log_f)
    if not start_epoch:
        log.writerow(["step", "epoch", "loss_d", "loss_g", "d_real", "d_fake"])
    amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(args.amp))
    pending = []  # per-iteration stats kept on GPU, flushed every log_every iters (avoids syncs)

    def flush():
        if pending:
            vals = torch.stack([torch.stack(p) for _, _, p in pending]).float().cpu().tolist()
            log.writerows([s, e, *v] for (s, e, _), v in zip(pending, vals))
            pending.clear()

    for epoch in range(start_epoch, args.epochs):
        G.train()
        t0 = time.time()
        perm = torch.randperm(n, device=dev)
        for it in range(iters_per_epoch):
            idx = perm[it * args.bs:(it + 1) * args.bs]
            # --- D step(s) ---
            for k in range(args.n_dis):
                real = get_batch(idx if k == 0 else torch.randint(0, n, (args.bs,), device=dev))
                z = torch.randn(args.bs, args.nz, device=dev)
                with amp:
                    fake = G(z)
                    d_real, d_fake = D(real), D(fake.detach())
                    loss_d = d_loss_fn(args.loss, d_real.float(), d_fake.float())
                opt_d.zero_grad(set_to_none=True)
                loss_d.backward()
                opt_d.step()
            # --- G step ---
            with amp:
                loss_g = g_loss_fn(args.loss, D(fake).float())
            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            opt_g.step()
            ema_update(G_ema, G, args.ema)

            step += 1
            pending.append((step, epoch + 1, [loss_d.detach(), loss_g.detach(),
                                              d_real.detach().float().mean(), d_fake.detach().float().mean()]))
            if step % args.log_every == 0:
                flush()
        flush()
        log_f.flush()

        torch.cuda.synchronize()
        dt = time.time() - t0
        with torch.no_grad():
            save_image(G_ema(fixed_z).float() * 0.5 + 0.5, out / "samples" / f"epoch_{epoch + 1:03d}.png", nrow=8)
        print(f"epoch {epoch + 1}/{args.epochs}  loss_d {loss_d.item():.3f}  loss_g {loss_g.item():.3f}  "
              f"{dt:.1f}s ({iters_per_epoch * args.bs / dt:.0f} img/s)", flush=True)

        e = epoch + 1
        state = {"G": G.state_dict(), "G_ema": G_ema.state_dict(), "args": vars(args), "epoch": e}
        if e % args.ckpt_every == 0 or e == args.epochs:
            torch.save(state, out / f"ckpt_{e:03d}.pt")
        torch.save({**state, "D": D.state_dict(), "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(),
                    "step": step, "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state()},
                   out / "last.pt")
    log_f.close()


if __name__ == "__main__":
    main()
