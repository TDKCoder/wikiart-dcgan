"""Decode WikiArt JPEGs straight from the Kaggle zip into square uint8 arrays.

Writes data/wikiart_{res}.npy (N, res, res, 3) for each --res, plus data/meta.npz with
the style label of every kept image. Decoding happens once here so training never
touches JPEGs.
"""
import argparse
import io
import time
import zipfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # some paintings are huge scans
_ZF = None
_RES = None


def _init(zip_path, res):
    global _ZF, _RES
    _ZF = zipfile.ZipFile(zip_path)
    _RES = res


def _process(name):
    try:
        im = Image.open(io.BytesIO(_ZF.read(name)))
        # JPEG DCT-domain downscale: much faster decode, still >= 2x the largest target size.
        im.draft("RGB", (2 * max(_RES), 2 * max(_RES)))
        im = im.convert("RGB")
        w, h = im.size
        s = min(w, h)
        left, top = (w - s) // 2, (h - s) // 2
        im = im.crop((left, top, left + s, top + s))
        return [np.asarray(im.resize((r, r), Image.LANCZOS), dtype=np.uint8) for r in _RES]
    except Exception as e:  # corrupt/truncated files are skipped
        return repr(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default="data/raw/wikiart.zip")
    ap.add_argument("--out", default="data")
    ap.add_argument("--res", type=int, nargs="+", default=[64, 128])
    ap.add_argument("--workers", type=int, default=22)
    ap.add_argument("--limit", type=int, default=0, help="only process the first N images (debug)")
    args = ap.parse_args()

    with zipfile.ZipFile(args.zip) as zf:
        names = sorted(n for n in zf.namelist() if n.lower().endswith((".jpg", ".jpeg", ".png")))
    if args.limit:
        names = names[: args.limit]
    styles = sorted({n.split("/")[-2] for n in names})
    style_idx = {s: i for i, s in enumerate(styles)}
    print(f"{len(names)} images, {len(styles)} styles")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    arrays = {
        r: np.lib.format.open_memmap(out / f"wikiart_{r}.npy.tmp", mode="w+", dtype=np.uint8,
                                     shape=(len(names), r, r, 3))
        for r in args.res
    }
    labels, kept_names, n, t0 = [], [], 0, time.time()
    with Pool(args.workers, initializer=_init, initargs=(args.zip, args.res)) as pool:
        for i, result in enumerate(pool.imap(_process, names, chunksize=64)):
            if isinstance(result, str):
                print(f"skip {names[i]}: {result}")
                continue
            for r, img in zip(args.res, result):
                arrays[r][n] = img
            labels.append(style_idx[names[i].split("/")[-2]])
            kept_names.append(names[i])
            n += 1
            if (i + 1) % 5000 == 0:
                print(f"{i + 1}/{len(names)}  {(i + 1) / (time.time() - t0):.0f} img/s", flush=True)

    for r, arr in arrays.items():
        arr.flush()
        final = np.lib.format.open_memmap(out / f"wikiart_{r}.npy", mode="w+", dtype=np.uint8,
                                          shape=(n, r, r, 3))
        final[:] = arr[:n]
        final.flush()
        del arr, final
        (out / f"wikiart_{r}.npy.tmp").unlink()
    np.savez(out / "meta.npz", labels=np.array(labels, dtype=np.int16), styles=np.array(styles),
             names=np.array(kept_names))
    print(f"kept {n}/{len(names)} images in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
