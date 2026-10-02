"""FineWeb as GPT-2 tokens (kjj0/fineweb10B-gpt2): download, then fixed-order batches.

    uv run data.py --shards 1        # val + 1 train shard, 200 MB each, into data/fineweb10B

Shard format: 256 int32 header (magic 20240520, version 1, token count), then uint16 tokens.
"""

import argparse
import urllib.request
from pathlib import Path

import numpy as np
import torch

URL = "https://huggingface.co/datasets/kjj0/fineweb10B-gpt2/resolve/main/{}"
DIR = Path(__file__).resolve().parent / "data" / "fineweb10B"
MAGIC, HEADER = 20240520, 256
VAL = "fineweb_val_000000.bin"


def load(path):
    """uint16 tokens of one shard, memory-mapped, after checking its header."""
    h = np.fromfile(path, dtype=np.int32, count=HEADER)
    if len(h) < 3 or h[0] != MAGIC or h[1] != 1:
        raise ValueError(f"{path}: not a FineWeb GPT-2 shard")
    if Path(path).stat().st_size != HEADER * 4 + 2 * int(h[2]):
        raise ValueError(f"{path}: truncated")
    return np.memmap(path, dtype=np.uint16, mode="r", offset=HEADER * 4, shape=(int(h[2]),))


def download(shards):
    DIR.mkdir(parents=True, exist_ok=True)
    for name in [VAL] + [f"fineweb_train_{i:06d}.bin" for i in range(1, shards + 1)]:
        path, url = DIR / name, URL.format(name)
        size = int(urllib.request.urlopen(urllib.request.Request(url, method="HEAD"))
                   .headers["Content-Length"])
        have = path.stat().st_size if path.exists() else 0
        if have > size:
            path.unlink()
            have = 0
        if have < size:                                  # resume a partial download
            req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
            with urllib.request.urlopen(req) as r, open(path, "ab") as f:
                while chunk := r.read(1 << 22):
                    f.write(chunk)
        load(path)
        print(f"{name}: {size / 1e6:.0f} MB, ok")


class Batches:
    """Fixed-order (x, y) batches. Rank r of w reads every w-th slice, the same for every method."""

    def __init__(self, split, batch, seq, rank=0, world=1, device="cuda"):
        names = [VAL] if split == "val" else sorted(p.name for p in DIR.glob("fineweb_train_*.bin"))
        if not names:
            raise FileNotFoundError(f"no {split} shards in {DIR}; run `uv run data.py` first")
        self.shards = [load(DIR / n) for n in names]
        self.batch, self.seq, self.rank, self.world, self.device = batch, seq, rank, world, device
        self.reset()

    def reset(self):
        self.shard, self.pos = 0, self.rank * self.batch * self.seq

    def next(self):
        n = self.batch * self.seq
        if self.pos + n + 1 > len(self.shards[self.shard]):
            self.shard = (self.shard + 1) % len(self.shards)
            self.pos = self.rank * n
        buf = torch.from_numpy(self.shards[self.shard][self.pos:self.pos + n + 1].astype(np.int64))
        self.pos += self.world * n
        x, y = buf[:-1].view(self.batch, self.seq), buf[1:].view(self.batch, self.seq)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--shards", type=int, default=1, help="train shards of 100M tokens each")
    download(p.parse_args().shards)
