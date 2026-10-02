"""GPT-2 on FineWeb, one method per run, fixed settings; appends a row to <out>/train.csv.

    uv run train.py --preset tiny --wire int8ef --gemm mxfp4 --out results/x
    torchrun --standalone --nproc-per-node=gpu train.py --preset gpt2 --wire int16 --out results/x

Methods: --wire {fp32,bf16,int16,int8ef} (gradient all-reduce, wire.py) and
--gemm {dense,mxfp4} (block linears, fp4_linear.py). Model, data order, seed and
schedule are fixed, so runs differ only by method. Ranks come from the environment.
"""

import argparse
import contextlib
import csv
import math
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

import data
import fp4_linear
import wire

# Per-GPU batch; tokens = (1 GPU, several GPUs). No sweeps: every method gets these.
PRESETS = {
    "tiny": dict(layers=4, dim=256, heads=4, seq=256, batch=8, lr=1e-3, warmup=50,
                 tokens=(1_000_000, 1_000_000)),
    "gpt2": dict(layers=12, dim=768, heads=12, seq=1024, batch=16, lr=6e-4, warmup=100,
                 tokens=(20_000_000, 200_000_000)),
}
VOCAB = 50304                      # GPT-2's 50257, padded to a multiple of 64
EVAL_BATCHES = 20


class Block(nn.Module):
    def __init__(self, dim, heads, linear):
        super().__init__()
        self.heads = heads
        self.ln1, self.ln2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv, self.proj = linear(dim, 3 * dim, bias=False), linear(dim, dim, bias=False)
        self.fc, self.out = linear(dim, 4 * dim, bias=False), linear(4 * dim, dim, bias=False)

    def forward(self, x):
        b, t, c = x.shape
        q, k, v = (z.view(b, t, self.heads, -1).transpose(1, 2)
                   for z in self.qkv(self.ln1(x)).split(c, 2))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).reshape(b, t, c))
        return x + self.out(F.gelu(self.fc(self.ln2(x))))


class GPT(nn.Module):
    def __init__(self, layers, dim, heads, seq, linear):
        super().__init__()
        self.wte, self.wpe = nn.Embedding(VOCAB, dim), nn.Embedding(seq, dim)
        self.blocks = nn.ModuleList(Block(dim, heads, linear) for _ in range(layers))
        self.ln = nn.LayerNorm(dim)
        for name, p in self.named_parameters():
            if p.dim() >= 2:                               # residual outputs start smaller
                residual = name.endswith(("proj.weight", "out.weight"))
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * layers) if residual else 0.02)

    def forward(self, x, y):
        h = self.wte(x) + self.wpe(torch.arange(x.shape[1], device=x.device))
        for blk in self.blocks:
            h = blk(h)
        logits = self.ln(h) @ self.wte.weight.t()        # tied head, always dense
        return F.cross_entropy(logits.float().view(-1, VOCAB), y.view(-1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=PRESETS, default="tiny")
    p.add_argument("--wire", choices=wire.METHODS, default="fp32")
    p.add_argument("--gemm", choices=("dense", "mxfp4"), default="dense")
    p.add_argument("--out", default="results")
    p.add_argument("--tokens", type=int, help="token budget across all GPUs (default: preset's)")
    p.add_argument("--batch", type=int, help="override the preset (smoke tests)")
    p.add_argument("--steps", type=int, help="override the token budget (smoke tests)")
    a = p.parse_args()

    rank, world, dev = wire.init()
    cfg = PRESETS[a.preset]
    batch, seq = a.batch or cfg["batch"], cfg["seq"]
    tokens = a.tokens or cfg["tokens"][world > 1]
    steps = a.steps or max(1, tokens // (batch * seq * world))
    bf16 = dev.type == "cuda" and torch.cuda.get_device_capability(dev)[0] >= 8
    amp = torch.autocast("cuda", torch.bfloat16) if bf16 else contextlib.nullcontext()

    torch.manual_seed(0)                                   # same init for every method
    linear = nn.Linear if a.gemm == "dense" else fp4_linear.MXFP4Linear
    model = GPT(cfg["layers"], cfg["dim"], cfg["heads"], seq, linear).to(dev)
    params = list(model.parameters())
    if world > 1:
        for t in params:
            dist.broadcast(t.data, 0)
    flat = torch.zeros(sum(t.numel() for t in params), device=dev)
    off = 0
    for t in params:                                       # grads live in one flat buffer
        t.grad = flat[off:off + t.numel()].view_as(t)
        off += t.numel()
    sync = wire.Wire(a.wire, flat.numel(), world, rank)
    opt = torch.optim.AdamW([{"params": [t for t in params if t.dim() >= 2], "weight_decay": 0.1},
                             {"params": [t for t in params if t.dim() < 2], "weight_decay": 0.0}],
                            lr=cfg["lr"], betas=(0.9, 0.95), fused=dev.type == "cuda")
    train = data.Batches("train", batch, seq, rank, world, dev)
    val = data.Batches("val", batch, seq, rank, world, dev)

    def lr_at(s):
        if s < cfg["warmup"]:
            return cfg["lr"] * (s + 1) / cfg["warmup"]
        frac = (s - cfg["warmup"]) / max(1, steps - cfg["warmup"])
        return cfg["lr"] * (0.1 + 0.45 * (1 + math.cos(math.pi * frac)))

    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
              for _ in range(steps)] if dev.type == "cuda" else []
    curve = []
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for s in range(steps):
        x, y = train.next()
        flat.zero_()
        with amp:
            loss = model(x, y)
        loss.backward()
        if events:
            events[s][0].record()
        sync(flat)
        if events:
            events[s][1].record()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        for group in opt.param_groups:
            group["lr"] = lr_at(s)
        opt.step()
        if s % 10 == 0 or s == steps - 1:
            curve.append((s, (s + 1) * batch * seq * world, loss.item()))
    if dev.type == "cuda":
        torch.cuda.synchronize()
    secs = time.perf_counter() - t0

    model.eval()
    vl = torch.zeros((), device=dev)
    with torch.no_grad(), amp:
        for _ in range(EVAL_BATCHES):
            vl += model(*val.next()) / EVAL_BATCHES
    if world > 1:
        dist.all_reduce(vl)
        vl /= world

    if rank == 0:
        tokens = steps * batch * seq * world
        comm = sum(e0.elapsed_time(e1) for e0, e1 in events) / steps if events else 0.0
        row = dict(method=f"{a.wire} / {a.gemm}", wire=a.wire, gemm=a.gemm,
                   gemm_path="dense" if a.gemm == "dense" else fp4_linear.path(),
                   preset=a.preset, gpus=world, compute="bf16" if bf16 else "fp32",
                   tokens=tokens, steps=steps, val_loss=round(vl.item(), 4),
                   train_loss=round(curve[-1][2], 4), seconds=round(secs, 1),
                   tok_per_s=round(tokens / secs), ms_per_step=round(1e3 * secs / steps, 1),
                   comm_ms_per_step=round(comm, 2), wire_B_per_el=round(sync.bytes_per_el, 4),
                   peak_GB=round(torch.cuda.max_memory_allocated(dev) / 1e9, 2)
                   if dev.type == "cuda" else 0.0)
        out = Path(a.out)
        (out / "curves").mkdir(parents=True, exist_ok=True)
        with open(out / "curves" / f"{a.wire}-{a.gemm}.csv", "w", newline="") as f:
            csv.writer(f).writerows([("step", "tokens", "loss"), *curve])
        new = not (out / "train.csv").exists()
        with open(out / "train.csv", "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
        print(row, flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
