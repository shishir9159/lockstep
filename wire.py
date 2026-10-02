"""Gradient all-reduce over four wires, and a microbenchmark that compares them.

  fp32    NCCL all_reduce in fp32 (the baseline)
  bf16    NCCL all_reduce in bf16, which re-rounds the partial sum at every hop
  int16   integers on a shared per-64 grid, sent once to each chunk's owner, summed
          exactly in int32, re-encoded by the owner and gathered back
  int8ef  the same at 8 bits, with error feedback at the sender and at the owner

    torchrun --standalone --nproc-per-node=gpu wire.py --out results/x   # all GPUs

With one process the same math runs without communication (rounding at the sender
and at the owner), so single-GPU training still sees each wire's error. Ranks come
from the environment (torchrun, SLURM or MPI variables).
"""

import argparse
import csv
import os
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist

CHUNK = 64                                     # elements per integer scale
METHODS = ("fp32", "bf16", "int16", "int8ef")


def env_rank():
    """(rank, world, local rank) from torchrun, SLURM or MPI variables; else one process."""
    def get(names, default):
        return next((int(os.environ[n]) for n in names if n in os.environ), default)
    return (get(("RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK"), 0),
            get(("WORLD_SIZE", "SLURM_NTASKS", "OMPI_COMM_WORLD_SIZE"), 1),
            get(("LOCAL_RANK", "SLURM_LOCALID", "OMPI_COMM_WORLD_LOCAL_RANK"), 0))


def init():
    """Join the process group if there is more than one rank; return (rank, world, device)."""
    rank, world, local = env_rank()
    cuda = torch.cuda.is_available()
    device = torch.device("cuda", local) if cuda else torch.device("cpu")
    if cuda:
        torch.cuda.set_device(device)
    if world > 1 and not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        dist.init_process_group("nccl" if cuda else "gloo", rank=rank, world_size=world)
    return rank, world, device


def _as_bytes(x):
    """NCCL has no int16; move int16 payloads as raw bytes."""
    return x.view(torch.uint8) if x.dtype == torch.int16 else x


def _all_to_all(x, world):
    if world == 1:
        return x
    out = torch.empty_like(x)
    dist.all_to_all_single(_as_bytes(out), _as_bytes(x))
    return out


def _all_gather(x, world):
    if world == 1:
        return x
    out = torch.empty((world * x.shape[0],) + tuple(x.shape[1:]), dtype=x.dtype, device=x.device)
    if dist.get_backend() == "nccl":
        dist.all_gather_into_tensor(_as_bytes(out), _as_bytes(x))
    else:
        dist.all_gather(list(_as_bytes(out).chunk(world)), _as_bytes(x))
    return out


class Wire:
    """Averages a flat fp32 gradient across ranks, in place, with one method."""

    def __init__(self, method, numel, world=1, rank=0):
        bits = {"int16": 16, "int8": 8, "int8ef": 8}.get(method)
        self.method, self.world, self.rank, self.n = method, world, rank, numel
        self.ef = method == "int8ef"
        self.lim = 2 ** (bits - 1) - 1 if bits else None
        self.dtype = {16: torch.int16, 8: torch.int8}.get(bits)
        span = world * CHUNK
        self.padded = -(-numel // span) * span
        self.res = self.ores = None                # sender and owner residuals (EF)
        # Element bytes on the wire; integer wires also carry one fp32 scale per chunk.
        self.bytes_per_el = {"fp32": 4.0, "bf16": 2.0}.get(method) or bits / 8 + 4 / CHUNK

    def __call__(self, g, reverse=False):
        if self.method == "fp32":
            if self.world > 1:
                dist.all_reduce(g)
            return g.div_(self.world)
        if self.method == "bf16":
            h = g.to(torch.bfloat16)
            if self.world > 1:
                dist.all_reduce(h)
            return g.copy_(h).div_(self.world)
        return self._integer(g, reverse)

    def _integer(self, g, reverse):
        world, lim = self.world, self.lim
        v = torch.zeros(self.padded, device=g.device)
        v[:self.n] = g
        if self.ef:
            if self.res is None:
                self.res = torch.zeros_like(v)
            v += self.res
        c = v.view(-1, CHUNK)
        m = c.abs().amax(1)
        if world > 1:
            dist.all_reduce(m, op=dist.ReduceOp.MAX)            # one grid shared by all ranks
        s = torch.where(m > 0, lim / m, torch.ones_like(m))
        q = torch.round(c * s[:, None]).clamp_(-lim, lim)
        if self.ef:
            self.res = v - (q / s[:, None]).view(-1)
        own = c.shape[0] // world                                 # chunks per owner
        parts = _all_to_all(q.to(self.dtype), world).view(world, own, CHUNK)
        if reverse:
            parts = parts.flip(0)
        val = parts.to(torch.int32).sum(0).float()                # exact in any order
        lo = self.rank * own
        val /= s[lo:lo + own, None]
        if self.ef:
            if self.ores is None:
                self.ores = torch.zeros_like(val)
            val += self.ores
        m2 = val.abs().amax(1)                                    # owner-local grid
        s2 = torch.where(m2 > 0, lim / m2, torch.ones_like(m2))
        q2 = torch.round(val * s2[:, None]).clamp_(-lim, lim)
        if self.ef:
            self.ores = val - q2 / s2[:, None]
        q2 = _all_gather(q2.to(self.dtype), world).view(-1, CHUNK)
        s2 = _all_gather(s2, world)
        return g.copy_((q2.float() / s2[:, None]).view(-1)[:self.n]).div_(world)


def bench(out, numel, iters):
    """Time and error of each wire on one gradient-sized buffer, on every rank."""
    rank, world, dev = init()
    if world < 2:
        print("wire microbench needs 2+ GPUs; skipped")
        return
    gen = torch.Generator(device=dev).manual_seed(1)
    signal = torch.randn(numel, device=dev, generator=gen)         # shared across ranks
    gen.manual_seed(100 + rank)
    g0 = signal / world + 3 * torch.randn(numel, device=dev, generator=gen) / world ** 0.5
    ref = g0.double()
    dist.all_reduce(ref)
    ref /= world
    rows = []
    for method in ("fp32", "bf16", "int16", "int8"):
        t0 = time.perf_counter()
        w = Wire(method, numel, world, rank)
        res = w(g0.clone())
        times = []
        for _ in range(iters):
            x = g0.clone()
            dist.barrier()
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            w(x)
            b.record()
            torch.cuda.synchronize()
            times.append(a.elapsed_time(b))
        ms = statistics.median(times)
        order = "-"
        if w.dtype is not None:
            order = "yes" if torch.equal(res, w(g0.clone(), reverse=True)) else "NO"
        rows.append(dict(method=method, gpus=world, B_per_el=round(w.bytes_per_el, 4),
                         ms=round(ms, 3), grad_GBps=round(numel * 4 / ms / 1e6, 1),
                         rel_err=f"{((res.double() - ref).norm() / ref.norm()).item():.3e}",
                         order_independent=order, seconds=round(time.perf_counter() - t0, 1)))
        if rank == 0:
            print(rows[-1], flush=True)
    if rank == 0:
        Path(out).mkdir(parents=True, exist_ok=True)
        with open(Path(out) / "wire.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
    dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results")
    p.add_argument("--numel", type=int, default=124_000_000, help="GPT-2 124M sized gradient")
    p.add_argument("--iters", type=int, default=5)
    a = p.parse_args()
    bench(a.out, a.numel, a.iters)
