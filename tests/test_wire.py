import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import wire

RANK_VARS = ("RANK", "WORLD_SIZE", "LOCAL_RANK", "SLURM_PROCID", "SLURM_NTASKS", "SLURM_LOCALID",
             "OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_SIZE", "OMPI_COMM_WORLD_LOCAL_RANK")


def test_ranks_come_from_torchrun_then_slurm(monkeypatch):
    for k in RANK_VARS:
        monkeypatch.delenv(k, raising=False)
    assert wire.env_rank() == (0, 1, 0)
    for k, v in (("SLURM_PROCID", "5"), ("SLURM_NTASKS", "16"), ("SLURM_LOCALID", "1")):
        monkeypatch.setenv(k, v)
    assert wire.env_rank() == (5, 16, 1)
    for k, v in (("RANK", "3"), ("WORLD_SIZE", "8"), ("LOCAL_RANK", "3")):
        monkeypatch.setenv(k, v)
    assert wire.env_rank() == (3, 8, 3)


@pytest.mark.parametrize("method,tol", [("bf16", 1e-2), ("int16", 2e-4), ("int8", 3e-2)])
def test_one_rank_error(method, tol):
    g = torch.randn(10_000)
    out = wire.Wire(method, g.numel())(g.clone())
    assert ((out - g).norm() / g.norm()) < tol


def test_error_feedback_delivers_small_gradients():
    g = torch.full((64,), 1e-3)
    g[0] = 1.0                                    # sets the grid; the rest round to zero
    plain, ef = wire.Wire("int8", 64), wire.Wire("int8ef", 64)
    sent_plain = sum(plain(g.clone()) for _ in range(100))
    sent_ef = sum(ef(g.clone()) for _ in range(100))
    assert (sent_plain[1:] == 0).all()
    assert torch.allclose(sent_ef[1:], 100 * g[1:], atol=2 / 127)


def _ranks(rank, world, port):
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}",
                            rank=rank, world_size=world)
    torch.manual_seed(rank)
    g = torch.randn(5000) * (1 + rank)
    ref = g.double()
    dist.all_reduce(ref)
    ref /= world
    for method, tol in (("fp32", 1e-6), ("int16", 1e-3), ("int8", 5e-2)):
        w = wire.Wire(method, g.numel(), world, rank)
        out = w(g.clone())
        assert ((out.double() - ref).norm() / ref.norm()).item() < tol, method
        everyone = [torch.empty_like(out) for _ in range(world)]
        dist.all_gather(everyone, out)
        assert all(torch.equal(out, o) for o in everyone), f"{method}: ranks disagree"
        if w.dtype is not None:
            assert torch.equal(out, w(g.clone(), reverse=True)), f"{method}: order changed bits"
    dist.destroy_process_group()


def test_four_ranks_agree_and_integers_ignore_order():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    mp.spawn(_ranks, args=(4, port), nprocs=4)
