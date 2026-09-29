"""Device guard: the Triton and CUDA paths here are only valid on an H100."""

import sys

import torch


def require_h100():
    """Return GPU 0's properties, or exit unless it is an H100 (sm_90)."""
    if not torch.cuda.is_available():
        sys.exit("needs an H100: no CUDA device")
    p = torch.cuda.get_device_properties(0)
    if (p.major, p.minor) != (9, 0) or "H100" not in p.name:
        sys.exit(f"needs an H100: found {p.name} (sm_{p.major}{p.minor})")
    return p
