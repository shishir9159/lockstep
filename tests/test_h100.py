import pytest
import torch

import h100


def test_guard_refuses_anything_but_an_h100():
    if torch.cuda.is_available() and "H100" in torch.cuda.get_device_name(0):
        pytest.skip("running on an H100")
    with pytest.raises(SystemExit, match="needs an H100"):
        h100.require_h100()
