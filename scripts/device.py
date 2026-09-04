#!/usr/bin/env python3
"""
device.py  -  central device selection for the BEATRICE family.

WHY THIS EXISTS
---------------
Upstream shipped `convert_to_gpu.py` / `_and_tensor` / `_scalar` with their
bodies hardcoded to `.to("cpu")`, so the "gpu" helpers moved everything onto the
CPU regardless of what hardware was available. Every tensor in the package flows
through those three helpers, so making the device selectable *here* makes the
whole package device-selectable without touching the model code.

CONTRACT
--------
The device is read ONCE from $FMBENCH_DEVICE and cached:

    unset / "cpu"   -> torch.device("cpu")      [DEFAULT - byte-identical to the
                                                 pre-existing behaviour, so
                                                 Iteration 004 is unaffected]
    "cuda" / "gpu"  -> torch.device("cuda")
    anything else   -> passed to torch.device() verbatim (e.g. "mps", "cuda:1")

If CUDA is requested and unavailable this RAISES rather than falling back. A
silent fallback would report a CPU timing as a GPU timing, which is precisely
the measurement this module exists to make; a run that cannot use the requested
device must fail loudly, not quietly produce a misleading number.

RANDOM NUMBERS
--------------
`rand_like_hostrng` draws uniforms from the HOST (CPU) generator and then moves
them to the target device. torch's CUDA generator produces a different sequence
from the CPU generator for the same seed, so a device-native `torch.rand_like`
would silently change the Gumbel draws and make a GPU run non-comparable with
the CPU baseline. Verified equal to `torch.rand_like` on CPU, including
interleaved call sequences, so this does not perturb existing CPU results.
"""
import os
import torch

_DEVICE = None


def get_device():
    """Resolve (and cache) the target torch device from $FMBENCH_DEVICE."""
    global _DEVICE
    if _DEVICE is not None:
        return _DEVICE

    want = os.environ.get("FMBENCH_DEVICE", "cpu").strip().lower()
    if want in ("", "cpu"):
        _DEVICE = torch.device("cpu")
    elif want in ("gpu", "cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "FMBENCH_DEVICE=%s but torch.cuda.is_available() is False. "
                "Refusing to fall back to CPU: a silent fallback would report a "
                "CPU runtime as a GPU runtime. Check the job requested a GPU "
                "(e.g. -l select=1:ncpus=4:mem=32gb:ngpus=1) and that the venv's "
                "torch is a CUDA build (torch.version.cuda is not None)." % want)
        _DEVICE = torch.device("cuda")
    else:
        _DEVICE = torch.device(want)

    return _DEVICE


def device_report():
    """One-line provenance string for the run log."""
    d = get_device()
    if d.type == "cuda":
        return ("device=cuda name=%s torch=%s cuda=%s"
                % (torch.cuda.get_device_name(0), torch.__version__,
                   torch.version.cuda))
    return "device=%s torch=%s" % (d.type, torch.__version__)


def rand_like_hostrng(alpha):
    """U(0,1) shaped like `alpha`, drawn on the HOST generator, moved to alpha's
    device. Keeps the random stream identical to a CPU run."""
    return torch.rand(alpha.size(), dtype=alpha.dtype).to(alpha.device)
