"""Hyper-DoRA: HyperLoRA's adapter with ``correction_shape: dora``.

Everything except the emitted shape is ``methods/hyperlora``; the shape is
``models/shapes.py``. This file exists because the harness resolves an adapter inside
the method's own directory.
"""

from methods.hyperlora.adapter import HyperLoRAAdapter

__all__ = ["HyperLoRAAdapter"]
