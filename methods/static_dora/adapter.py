"""Static DoRA: Static LoRA's adapter with ``correction_shape: dora``.

The control for ``methods/hyper_dora``. Everything except the shape is
``methods/static_lora``; the shape is ``models/shapes.py``.
"""

from methods.static_lora.adapter import StaticLoRAAdapter

__all__ = ["StaticLoRAAdapter"]
