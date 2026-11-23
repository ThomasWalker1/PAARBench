"""Static AdaLN: Static LoRA's adapter with ``correction_shape: adaln``.

The control for ``methods/hyper_adaln``. Everything except the shape is
``methods/static_lora``; the shape is ``models/shapes.py``.
"""

from methods.static_lora.adapter import StaticLoRAAdapter

__all__ = ["StaticLoRAAdapter"]
