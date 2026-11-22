"""Static SSF: Static LoRA's adapter with ``correction_shape: ssf``.

The control for ``methods/hyper_ssf``. Everything except the shape is
``methods/static_lora``; the shape is ``models/shapes.py``.
"""

from methods.static_lora.adapter import StaticLoRAAdapter

__all__ = ["StaticLoRAAdapter"]
