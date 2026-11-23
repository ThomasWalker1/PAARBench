"""Static (IA)³: Static LoRA's adapter with ``correction_shape: ia3``.

The control for ``methods/hyper_ia3``. Everything except the shape is
``methods/static_lora``; the shape is ``models/shapes.py``.
"""

from methods.static_lora.adapter import StaticLoRAAdapter

__all__ = ["StaticLoRAAdapter"]
