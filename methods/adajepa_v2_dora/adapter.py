"""AdaJEPA v2 with a different correction shape: same adapter, `correction_shape: dora`.

The fitting rule, brake and optimizer are `methods/adajepa_v2`; the shape is
`models/shapes.py`. This file exists only because the harness resolves an adapter
inside the method's own directory.
"""

from methods.adajepa_v2.adapter import AdaJEPAv2Adapter

__all__ = ["AdaJEPAv2Adapter"]
