"""PAARBench - benchmark harness for test-time adaptation of latent world models.

Ported evaluation code (``plan.py``, ``planning/``, ``env/``, ``models/``, ...) lives
as top-level modules so it stays diffable against the predecessor project it came
from; see ``PROVENANCE.md``.  Everything under ``paarbench/`` is new benchmark code:
the settings registry, dataset staging, the adapter protocol (M1), the output schema
and metrics (M2), and the submission harness (M3).
"""

__version__ = "0.0.1"
