# Third-party notices

PAARBench is released under the MIT License (see [`LICENSE`](LICENSE)).
This file records third-party material whose terms also apply.

## AdaJEPA

Substantial portions of the world-model stack, planners, environments, and
datasets are derived from
[AdaJEPA](https://github.com/agentic-learning-ai-lab/adajepa)
(MIT License, Copyright (c) 2025 gaoyuezhou). That copyright is retained in
[`LICENSE`](LICENSE).

## Sonnet VQ-VAE (Apache-2.0)

`models/vqvae.py` includes code borrowed from
[deepmind/sonnet](https://github.com/deepmind/sonnet) and ported to PyTorch.
It is licensed under the Apache License, Version 2.0; the file header carries
the required copyright and license text.

## Checkpoints, datasets, and simulation assets

Base world-model weights, optional training trajectories, and maze assets are
not covered by this repository's MIT grant. Redistribution and use of those
artifacts remain subject to their upstream licenses and hosting terms (see
[`docs/CHECKPOINTS.md`](docs/CHECKPOINTS.md), [`docs/DATASET.md`](docs/DATASET.md),
and [`docs/MAZE.md`](docs/MAZE.md)).
