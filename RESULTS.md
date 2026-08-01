# Results

Reproduction log. One section per milestone acceptance, newest last. Numbers here are
produced by this repo; targets are the predecessor's, pinned at `628dff7` (`PROVENANCE.md`).

---

## M0 — scaffold

**Accept:** a frozen PushObj column runs end to end and reports success 0.490 ± noise on
seed 300.

**Status: accepted, exactly.** 2026-08-01.

```
.venv/bin/python scripts/eval_column.py \
    --setting pushobj --cohort selection --tag frozen --gpus 0,1,2,3 \
    --tmpfs-root /dev/shm/tw78/data
```

| shape | PAARBench | predecessor (`eval_outputs/indist_matrix/frozen/seed300`) |
|---|---|---|
| T | 0.500 | 0.50 |
| L | 0.440 | 0.44 |
| Z | 0.600 | 0.60 |
| + | 0.420 | 0.42 |
| **mean** | **0.4900** | **0.490** |

n=200 (4 shapes × 50), seed 300, 20 MPC replans. Agreement is per-shape and exact, not
just on the mean — which is the stronger check, since a mean can match while two shapes
move in opposite directions. Exactness is expected rather than lucky: the planner is
deterministic (§6), and that determinism is what licenses paired comparison later.

Environment differences from the predecessor that this run shows are immaterial: a fresh
`uv` venv (`torch 2.3.0+cu121`, same pins), `conf/eval.yaml` in place of
`conf/plan_gd_mpc_local.yaml` (byte-identical content), and the training dataset read from
tmpfs via `dataset_data_path` rather than the SMB mount (same bytes; verified by a
path-and-size manifest comparison over all 16,051 files).

### What M0 covers

- `uv` project, Python 3.9, cu121 torch pinned explicitly per §9. `torch.cuda.is_available()`
  verified `True` on 8× A6000.
- §7 "take" list extracted from `628dff7` via `git archive`; one path fixed
  (`env/__init__.py`'s `LOCAL_DEPS_DIR` pointed into the predecessor). `PROVENANCE.md`
  records every file's origin and every deliberate omission.
- `paarbench/settings.py` — the §6 suite as data, with the selection/test cohort split
  declared by the benchmark. `pointmaze` is registered but **disabled** pending §2.3.
- `paarbench/staging.py` — tmpfs staging baked into the harness (§9) rather than left to
  per-config overrides. Adopts an already-correct stage instead of re-copying 2 GB.
- `scripts/eval_column.py` — the column driver: shapes fan out one per GPU, resumable,
  tees to disk.

### What M0 does not cover

The frozen arm exercises no adapter code, so this says nothing about the two methods being
portable. That is M1, and it is the first point at which a number could disagree.

### Note for M1

`AdaJEPAAdapter` has **no episode-reset method at all** — it is constructed once per
planning process and each process evaluates one batch, so the need never arose. The reset
that §5 requires is therefore new work for that method, not a port. See
`docs/ADAPTER_PROTOCOL.md`, which records the full call-site analysis behind the protocol.
