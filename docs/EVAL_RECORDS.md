# Held-out evaluation records

Compact result summaries live in `results/<method>/<setting>.json` and are always
tracked. Continuous leaderboard columns (`median dist Δ`, catastrophe, compounding,
regret, adaptation latency, peak memory) are recomputed from the per-episode /
per-replan schema written by the planner:

```text
eval_outputs/<tag>/<setting>/test{100,200,300}/**/episodes.jsonl
```

Only those held-out `episodes.jsonl` files are tracked in git (~40 MB). Selection
sweeps (`select*`, `selection/`), planner logs, Hydra dumps, and copied
`plan_targets.pkl` files remain git-ignored.

## Layout

| path | tracked? | role |
|---|---|---|
| `results/<method>/<setting>.json` | yes | selected params, success, selection cost, `out_dir` |
| `eval_outputs/<method>/<setting>/test*/**/episodes.jsonl` | yes | paired continuous metrics |
| `eval_outputs/**/select*/`, `**/selection/` | no | tuning only; never scored on the leaderboard |
| `plan.log`, `.hydra/`, `logs.json`, `plan_targets.pkl` | no | resume / debug / already available via target download |

`schema.load()` already skips selection directories. Tags must be the method name
(`pad`, not a dated selection workspace): `results/*/out_dir` points at
`eval_outputs/<method>/...`.

## Regenerating the leaderboard

```bash
.venv/bin/python scripts/leaderboard.py --out LEADERBOARD.md
```

A fresh clone with the tracked episode records is enough to rebuild every continuous
column. Re-running evaluations still requires checkpoints and goal files
(`docs/CHECKPOINTS.md`).

## Submissions

Include your method's held-out `episodes.jsonl` files in the PR alongside the result
record. Do not commit selection columns or planner dumps.
