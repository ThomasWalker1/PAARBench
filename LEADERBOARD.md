# Leaderboard

Pending rerun under the standardized hyperparameter-selection protocol
(3-point grid per tunable axis, up to two boundary expansions on the selection cohort).

All previous `results/*.json` records were removed on 2026-08-06. Regenerate this file with:

```bash
.venv/bin/python scripts/download_targets.py          # if goal files are not staged
.venv/bin/python scripts/download_checkpoints.py all  # if checkpoints are not staged
bash scripts/rerun_all.sh
```

Until `scripts/rerun_all.sh` completes, there are no comparable rows to report.
