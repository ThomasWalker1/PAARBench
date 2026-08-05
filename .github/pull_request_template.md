<!--
Adding an adaptation method? Fill in the first section and delete the rest.
Anything else (a fix, a doc change)? Delete the first section.
See CONTRIBUTING.md.
-->

## Submission

**Method**: `methods/<name>/` — one sentence on what it adapts and what failure it targets.

**Settings evaluated**: <!-- pushobj / pushobj_shift / pusht -->

**Selection rule and cost**: <!-- e.g. selection:MyGrid, 8 columns, scored on success -->
<!-- If you declare no rule, say which zero you are claiming: no hyperparameters, or
     hyperparameters that were authored rather than selected. See methods/README.md. -->

**Episode isolation**: <!-- requires_episode_isolation true/false, and why -->

**Results**: <!-- the rows you added to LEADERBOARD.md, or a summary table -->

### Checklist

- [ ] `scripts/validate_method.py <name>` passes
- [ ] `python -m pytest -q` passes
- [ ] `results/<name>/<setting>.json` included, `complete: true`
- [ ] Row(s) **added** to `LEADERBOARD.md` — not regenerated (it would blank other methods'
      columns; see CONTRIBUTING.md)
- [ ] `results/frozen/*.json` untouched
- [ ] Frozen baseline re-run locally to pair against; it reproduced the committed record
      (if it did not, say so below — that is a finding)
- [ ] Method `README.md` says what it reports and where its hyperparameters came from

### Did you change anything outside `methods/<name>/`?

<!-- If yes: list those commits and why the submission could not be produced without them.
     Each fix should be its own commit. If no: delete this section. -->

## Notes for reviewers

<!-- Anything surprising: a negative result, a metric that moved the wrong way, a limitation
     you want on the record. Negative results are submissions too. -->
