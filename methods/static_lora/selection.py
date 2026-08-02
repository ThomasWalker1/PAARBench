"""Static LoRA's selection rule: which training epoch to deploy.

The method has no step size and no context, so the checkpoint is all there is
to choose.

The method's only hyperparameter is which trained checkpoint to deploy. That is still
a hyperparameter, and picking it on a test cohort is the leak the protocol exists to
prevent -- so it is chosen on the selection cohort at a declared cost of one column
per epoch.

The checkpoint directory depends on the setting: the architecture is shared across
domains but the weights are not. ``harness.setting_id`` is the only thing a selection
rule is told about where it is running, which is deliberate -- it is enough to pick
the right weights and not enough to reach a test cohort.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

ADAPTER_DIRS = {
    "pushobj": "checkpoints/pushobj_adapters/static_r2",
    "pushobj_shift": "checkpoints/pushobj_adapters/static_r2",
    "pusht": "checkpoints/pvs_adapters/static_r2",
}


class EpochSelection:
    """Pick the epoch that scores best on the selection cohort. One column each."""

    def __init__(self, adapter_dirs=None, epochs=range(1, 6)):
        self.adapter_dirs = dict(adapter_dirs or ADAPTER_DIRS)
        self.epochs = list(epochs)

    def adapter_dir(self, setting_id):
        try:
            return REPO_ROOT / self.adapter_dirs[setting_id]
        except KeyError:
            raise KeyError(
                f"{type(self).__name__} has no checkpoint directory for setting "
                f"{setting_id!r}; known: {sorted(self.adapter_dirs)}"
            ) from None

    def candidates(self, setting_id):
        """``(epoch, repo-relative path)`` for every staged epoch checkpoint.

        Repo-relative, not absolute, because that is the form ``method.yaml``
        declares and the form the harness resolves at adapter construction. Handing
        back an absolute path would make a selected checkpoint compare unequal to the
        identical declared one, so a rule that reconfirms the current choice would
        still look like a configuration change and force a needless re-run.
        """
        directory = self.adapter_dir(setting_id)
        found = [(e, directory / f"hyper_lora_epoch_{e}.pth") for e in self.epochs]
        found = [(e, p.relative_to(REPO_ROOT)) for e, p in found if p.is_file()]
        if not found:
            raise FileNotFoundError(
                f"no hyper_lora_epoch_*.pth under {directory}; "
                f"see docs/CHECKPOINTS.md for what needs staging"
            )
        return found

    def select(self, harness):
        best, best_score = None, float("-inf")
        for _epoch, path in self.candidates(harness.setting_id):
            result = harness.run(checkpoint_path=str(path))
            if result.success is None:
                continue
            if result.success > best_score:
                best, best_score = {"checkpoint_path": str(path)}, result.success
        if best is None:
            raise RuntimeError("every epoch's selection column failed")
        return best
