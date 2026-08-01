"""HyperJEPA's hyperparameter-selection rule.

This method has no step size to tune -- that is the point of amortizing -- but it does
have one hyperparameter that is easy to forget is a hyperparameter: **which training
epoch's adapter to deploy**. Picking it on a test cohort is exactly the leak the
protocol exists to prevent, so it is selected here, on the selection cohort, and the
cost is declared.

Cheap by comparison with a step-size grid: one column per epoch, against
``adajepa``'s 16.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ADAPTER_DIR = REPO_ROOT / "checkpoints/pushobj_adapters/hyper_r2_distill0"


class EpochSelection:
    """Pick the adapter epoch that scores best on the selection cohort."""

    def __init__(self, adapter_dir=None, epochs=range(1, 6)):
        self.adapter_dir = Path(adapter_dir) if adapter_dir else ADAPTER_DIR
        self.epochs = list(epochs)

    def candidates(self):
        found = []
        for epoch in self.epochs:
            path = self.adapter_dir / f"hyper_lora_epoch_{epoch}.pth"
            if path.is_file():
                found.append((epoch, path))
        if not found:
            raise FileNotFoundError(
                f"no hyper_lora_epoch_*.pth under {self.adapter_dir}; "
                f"see docs/CHECKPOINTS.md for what needs staging"
            )
        return found

    def select(self, harness):
        best, best_score = None, float("-inf")
        for epoch, path in self.candidates():
            result = harness.run(checkpoint_path=str(path))
            if result.success is None:
                continue
            if result.success > best_score:
                best, best_score = {"checkpoint_path": str(path)}, result.success
        if best is None:
            raise RuntimeError("every epoch's selection column failed")
        return best


class TunedFixed:
    """Epoch 2, the already-selected checkpoint, for reproducing a published number.

    Use ``EpochSelection`` for a submission; this skips the search that justifies it.
    """

    def select(self, harness):
        return {"checkpoint_path": str(ADAPTER_DIR / "hyper_lora_epoch_2.pth")}
