"""Discovery and loading of contributed methods.

A method is a **self-contained directory** under ``methods/``. Adding one means
creating that directory; it means editing nothing else. There is no central registry
to append to, because a registry is a merge conflict and a review burden, and because
"you must also add your name to a list" is exactly the kind of friction that stops
people contributing.

    methods/
      my_method/
        method.yaml     required -- metadata and adapter kwargs
        adapter.py      the TestTimeAdapter implementation
        selection.py    optional -- the hyperparameter-selection rule
        README.md       what it is, what it reports

Discovery is by directory listing. A directory whose name starts with ``_`` is
skipped, so ``methods/_template/`` is a template rather than a method.

Method modules are loaded under a private module name (``paarbench._methods.<name>``)
so two methods can both define ``adapter.py`` without colliding, and so a method
cannot shadow a benchmark module by naming a file ``settings.py``.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
METHODS_DIR = REPO_ROOT / "methods"

_MODULE_PREFIX = "paarbench._methods"


class MethodError(Exception):
    """A method directory is malformed. Always names the file and the fix."""


def resolve_repo_paths(params: Dict[str, Any]) -> Dict[str, Any]:
    """Make relative ``*_path`` / ``*_dir`` params repo-relative.

    Hydra chdirs into the run directory before the planner is built, so a relative
    path written in ``method.yaml`` -- the obvious thing to write -- resolves against
    somewhere under ``eval_outputs/`` and fails. Every method shipping a checkpoint
    would hit this, so it is fixed once here rather than in each method.

    Only rewrites when the key looks like a path, the value is a relative string, and
    the repo-relative interpretation actually exists on disk. A value that does not
    name a real file is left alone, so this cannot mangle an arbitrary string that
    happens to end in ``_path``.
    """
    resolved = {}
    for key, value in params.items():
        if (
            isinstance(value, str)
            and (key.endswith("_path") or key.endswith("_dir"))
            and not Path(value).is_absolute()
            and (REPO_ROOT / value).exists()
        ):
            resolved[key] = str(REPO_ROOT / value)
        else:
            resolved[key] = value
    return resolved


@dataclass
class Method:
    """A loaded method definition. Metadata only -- nothing is instantiated yet."""

    name: str
    path: Path
    display_name: str
    adapter_ref: str
    """``"<module>:<attr>"``, resolved relative to the method directory."""

    settings: List[str]
    """Setting ids this method declares support for."""

    params: Dict[str, Any] = field(default_factory=dict)
    """Adapter constructor kwargs -- the frozen, selected hyperparameters."""

    selection_ref: Optional[str] = None
    """``"<module>:<attr>"`` of a SelectionRule, or None for no hyperparameters."""

    requires_episode_isolation: bool = False
    """Must each episode get its own process, with its own adapter instance?

    The planner evaluates a whole cohort as one batch, so by default an adapter sees
    a batched observation and all episodes step in lockstep. A method that owns
    *mutable shared state* -- model weights and an optimizer, say -- cannot work that
    way: one gradient step would average unrelated episodes into a single correction,
    which is a different method from per-episode adaptation.

    Setting this makes the harness fan out one process per episode instead of one per
    shape. It is much more expensive (n_evals x the processes), so declare it only if
    batching genuinely changes what the method computes.
    """

    description: str = ""
    reference: str = ""

    def load_adapter_class(self):
        return _load_ref(self, self.adapter_ref, "adapter")

    def load_selection_rule(self):
        if self.selection_ref is None:
            return None
        return _load_ref(self, self.selection_ref, "selection")

    def build(self, wm, preprocessor, **overrides):
        """Instantiate this method's adapter against a world model.

        ``overrides`` beat ``method.yaml``'s ``params``, which is how a selection rule
        evaluates a candidate configuration without rewriting the file.
        """
        cls = self.load_adapter_class()
        params = resolve_repo_paths({**self.params, **overrides})
        try:
            return cls(wm=wm, preprocessor=preprocessor, **params)
        except TypeError as exc:
            raise MethodError(
                f"method {self.name!r}: could not construct {self.adapter_ref} with "
                f"params {sorted(params)}: {exc}\n"
                f"  fix: the adapter's __init__ must accept (wm, preprocessor, **params) "
                f"where params are the keys under `params:` in {self.path / 'method.yaml'}"
            ) from exc

    def supports(self, setting_id: str) -> bool:
        return setting_id in self.settings


def _load_ref(method: Method, ref: str, kind: str):
    if ":" not in ref:
        raise MethodError(
            f"method {method.name!r}: {kind} reference {ref!r} must be "
            f"'<module>:<attribute>', e.g. 'adapter:MyAdapter'"
        )
    module_name, _, attr = ref.partition(":")
    module = _import_method_module(method, module_name)
    try:
        return getattr(module, attr)
    except AttributeError:
        raise MethodError(
            f"method {method.name!r}: {module_name}.py defines no {attr!r} "
            f"(referenced as {kind}: {ref!r} in method.yaml)"
        ) from None


def _import_method_module(method: Method, module_name: str):
    """Import ``<method dir>/<module_name>.py`` under a private, collision-free name."""
    qualified = f"{_MODULE_PREFIX}.{method.name}.{module_name}"
    if qualified in sys.modules:
        return sys.modules[qualified]

    source = method.path / f"{module_name}.py"
    if not source.is_file():
        raise MethodError(f"method {method.name!r}: missing {source}")

    spec = importlib.util.spec_from_file_location(qualified, source)
    if spec is None or spec.loader is None:
        raise MethodError(f"method {method.name!r}: cannot import {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified] = module
    # The method directory goes on sys.path for the duration of the import so a
    # method can split itself across several files and import them normally.
    sys.path.insert(0, str(method.path))
    try:
        spec.loader.exec_module(module)
    except Exception:
        del sys.modules[qualified]
        raise
    finally:
        sys.path.remove(str(method.path))
    return module


_REQUIRED = ("display_name", "adapter", "settings")


def load(name: str, methods_dir: Optional[Path] = None) -> Method:
    """Load one method by directory name."""
    root = Path(methods_dir) if methods_dir is not None else METHODS_DIR
    path = root / name
    manifest = path / "method.yaml"
    if not manifest.is_file():
        available = [m.name for m in discover(root)]
        raise MethodError(
            f"no method {name!r}: {manifest} does not exist. "
            f"available: {available or '(none)'}"
        )

    try:
        raw = yaml.safe_load(manifest.read_text()) or {}
    except yaml.YAMLError as exc:
        raise MethodError(f"{manifest}: invalid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise MethodError(f"{manifest}: expected a mapping at the top level")

    missing = [k for k in _REQUIRED if k not in raw]
    if missing:
        raise MethodError(
            f"{manifest}: missing required key(s) {missing}. "
            f"see methods/_template/method.yaml"
        )

    declared = raw["settings"]
    if isinstance(declared, str):
        declared = [declared]
    if not isinstance(declared, list) or not declared:
        raise MethodError(f"{manifest}: 'settings' must be a non-empty list of setting ids")

    params = raw.get("params") or {}
    if not isinstance(params, dict):
        raise MethodError(f"{manifest}: 'params' must be a mapping")

    declared_name = raw.get("name", name)
    if declared_name != name:
        raise MethodError(
            f"{manifest}: 'name' is {declared_name!r} but the directory is {name!r}; "
            f"they must match so results are attributable"
        )

    return Method(
        name=name,
        path=path,
        display_name=str(raw["display_name"]),
        adapter_ref=str(raw["adapter"]),
        settings=[str(s) for s in declared],
        params=params,
        selection_ref=(str(raw["selection"]) if raw.get("selection") else None),
        requires_episode_isolation=bool(raw.get("requires_episode_isolation", False)),
        description=str(raw.get("description", "")),
        reference=str(raw.get("reference", "")),
    )


def discover(methods_dir: Optional[Path] = None) -> List[Method]:
    """Every well-formed method directory, sorted by name.

    A directory that fails to parse raises rather than being skipped: silently
    dropping a broken submission would let a contributor believe their method ran.
    """
    root = Path(methods_dir) if methods_dir is not None else METHODS_DIR
    if not root.is_dir():
        return []
    found = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name.startswith((".", "_")):
            continue
        if not (child / "method.yaml").is_file():
            continue
        found.append(load(child.name, root))
    return found


def validate(method: Method) -> List[str]:
    """Check a method without running it. Returns human-readable problems.

    This is what CI runs on a pull request, so the messages have to be actionable by
    someone who has never read the harness.
    """
    from paarbench import settings as settings_mod

    problems: List[str] = []

    for setting_id in method.settings:
        if setting_id not in settings_mod.SETTINGS:
            problems.append(
                f"declares setting {setting_id!r}, which does not exist. "
                f"known: {sorted(settings_mod.SETTINGS)}"
            )
        elif not settings_mod.SETTINGS[setting_id].enabled:
            problems.append(
                f"declares setting {setting_id!r}, which is currently disabled: "
                f"{settings_mod.SETTINGS[setting_id].notes.splitlines()[0]}"
            )

    try:
        adapter_cls = method.load_adapter_class()
    except MethodError as exc:
        problems.append(str(exc))
        return problems

    from paarbench.adapter import TestTimeAdapter

    required = [n for n in ("on_episode_start", "on_transition", "before_plan", "metrics")
                if not callable(getattr(adapter_cls, n, None))]
    if required:
        problems.append(
            f"{method.adapter_ref} is missing required hook(s) {required}. "
            f"see paarbench/adapter.py:TestTimeAdapter"
        )
    del TestTimeAdapter  # imported for the docstring reference above

    if method.selection_ref is not None:
        try:
            rule = method.load_selection_rule()
        except MethodError as exc:
            problems.append(str(exc))
        else:
            if not callable(getattr(rule, "select", None)):
                problems.append(
                    f"{method.selection_ref} has no callable 'select'. "
                    f"see paarbench/selection.py:SelectionRule"
                )

    if not method.description.strip():
        problems.append("method.yaml has no 'description'; say what the method does")

    return problems
