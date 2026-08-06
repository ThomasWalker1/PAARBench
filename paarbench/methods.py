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

from paarbench.tunable import TunableSpec, parse_tunable, validate_tunable

REPO_ROOT = Path(__file__).resolve().parent.parent
METHODS_DIR = REPO_ROOT / "methods"

_MODULE_PREFIX = "paarbench._methods"


_ARTIFACT_SUFFIXES = ("_path", "_dir")


class MethodError(Exception):
    """A method directory is malformed. Always names the file and the fix."""


def resolve_artifact_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Anchor repository-relative artifact parameters to the repository root.

    ``methods/README.md`` promises exactly this: "Any param key ending in ``_path`` or
    ``_dir`` whose value names a real repo-relative file is resolved for you before your
    adapter is constructed." A method must be able to declare
    ``checkpoint_path: checkpoints/...`` and be indifferent to where the planner was
    launched from, because ``plan.py`` moves into the run directory before planning and
    the obvious relative path would otherwise resolve somewhere under ``eval_outputs/``.
    ``validate`` below *rejects* absolute artifact paths, so repo-relative is the only
    thing a submission is allowed to declare and it therefore has to work.

    Anchored on this file's location rather than on the process CWD, so it gives the same
    answer whenever it is called. A value that names nothing in the repository is passed
    through untouched: it may be an output path, a Hub id, or a deliberately absolute
    path, and guessing would be worse than leaving it alone.
    """
    resolved = {}
    for key, value in params.items():
        if (
            isinstance(value, str)
            and key.endswith(_ARTIFACT_SUFFIXES)
            and not Path(value).is_absolute()
        ):
            candidate = REPO_ROOT / value
            if candidate.exists():
                value = str(candidate)
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

    params_by_setting: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    """Per-setting overrides layered on top of ``params``.

    A method carrying trained weights needs different ones per domain -- the same
    hypernetwork architecture, a different checkpoint for PushObj than for PushT.
    Without this a method would have to be duplicated per setting, which would split
    its results across two leaderboard rows for no reason.
    """

    selection_ref: Optional[str] = None
    """``"<module>:<attr>"`` of a SelectionRule, or None for no hyperparameters."""

    tunable: Optional[TunableSpec] = None
    """Declarative tunable axes; triggers the benchmark-standard selection protocol."""

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
        params = resolve_artifact_params({**self.params, **overrides})
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

    def params_for(self, setting_id: str) -> Dict[str, Any]:
        """The frozen parameters this method uses on one setting."""
        return {**self.params, **self.params_by_setting.get(setting_id, {})}


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

    by_setting = raw.get("params_by_setting") or {}
    if not isinstance(by_setting, dict) or any(
            not isinstance(v, dict) for v in by_setting.values()):
        raise MethodError(
            f"{manifest}: 'params_by_setting' must map a setting id to a mapping of "
            f"parameter overrides")
    unknown = set(by_setting) - set(declared)
    if unknown:
        raise MethodError(
            f"{manifest}: 'params_by_setting' names setting(s) {sorted(unknown)} that "
            f"are not in 'settings' {declared}")

    declared_name = raw.get("name", name)
    if declared_name != name:
        raise MethodError(
            f"{manifest}: 'name' is {declared_name!r} but the directory is {name!r}; "
            f"they must match so results are attributable"
        )

    try:
        tunable = parse_tunable(raw.get("tunable"), manifest=str(manifest))
    except ValueError as exc:
        raise MethodError(str(exc)) from exc

    return Method(
        name=name,
        path=path,
        display_name=str(raw["display_name"]),
        adapter_ref=str(raw["adapter"]),
        settings=[str(s) for s in declared],
        params=params,
        params_by_setting=by_setting,
        selection_ref=(str(raw["selection"]) if raw.get("selection") else None),
        tunable=tunable,
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
        for key, value in method.params_for(setting_id).items():
            if (
                isinstance(value, str)
                and (key.endswith("_path") or key.endswith("_dir"))
                and Path(value).is_absolute()
            ):
                problems.append(
                    f"{setting_id} parameter {key!r} must be repository-relative; "
                    "absolute artifact paths are machine-specific and may disclose "
                    "personal directory names"
                )

    for setting_id in method.settings:
        if setting_id not in settings_mod.SETTINGS:
            problems.append(
                f"declares setting {setting_id!r}, which does not exist. "
                f"known: {sorted(settings_mod.SETTINGS)}"
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

    problems.extend(validate_tunable(method))

    if method.tunable is None and method.selection_ref is not None:
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
