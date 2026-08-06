"""Declarative hyperparameter axes and the benchmark-standard selection rule.

Every submission with tunable hyperparameters declares them in ``method.yaml`` under
``tunable:``.  The harness runs the fixed protocol:

  1. Evaluate a 3-point grid per axis (3 cells for one axis, 3×3 for two) on the
     selection cohort.
  2. If the best cell lies on a grid boundary, shift that axis outward by one step
     (log/linear spacing for continuous values; adjacent pool members for discrete
     axes).  Repeat at most ``max_expansions`` times (default 2).
  3. Freeze the best configuration and evaluate once on each test cohort.

Custom ``selection.py`` rules are reserved for methods with no tunable axes (zero
selection cost) or for pre-standardization ablations recorded with
``--not-a-submission``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Dict, List, Mapping, Optional, Tuple

OBJECTIVES = frozenset({
    "success", "median_distance_delta", "catastrophe_rate", "compounding_slope",
})
SCALES = frozenset({"linear", "log", "discrete"})
GRID_SIZE = 3


@dataclass(frozen=True)
class AxisMapsTo:
    """Map a discrete semantic axis value to an adapter parameter."""

    param: str
    by_setting: Dict[str, str]

    def resolve(self, setting_id: str, value: Any) -> str:
        try:
            template = self.by_setting[setting_id]
        except KeyError as exc:
            raise KeyError(
                f"tunable axis maps_to has no template for setting {setting_id!r}; "
                f"known: {sorted(self.by_setting)}"
            ) from exc
        return template.format(value)


@dataclass
class TunableAxis:
    name: str
    initial: List[Any]
    scale: str = "linear"
    pool: Optional[List[Any]] = None
    pool_by_setting: Optional[Dict[str, List[Any]]] = None
    maps_to: Optional[AxisMapsTo] = None

    def pool_for(self, setting_id: str) -> Optional[List[Any]]:
        if self.pool_by_setting is not None:
            return list(self.pool_by_setting.get(setting_id, self.pool or []))
        return list(self.pool) if self.pool is not None else None


@dataclass
class TunableSpec:
    axes: List[TunableAxis]
    objective: str = "success"
    max_expansions: int = 2


def parse_tunable(raw: Any, *, manifest: str) -> Optional[TunableSpec]:
    """Parse the ``tunable:`` block from a method manifest."""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"{manifest}: 'tunable' must be a mapping")

    objective = str(raw.get("objective", "success"))
    if objective not in OBJECTIVES:
        raise ValueError(
            f"{manifest}: tunable.objective must be one of {sorted(OBJECTIVES)}"
        )

    max_expansions = raw.get("max_expansions", 2)
    if not isinstance(max_expansions, int) or max_expansions < 0:
        raise ValueError(f"{manifest}: tunable.max_expansions must be a non-negative int")

    axes_raw = raw.get("axes")
    if not isinstance(axes_raw, dict) or not axes_raw:
        raise ValueError(f"{manifest}: tunable.axes must be a non-empty mapping")
    if len(axes_raw) > 2:
        raise ValueError(
            f"{manifest}: tunable.axes declares {len(axes_raw)} axis(es); "
            f"the protocol allows at most 2"
        )

    axes: List[TunableAxis] = []
    for name, spec in axes_raw.items():
        if not isinstance(spec, dict):
            raise ValueError(f"{manifest}: tunable.axes.{name} must be a mapping")
        initial = spec.get("initial")
        if not isinstance(initial, list) or len(initial) != GRID_SIZE:
            raise ValueError(
                f"{manifest}: tunable.axes.{name}.initial must be a list of exactly "
                f"{GRID_SIZE} values"
            )
        scale = str(spec.get("scale", "linear"))
        if scale not in SCALES:
            raise ValueError(
                f"{manifest}: tunable.axes.{name}.scale must be one of {sorted(SCALES)}"
            )
        pool = spec.get("pool")
        if pool is not None:
            if not isinstance(pool, list) or not pool:
                raise ValueError(
                    f"{manifest}: tunable.axes.{name}.pool must be a non-empty list"
                )
            pool = list(pool)
        pool_by_setting = spec.get("pool_by_setting")
        if pool_by_setting is not None:
            if not isinstance(pool_by_setting, dict) or any(
                    not isinstance(v, list) or not v
                    for v in pool_by_setting.values()):
                raise ValueError(
                    f"{manifest}: tunable.axes.{name}.pool_by_setting must map "
                    f"settings to non-empty lists"
                )
            pool_by_setting = {str(k): list(v) for k, v in pool_by_setting.items()}
        maps_raw = spec.get("maps_to")
        maps_to = None
        if maps_raw is not None:
            if scale != "discrete":
                raise ValueError(
                    f"{manifest}: tunable.axes.{name}.maps_to requires scale: discrete"
                )
            if not isinstance(maps_raw, dict):
                raise ValueError(
                    f"{manifest}: tunable.axes.{name}.maps_to must be a mapping"
                )
            param = maps_raw.get("param")
            by_setting = maps_raw.get("by_setting")
            if not param or not isinstance(by_setting, dict):
                raise ValueError(
                    f"{manifest}: tunable.axes.{name}.maps_to needs 'param' and "
                    f"'by_setting'"
                )
            maps_to = AxisMapsTo(param=str(param),
                                 by_setting={str(k): str(v)
                                             for k, v in by_setting.items()})
        axes.append(TunableAxis(name=str(name), initial=list(initial), scale=scale,
                                pool=pool, pool_by_setting=pool_by_setting,
                                maps_to=maps_to))

    return TunableSpec(axes=axes, objective=objective, max_expansions=max_expansions)


def validate_tunable(method) -> List[str]:
    """Return human-readable problems with a method's tunable declaration."""
    problems: List[str] = []
    if method.tunable is None:
        return problems
    if method.selection_ref is not None:
        problems.append(
            "declares both 'tunable' and 'selection'; remove 'selection' and let the "
            "standard protocol read tunable.axes"
        )
    for axis in method.tunable.axes:
        if axis.maps_to is not None:
            for setting_id in method.settings:
                if setting_id not in axis.maps_to.by_setting:
                    problems.append(
                        f"tunable.axes.{axis.name}.maps_to.by_setting is missing "
                        f"setting {setting_id!r}"
                    )
        if axis.scale == "discrete" and axis.pool is None and not axis.pool_by_setting:
            problems.append(
                f"tunable.axes.{axis.name} uses scale: discrete but declares neither "
                f"'pool' nor 'pool_by_setting'"
            )
        if axis.pool_by_setting is not None:
            for setting_id in method.settings:
                if setting_id not in axis.pool_by_setting:
                    problems.append(
                        f"tunable.axes.{axis.name}.pool_by_setting is missing "
                        f"setting {setting_id!r}"
                    )
                else:
                    pool = axis.pool_by_setting[setting_id]
                    for value in axis.initial:
                        if value not in pool:
                            problems.append(
                                f"tunable.axes.{axis.name}.initial value {value!r} is "
                                f"not in pool_by_setting.{setting_id}"
                            )
    return problems


def _params_key(params: Mapping[str, Any]) -> Tuple[Tuple[str, Any], ...]:
    return tuple(sorted(params.items()))


def _expand_axis(values: List[Any], scale: str, direction: str,
                 pool: Optional[List[Any]]) -> Optional[List[Any]]:
    """Shift a 3-point window outward. Returns the new values or None if blocked."""
    if len(values) != GRID_SIZE:
        raise ValueError("internal error: axis grid must have 3 points")
    if scale == "discrete":
        if pool is None:
            return None
        ordered = list(pool)
        if direction == "low":
            idx = ordered.index(values[0])
            if idx == 0:
                return None
            return [ordered[idx - 1]] + values[:-1]
        idx = ordered.index(values[-1])
        if idx == len(ordered) - 1:
            return None
        return values[1:] + [ordered[idx + 1]]

    if scale == "log":
        ratio = values[1] / values[0]
        if ratio <= 0:
            raise ValueError("log-scale axis needs positive values with ratio > 0")
        if direction == "low":
            new = values[0] / ratio
            if new <= 0:
                return None
            return [new] + values[:-1]
        return values[1:] + [values[-1] * ratio]

    # linear
    step = values[1] - values[0]
    if direction == "low":
        return [values[0] - step] + values[:-1]
    return values[1:] + [values[-1] + step]


class StandardSelection:
    """Benchmark-standard 3-point grid with optional boundary expansion."""

    RULE_NAME = "standard (3-point grid + boundary expansion)"

    def __init__(self, spec: TunableSpec, setting_id: str):
        self.spec = spec
        self.setting_id = setting_id
        self._axis_values = {axis.name: list(axis.initial) for axis in spec.axes}

    def _axis_by_name(self) -> Dict[str, TunableAxis]:
        return {axis.name: axis for axis in self.spec.axes}

    def _grid_cells(self) -> List[Dict[str, Any]]:
        names = [axis.name for axis in self.spec.axes]
        value_lists = [self._axis_values[name] for name in names]
        return [dict(zip(names, combo)) for combo in product(*value_lists)]

    def _cell_to_params(self, cell: Dict[str, Any]) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        axis_map = self._axis_by_name()
        for name, value in cell.items():
            axis = axis_map[name]
            if axis.maps_to is not None:
                params[axis.maps_to.param] = axis.maps_to.resolve(self.setting_id, value)
            else:
                params[name] = value
        return params

    def _score(self, result) -> Optional[float]:
        value = getattr(result, self.spec.objective)
        if value is None:
            return None
        if self.spec.objective == "success":
            return float(value)
        return -float(value)

    def _best_cell(self, scores: Dict[Tuple[Tuple[str, Any], ...], float],
                   cells: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        best_cell = None
        best_score = float("-inf")
        for cell in cells:
            params = self._cell_to_params(cell)
            score = scores.get(_params_key(params))
            if score is None or score <= best_score:
                continue
            best_cell, best_score = cell, score
        return best_cell

    def _boundary_directions(self, cell: Dict[str, Any]) -> Dict[str, str]:
        directions: Dict[str, str] = {}
        for name, value in cell.items():
            values = self._axis_values[name]
            idx = values.index(value)
            if idx == 0:
                directions[name] = "low"
            elif idx == GRID_SIZE - 1:
                directions[name] = "high"
        return directions

    def select(self, harness) -> Dict[str, Any]:
        axis_map = self._axis_by_name()
        evaluated: Dict[Tuple[Tuple[str, Any], ...], Any] = {}
        scores: Dict[Tuple[Tuple[str, Any], ...], float] = {}
        best_params: Optional[Dict[str, Any]] = None
        best_score = float("-inf")

        for _round in range(self.spec.max_expansions + 1):
            cells = self._grid_cells()
            for cell in cells:
                params = self._cell_to_params(cell)
                key = _params_key(params)
                if key not in evaluated:
                    evaluated[key] = harness.run(**params)
                result = evaluated[key]
                score = self._score(result)
                if score is None:
                    continue
                scores[key] = score
                if score > best_score:
                    best_params, best_score = dict(params), score

            if _round >= self.spec.max_expansions:
                break

            winner = self._best_cell(scores, cells)
            if winner is None:
                break
            directions = self._boundary_directions(winner)
            if not directions:
                break

            expanded = False
            for name, direction in sorted(directions.items()):
                axis = axis_map[name]
                pool = axis.pool_for(self.setting_id)
                new_values = _expand_axis(self._axis_values[name], axis.scale,
                                          direction, pool)
                if new_values is None:
                    continue
                self._axis_values[name] = new_values
                expanded = True
            if not expanded:
                break

        if best_params is None:
            raise RuntimeError(
                "every selection column failed or returned an incomplete objective; "
                "nothing to select. check the per-shape logs under eval_outputs/"
            )
        return best_params
