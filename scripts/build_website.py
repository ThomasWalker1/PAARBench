#!/usr/bin/env python3
"""Generate static HTML pages for PAARBench from results/ and methods/."""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEBSITE = ROOT / "website"
RESULTS = ROOT / "results"
METHODS = ROOT / "methods"
LEADERBOARD = ROOT / "LEADERBOARD.md"

METHOD_ORDER = [
    "adajepa", "hyperjepa", "static_lora", "pad", "frozen",
]

FROZEN_SLUGS = frozenset({"frozen"})
SETTING_ORDER = (
    "pushobj", "pushobj_shift", "pusht", "maze_medium",
    "maze_medium_low_density", "maze_medium_high_damping",
    "maze_diverse",
)

FROZEN_META = {
    "frozen": {
        "name": "frozen",
        "display_name": "Frozen",
        "description": (
            "The built-in do-nothing baseline: the world model and planner run with "
            "frozen pretrained weights. Paired metrics compare every adapted method "
            "against this reference on the same episodes (batched evaluation mode)."
        ),
        "reference": "",
        "settings": list(SETTING_ORDER),
        "requires_episode_isolation": False,
        "selection": "",
        "params": {},
        "readme": "",
    },
}

SETTING_INFO = {
    "pushobj": {
        "title": "PushObj",
        "description": (
            "Four shapes (T, L, Z, +) on the pushobj_shape_shift checkpoint — "
            "50 episodes per shape per cohort (n=600 pooled across test seeds)."
        ),
    },
    "pushobj_shift": {
        "title": "PushObj Shift",
        "description": (
            "Held-out shapes (I, small_tee, square) on the same base model — "
            "n=450 pooled. Hyperparameters are inherited from PushObj; no selection here."
        ),
    },
    "pusht": {
        "title": "PushT",
        "description": (
            "Visual pushing on pusht_visual_shift with three shapes (T, L, Z) — "
            "50 episodes per shape per cohort (n=450 pooled)."
        ),
    },
    "maze_medium": {
        "title": "PointMaze Medium",
        "description": "Nominal medium maze (n=150 pooled); selection happens on seed 0.",
    },
    "maze_medium_low_density": {
        "title": "PointMaze Low Density",
        "description": "Medium maze with density scale 0.2; parameters inherit from PointMaze Medium.",
    },
    "maze_medium_high_damping": {
        "title": "PointMaze High Damping",
        "description": "Medium maze with damping scale 20; parameters inherit from PointMaze Medium.",
    },
    "maze_diverse": {
        "title": "DiverseMaze",
        "description": (
            "Held-out DiverseMaze layouts on the PointMaze Medium base (n=150 pooled); "
            "parameters inherit from PointMaze Medium."
        ),
    },
}

SETTINGS_INTRO = (
    "Each setting pairs a base world model with a fixed episode distribution. "
    "Hyperparameter tuning uses one selection cohort (seed&nbsp;0); leaderboard "
    "scores are pooled over three held-out test cohorts (seeds&nbsp;100, 200, and 300). "
    "Selection and test draws are always disjoint."
)

SETTINGS_PARAGRAPHS = {
    "pushobj": (
        "<strong>PushObj</strong> (<code>pushobj</code>) is the primary object-pushing "
        "setting: checkpoint <code>pushobj_shape_shift</code>, shapes T, L, Z, and +. "
        "Methods select hyperparameters on seed&nbsp;0, then run once on each test seed."
    ),
    "pushobj_shift": (
        "<strong>PushObj Shift</strong> (<code>pushobj_shift</code>) holds out different "
        "shapes (I, small_tee, square) on the same checkpoint. There is no selection "
        "cohort — parameters are frozen from PushObj — but evaluation still uses the "
        "same three test seeds."
    ),
    "pusht": (
        "<strong>PushT</strong> (<code>pusht</code>) is the visual pushing setting on "
        "<code>pusht_visual_shift</code> with shapes T, L, and Z. It follows the same "
        "one-seed selection, three-seed evaluation layout as PushObj."
    ),
    "maze_medium": (
        "<strong>PointMaze Medium</strong> (<code>maze_medium</code>) uses the "
        "<code>mediummaze_dynamics_shift</code> base and immutable hard-goal episodes."
    ),
    "maze_medium_low_density": (
        "<strong>PointMaze Low Density</strong> changes only mass through density scale 0.2; "
        "it inherits parameters from <code>maze_medium</code>."
    ),
    "maze_medium_high_damping": (
        "<strong>PointMaze High Damping</strong> changes only joint damping to 20; "
        "it inherits parameters from <code>maze_medium</code>."
    ),
    "maze_diverse": (
        "<strong>DiverseMaze</strong> (<code>maze_diverse</code>) evaluates held-out "
        "layouts using the <code>mediummaze_dynamics_shift</code> base and fixed "
        "BFS-distance-controlled goal corpora. Parameters are frozen on "
        "<code>maze_medium</code>."
    ),
}

METRICS_EXPLAINER = """
<p><strong>Success rate is the weakest column here.</strong> At these sample sizes its binomial
standard error is around 0.02, so adjacent rows are usually not separable on it. The continuous
columns carry far more information:</p>
<ul>
  <li><strong>median dist Δ</strong> — median paired change in final distance-to-goal vs. the frozen
  model on the same episodes (both-fail episodes only). Negative is better.</li>
  <li><strong>catastrophe</strong> — fraction of episodes ending more than 2× further from the goal
  than frozen. How often adapting actively hurts.</li>
  <li><strong>regret</strong> — fraction of episodes where adapting ended further from the goal than
  not adapting, and the 90th-percentile size of that loss.</li>
  <li><strong>adapt s/replan</strong> and <strong>peak MB</strong> — adaptation cost around the
  adapter hooks only.</li>
</ul>
<p>Intervals are 95% percentile bootstrap over episodes (2000 resamples). There is deliberately
no overall rank — sort by whichever column matters for your use case.</p>
"""


def esc(text: str | None) -> str:
    return html.escape(text or "", quote=True)


def load_yaml_simple(path: Path) -> dict:
    """Minimal YAML reader for method.yaml (no external deps)."""
    text = path.read_text()
    data: dict = {}
    current_key: str | None = None
    current_block: list[str] = []
    in_block = False

    def flush_block() -> None:
        nonlocal current_key, current_block, in_block
        if current_key is None:
            return
        if in_block:
            data[current_key] = " ".join(s.strip() for s in current_block).strip()
        current_block = []
        in_block = False

    for line in text.splitlines():
        if not line.strip() or line.strip().startswith("#"):
            continue
        if re.match(r"^\s", line) and in_block:
            current_block.append(line.strip())
            continue
        flush_block()
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if value == ">":
            current_key = key
            in_block = True
            continue
        current_key = key
        if value.startswith("[") and value.endswith("]"):
            inner = value[1:-1].strip()
            data[key] = [v.strip() for v in inner.split(",") if v.strip()] if inner else []
        elif value in ("true", "false"):
            data[key] = value == "true"
        else:
            data[key] = value.strip('"')
    flush_block()
    return data


def parse_leaderboard_tables() -> dict[str, list[dict[str, str]]]:
    content = LEADERBOARD.read_text()
    sections: dict[str, list[dict[str, str]]] = {}
    current_setting: str | None = None
    headers: list[str] = []

    for line in content.splitlines():
        setting_match = re.match(r"^## (\w+)$", line)
        if setting_match:
            current_setting = setting_match.group(1)
            sections[current_setting] = []
            headers = []
            continue
        if not current_setting or not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if all(set(c) <= {"-", " "} for c in cells):
            continue
        if cells[0].lower() == "method":
            headers = [h.lower() for h in cells]
            continue
        if not headers:
            continue
        row = {headers[i]: cells[i] if i < len(cells) else "" for i in range(len(headers))}
        method = row.get("method", "")
        # Match the paper: one Frozen row; hide the episode-isolated frozen arm.
        if method == "Frozen (Individual)":
            continue
        if method == "Frozen (Batched)":
            row["method"] = "Frozen"
        sections[current_setting].append(row)
    return sections


def load_method_info() -> dict[str, dict]:
    info: dict[str, dict] = {}
    for name in METHOD_ORDER:
        if name in FROZEN_META:
            info[name] = dict(FROZEN_META[name])
            continue
        method_dir = METHODS / name
        yaml_data = load_yaml_simple(method_dir / "method.yaml")
        readme_path = method_dir / "README.md"
        info[name] = {
            **yaml_data,
            "readme": readme_path.read_text() if readme_path.exists() else "",
        }
    return info


def display_to_slug(display_name: str, methods: dict[str, dict]) -> str | None:
    for slug, meta in methods.items():
        if meta.get("display_name") == display_name:
            return slug
    return None


def vs_frozen_class(value: str) -> str:
    if value in ("—", "-", ""):
        return ""
    if value.startswith("+"):
        return "delta-pos"
    if value.startswith("-"):
        return "delta-neg"
    return ""


def page_shell(title: str, body: str, *, depth: int = 0) -> str:
    prefix = "../" * depth
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{esc(title)} — PAARBench</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="{prefix}css/style.css">
</head>
<body>
  <header class="site-header">
    <div class="site-header-inner">
      <h1><a href="{prefix}index.html" style="color:inherit;text-decoration:none;">PAARBench</a></h1>
      <p class="tagline">PlanActAdaptRepeatBench — test-time adaptation for latent world models in closed-loop control</p>
      <nav class="site-nav">
        <a href="{prefix}index.html#about">About</a>
        <a href="{prefix}index.html#settings">Settings</a>
        <a href="{prefix}index.html#leaderboards">Leaderboards</a>
        <a href="{prefix}index.html#contribute">Contribute</a>
      </nav>
    </div>
  </header>
  <main class="page-main">
    {body}
  </main>
  <footer class="site-footer">
    <div class="site-footer-inner">
      <p>PAARBench — evaluate adaptation methods under a fixed MPC planner with transparent hyperparameter-selection cost.</p>
    </div>
  </footer>
  <script src="{prefix}js/sort-tables.js" defer></script>
</body>
</html>
"""


LEADERBOARD_HEAD = """
        <tr>
          <th data-type="text">Method</th>
          <th class="num" data-type="number">Success</th>
          <th class="num" data-type="number">vs Frozen</th>
          <th class="num" data-type="number">Median dist Δ</th>
          <th class="num" data-type="number">Catastrophe</th>
          <th class="num" data-type="number">Regret</th>
          <th class="num" data-type="number">Adapt s/replan</th>
          <th class="num" data-type="number">Peak MB</th>
        </tr>"""


def format_success(row: dict[str, str]) -> str:
    """Render success with binomial SE in parentheses, e.g. ``0.485 (±0.020)``."""
    success = row.get("success", "").strip()
    se = row.get("±1 se", row.get("±1 SE", "")).strip()
    if not success or success.startswith("*"):
        return success
    if se and se not in ("—", "-", ""):
        return f"{success} (±{se.lstrip('±')})"
    return success


def render_leaderboard_row(row: dict[str, str], slug: str | None) -> str:
    method_name = row.get("method", "")
    is_baseline = slug in FROZEN_SLUGS
    row_class = ' class="baseline-row"' if is_baseline else ""
    if slug:
        method_cell = (
            f'<a class="method-link" href="methods/{esc(slug)}.html">{esc(method_name)}</a>'
        )
    else:
        method_cell = f'<span class="method-link">{esc(method_name)}</span>'

    vs = row.get("vs frozen", "—")
    vs_class = vs_frozen_class(vs)

    return f"""<tr{row_class}>
  <td>{method_cell}</td>
  <td class="num">{esc(format_success(row))}</td>
  <td class="num {vs_class}">{esc(vs)}</td>
  <td class="num">{esc(row.get("median dist δ [95% ci]", row.get("median dist Δ [95% CI]", "")))}</td>
  <td class="num">{esc(row.get("catastrophe [95% ci]", row.get("catastrophe [95% CI]", "")))}</td>
  <td class="num">{esc(row.get("regret", ""))}</td>
  <td class="num">{esc(row.get("adapt s/replan", ""))}</td>
  <td class="num">{esc(row.get("peak mb", row.get("peak MB", "")))}</td>
</tr>"""


def render_index(methods: dict[str, dict], tables: dict[str, list[dict]]) -> str:
    leaderboard_html = []
    for setting_id in SETTING_ORDER:
        info = SETTING_INFO[setting_id]
        rows = tables.get(setting_id, [])
        row_html = []
        for row in rows:
            slug = display_to_slug(row.get("method", ""), methods)
            row_html.append(render_leaderboard_row(row, slug))
        tbody = (
            "\n        " + "\n        ".join(row_html) + "\n      "
            if row_html else ""
        )
        leaderboard_html.append(f"""
<section class="setting-block" id="{setting_id}">
  <h3>{esc(info["title"])}</h3>
  <p class="setting-meta">{esc(info["description"])}</p>
  <div class="table-wrap">
    <table class="sortable">
      <thead>
{LEADERBOARD_HEAD}
      </thead>
      <tbody>{tbody}</tbody>
    </table>
  </div>
</section>
""")

    body = f"""
<section id="about">
  <h2>About the Benchmark</h2>
  <div class="about-copy">
    <p class="lead">
      PAARBench (PlanActAdaptRepeatBench) evaluates <strong>test-time adaptation (TTA) strategies</strong>
      for latent world models in closed-loop control. Each submission declares at most two tunable
      hyperparameters; the harness runs a fixed selection protocol (3-point grid per axis on the
      selection cohort, up to two boundary expansions), freezes the best configuration, and
      evaluates once on held-out test cohorts.
    </p>
    <p>
      The benchmark asks whether a world model can adapt online during MPC replanning — and whether
      that adaptation helps control without compounding error, catastrophic failures, or hidden
      tuning cost. Tasks include <strong>PushObj</strong> (pushing objects of varied shapes),
      <strong>PushT</strong> (visual pushing with distribution shift), and
      <strong>PointMaze / DiverseMaze</strong> (navigation under dynamics and layout shift).
    </p>
    <p>
      Planning is held fixed within each setting family (Push*: goal horizon 25; maze: goal
      horizon 50), with 100 gradient-descent steps, zero action initialization, and no action
      noise, so episodes are deterministic and paired comparisons against the frozen baseline
      are meaningful.
    </p>
    <p>
      Results are reported as a multi-objective frontier — success rate (with binomial SE in
      parentheses), paired distance change, catastrophe rate, regret, adaptation latency, and peak
      memory — rather than a single rank.
    </p>
  </div>
</section>

<section id="how-it-works">
  <h2>How It Works</h2>
  <div class="prose">
    <p>The evaluation protocol has four stages:</p>
    <ol>
      <li><strong>Selection</strong> — On settings with a selection cohort, the standard protocol
      evaluates a 3-point grid per tunable axis (up to two boundary expansions). Every cell is
      counted as selection cost.</li>
      <li><strong>Freeze</strong> — The best hyperparameters (per the rule's objective) are frozen.</li>
      <li><strong>Test</strong> — The frozen configuration runs once on each held-out test cohort.
      For <code>pushobj_shift</code> and the maze OOD settings, parameters are inherited from the
      nominal setting with no new selection.</li>
      <li><strong>Record</strong> — Compact results go to <code>results/&lt;method&gt;/&lt;setting&gt;.json</code>.
      All continuous metrics are paired against the frozen baseline on the same episodes.</li>
    </ol>
    <p>Methods that own mutable shared state (model weights plus an optimizer) must declare
    <code>requires_episode_isolation: true</code> so each episode is evaluated independently.</p>
  </div>
</section>

<section id="settings">
  <h2>Settings</h2>
  <div class="prose">
    <p>{SETTINGS_INTRO}</p>
    <p>{SETTINGS_PARAGRAPHS["pushobj"]}</p>
    <p>{SETTINGS_PARAGRAPHS["pushobj_shift"]}</p>
    <p>{SETTINGS_PARAGRAPHS["pusht"]}</p>
    <p>{SETTINGS_PARAGRAPHS["maze_medium"]}</p>
    <p>{SETTINGS_PARAGRAPHS["maze_medium_low_density"]}</p>
    <p>{SETTINGS_PARAGRAPHS["maze_medium_high_damping"]}</p>
    <p>{SETTINGS_PARAGRAPHS["maze_diverse"]}</p>
  </div>
</section>

<section id="leaderboards">
  <h2>Leaderboards</h2>
  <p class="lead">Multi-objective by design — click a column header to sort, or a method name for details.</p>
  {"".join(leaderboard_html)}
  <div class="metrics-explainer">{METRICS_EXPLAINER}</div>
</section>

<section id="contribute">
  <h2>How to Contribute</h2>
  <p class="lead">A submission is an adaptation method with its hyperparameters chosen by the
  fixed selection protocol, evaluated under that protocol, with its result record included.</p>

  <div class="card" style="margin-bottom:1.5rem;">
    <h3 style="margin-top:0;">Prerequisites</h3>
    <div class="table-wrap table-wrap--center">
      <table class="settings-table">
        <thead>
          <tr><th>Artifact</th><th>How to get it</th><th>Needed for</th></tr>
        </thead>
        <tbody>
          <tr>
            <td>Base + adapter checkpoints</td>
            <td><code>scripts/download_checkpoints.py all</code> (see <code>docs/CHECKPOINTS.md</code>)</td>
            <td>Everything</td>
          </tr>
          <tr>
            <td>Push goal files <code>data/pushobj_eval/val_&lt;shape&gt;/plan_targets.pkl</code></td>
            <td><code>scripts/download_targets.py</code></td>
            <td>Push settings</td>
          </tr>
          <tr>
            <td>Maze episode corpora <code>data/maze_eval/*/seed_*.pkl</code></td>
            <td>Tracked in git, or <code>scripts/generate_maze_targets.py</code></td>
            <td>Maze settings</td>
          </tr>
          <tr>
            <td>Maze MuJoCo runtime + staged Drive assets</td>
            <td><code>scripts/setup_mujoco_runtime.sh</code> + <code>docs/MAZE.md</code></td>
            <td>Maze settings</td>
          </tr>
          <tr>
            <td>Training trajectories</td>
            <td><code>scripts/download_data.py</code> (push); maze via <code>docs/MAZE.md</code></td>
            <td>Methods using offline data only</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>

  <ol class="contribute-steps">
    <li>
      <strong>Create your method</strong>
      Copy <code>methods/_template/</code> to <code>methods/my_method/</code> and implement
      <code>adapter.py</code>, <code>method.yaml</code>, and optionally a <code>tunable:</code>
      block. No benchmark registry or planner changes are required.
    </li>
    <li>
      <strong>Validate locally</strong>
<pre><code>.venv/bin/python scripts/validate_method.py my_method
.venv/bin/python -m pytest -q</code></pre>
    </li>
    <li>
      <strong>Run the full protocol</strong>
<pre><code>.venv/bin/python scripts/evaluate.py my_method --setting pushobj --gpus 0,1,2,3</code></pre>
      Also run the frozen baseline for paired metrics:
<pre><code>.venv/bin/python scripts/evaluate.py --frozen --setting pushobj</code></pre>
    </li>
    <li>
      <strong>Open a pull request</strong>
      Include your <code>methods/&lt;name&gt;/</code> directory (with README), the
      <code>results/&lt;name&gt;/&lt;setting&gt;.json</code> records, and your row(s) added to
      <code>LEADERBOARD.md</code>. Do not regenerate the full leaderboard — add only your rows.
    </li>
  </ol>

  <p style="margin-top:1.5rem;color:var(--text-muted);font-size:0.95rem;">
    Full adapter interface documentation is in <code>methods/README.md</code> and
    <code>CONTRIBUTING.md</code> in the repository.
  </p>
</section>
"""
    return page_shell("Home", body)


def load_result_summary(
    slug: str, tables: dict[str, list[dict]], methods: dict[str, dict],
) -> dict[str, dict]:
    summary: dict[str, dict] = {}
    method_results = RESULTS / slug
    if method_results.exists():
        for path in sorted(method_results.glob("*.json")):
            data = json.loads(path.read_text())
            summary[path.stem] = {
                "success": data.get("success"),
                "n": data.get("n"),
                "frozen_reference": data.get("frozen_reference"),
                "selection_cost_columns": data.get("selection_cost_columns"),
                "selection_rule": data.get("selection_rule", ""),
                "complete": data.get("complete", False),
                "params": data.get("params", {}),
            }
    # The repository intentionally tracks the leaderboard even when bulky result
    # records are absent. Preserve those published summaries when regenerating the
    # website from such a checkout instead of replacing every method page with
    # "No result records".
    if not summary:
        for setting_id, rows in tables.items():
            for row in rows:
                if display_to_slug(row.get("method", ""), methods) != slug:
                    continue
                try:
                    success = float(row.get("success", ""))
                except ValueError:
                    success = None
                try:
                    n = int(row.get("n", ""))
                except ValueError:
                    n = None
                summary[setting_id] = {"success": success, "n": n}
                break
    return summary


def params_for_method(slug: str, meta: dict, results: dict[str, dict]) -> dict:
    for res in results.values():
        params = res.get("params")
        if params:
            return params
    raw = meta.get("params", {})
    return raw if isinstance(raw, dict) else {}


def markdown_to_html(text: str) -> str:
    """Small markdown subset for method READMEs."""
    lines = text.strip().splitlines()
    out: list[str] = []
    in_para: list[str] = []

    def fmt_inline(s: str) -> str:
        s = esc(s)
        s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
        s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
        return s

    def flush_para() -> None:
        if in_para:
            out.append(f"<p>{fmt_inline(' '.join(in_para))}</p>")
            in_para.clear()

    for line in lines:
        stripped = line.strip()
        if not stripped:
            flush_para()
            continue
        if stripped.startswith("# "):
            flush_para()
            continue
        in_para.append(stripped)
    flush_para()
    return "\n".join(out)


def render_method_page(
    slug: str, meta: dict, tables: dict[str, list[dict]], methods: dict[str, dict]
) -> str:
    display = meta.get("display_name", slug)
    description = meta.get("description", "")
    reference = meta.get("reference", "")
    settings = meta.get("settings", [])
    if isinstance(settings, str):
        settings = [settings]
    results = load_result_summary(slug, tables, methods)
    params = params_for_method(slug, meta, results)
    selection = meta.get("selection", "")
    if meta.get("tunable"):
        selection = "standard tunable axes"
    isolated = meta.get("requires_episode_isolation", False)
    readme = meta.get("readme", "")

    badges = []
    if isolated:
        badges.append('<span class="badge badge-accent">Episode isolated</span>')
    if selection:
        badges.append(f'<span class="badge">Selection: {esc(selection)}</span>')
    else:
        badges.append('<span class="badge">No selection rule</span>')
    for s in settings:
        badges.append(f'<span class="badge">{esc(s)}</span>')

    param_items = "".join(
        f"<li><span>{esc(k)}</span> = {esc(str(v))}</li>" for k, v in sorted(params.items())
    )

    result_rows = []
    for setting_id in settings:
        res = results.get(setting_id, {})
        if not res:
            continue
        frozen = res.get("frozen_reference")
        success = res.get("success")
        vs = ""
        if success is not None and frozen is not None:
            vs = f"{success - frozen:+.3f}"
        success_str = f"{success:.3f}" if isinstance(success, float) else esc(str(success))
        result_rows.append(f"""<tr>
  <td><code>{esc(setting_id)}</code></td>
  <td class="num">{success_str}</td>
  <td class="num">{esc(vs)}</td>
  <td class="num">{res.get("n", "")}</td>
</tr>""")

    leaderboard_sections = []
    for setting_id in settings:
        rows = tables.get(setting_id, [])
        for row in rows:
            if display_to_slug(row.get("method", ""), methods) == slug:
                leaderboard_sections.append(f"""
<h3>{esc(SETTING_INFO.get(setting_id, {}).get("title", setting_id))}</h3>
<div class="table-wrap">
  <table class="sortable">
    <thead>
{LEADERBOARD_HEAD}
    </thead>
    <tbody>
      {render_leaderboard_row(row, slug)}
    </tbody>
  </table>
</div>
""")
                break

    readme_html = markdown_to_html(readme) if readme else ""

    body = f"""
<a class="back-link" href="../index.html">← Back to leaderboard</a>

<div class="method-hero">
  <h1>{esc(display)}</h1>
  <p class="lead">{esc(description)}</p>
  {"<p class='reference'><strong>Reference:</strong> " + esc(reference) + "</p>" if reference else ""}
  <div class="badge-row">{"".join(badges)}</div>
</div>

<section>
  <h2>Overview</h2>
  <div class="card prose">
    {readme_html if readme_html else f"<p>{esc(description)}</p>"}
  </div>
</section>

<section>
  <h2>Frozen Hyperparameters</h2>
  {"<ul class='param-list'>" + param_items + "</ul>" if param_items else "<p>No tunable hyperparameters.</p>"}
</section>

<section>
  <h2>Results Summary</h2>
  {"<div class='table-wrap'><table><thead><tr><th>Setting</th><th class='num'>Success</th><th class='num'>vs Frozen</th><th class='num'>n</th></tr></thead><tbody>" + "".join(result_rows) + "</tbody></table></div>" if result_rows else "<p>No result records checked in for this method.</p>"}
</section>

<section>
  <h2>Leaderboard Row</h2>
  {"".join(leaderboard_sections) if leaderboard_sections else "<p>Not on the leaderboard for any setting.</p>"}
</section>
"""
    return page_shell(display, body, depth=1)


def main() -> None:
    methods = load_method_info()
    tables = parse_leaderboard_tables()

    (WEBSITE / "methods").mkdir(parents=True, exist_ok=True)

    index_html = render_index(methods, tables)
    (WEBSITE / "index.html").write_text(index_html)

    for slug in METHOD_ORDER:
        page = render_method_page(slug, methods[slug], tables, methods)
        (WEBSITE / "methods" / f"{slug}.html").write_text(page)

    print(f"Generated website in {WEBSITE}")


if __name__ == "__main__":
    main()
