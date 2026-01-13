from __future__ import annotations

import csv
import json
import os
import sys
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass(frozen=True)
class SelectionResult:
    method: str
    selected: bool
    submission_path: str | None
    workspace_path: str | None
    info: dict[str, Any] | None
    error: str | None = None


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _copy_file(src: Path, dst: Path) -> None:
    _safe_mkdir(dst.parent)
    shutil.copy2(src, dst)


def _copy_submission_from_workspace(workspace_path: Path, out_dir: Path) -> Path | None:
    src = workspace_path / "submission.csv"
    if not src.is_file():
        return None
    dst = out_dir / "submission.csv"
    _copy_file(src, dst)
    return dst


def _copy_workspace_artifacts(workspace_path: Path, out_dir: Path) -> None:
    """
    Copy key "what happened?" artifacts for debugging/understanding a selected submission.

    These files are produced by the RD-Agent DS runner/evaluator in typical runs.
    """

    out_dir.mkdir(parents=True, exist_ok=True)

    # Common files at workspace root.
    for name in [
        "main.py",
        "runtime_info.py",
        "package_info.py",
        "trace.log",
        "scores.csv",
        "aide_metrics.json",
        "stdout.txt",
        "stderr.txt",
        "EDA.md",
        "mle_score.txt",
    ]:
        src = workspace_path / name
        if src.is_file():
            _copy_file(src, out_dir / name)

    # Test/validation outputs (if present).
    test_dir = workspace_path / "test"
    if test_dir.is_dir():
        for fp in test_dir.rglob("*"):
            if not fp.is_file() or fp.is_symlink():
                continue
            rel = fp.relative_to(workspace_path)
            _copy_file(fp, out_dir / rel)


def _copy_workspace_code(workspace_path: Path, out_dir: Path) -> None:
    allowed_suffixes = {".py", ".md", ".txt", ".yaml", ".yml", ".json", ".toml"}

    for fp in workspace_path.rglob("*"):
        if not fp.is_file():
            continue
        if fp.is_symlink():
            continue
        if fp.suffix.lower() not in allowed_suffixes:
            continue
        rel = fp.relative_to(workspace_path)
        _copy_file(fp, out_dir / rel)


def _load_latest_trace(log_path: Path):
    from rdagent.log.storage import FileStorage

    storage = FileStorage(log_path)
    latest = None
    for msg in storage.iter_msg(tag="trace"):
        latest = msg.content
    return latest


def _load_latest_sota_exp(log_path: Path):
    from rdagent.log.storage import FileStorage

    storage = FileStorage(log_path)
    latest = None
    for msg in storage.iter_msg(tag="sota_exp_to_submit"):
        latest = msg.content
    return latest


def _coerce_float(x: object) -> float | None:
    if x is None:
        return None
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if s.startswith("tensor(") and s.endswith(")"):
        s = s[len("tensor(") : -1]
    try:
        return float(s)
    except ValueError:
        return None


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    # Rows may have slightly different schemas (e.g., optional `selection_info` / grading fields).
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row.keys():
            if k in seen:
                continue
            fieldnames.append(k)
            seen.add(k)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_trace_tree_artifacts(log_path: Path, trace, highlight_indices: list[int]) -> None:
    _write_tree_plot_html(log_path, trace, highlight_indices=highlight_indices)


def _read_text(path: Path, *, max_chars: int = 200_000) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n...[truncated]...\n"


def _dag_depths(dag_parent: list[list[int]]) -> list[int]:
    n = len(dag_parent)
    depths = [0] * n
    changed = True
    # Relax until convergence (DAG is expected; this is safe and simple).
    while changed:
        changed = False
        for i, parents in enumerate(dag_parent):
            if not parents:
                continue
            cand = 1 + max(depths[p] for p in parents if 0 <= p < n)
            if cand != depths[i]:
                depths[i] = cand
                changed = True
    return depths


def _tree_layout(dag_parent: list[list[int]]) -> list[list[float]]:
    n = len(dag_parent)
    if n == 0:
        return []
    depths = _dag_depths(dag_parent)
    max_depth = max(depths) if depths else 0
    by_depth: dict[int, list[int]] = {}
    for i, d in enumerate(depths):
        by_depth.setdefault(d, []).append(i)

    idx_in_layer: dict[int, tuple[int, int]] = {}
    for d, nodes in by_depth.items():
        for j, i in enumerate(nodes):
            idx_in_layer[i] = (j, len(nodes))

    layout: list[list[float]] = []
    for i, d in enumerate(depths):
        j, k = idx_in_layer.get(i, (0, 1))
        x = (j + 1) / (k + 1)
        y = 0.0 if max_depth == 0 else float(d) / float(max_depth)
        layout.append([float(x), float(y)])
    return layout


def _build_tree_payload_from_trace(
    *,
    trace,
    log_path: Path,
    highlight_indices: list[int],
) -> dict[str, object]:
    dag_parent = getattr(trace, "dag_parent", []) or []
    if not isinstance(dag_parent, list):
        dag_parent = []
    edges: list[list[int]] = []
    for child, parents in enumerate(dag_parent):
        if not isinstance(parents, list):
            continue
        for p in parents:
            if isinstance(p, int):
                edges.append([p, child])

    layout = _tree_layout(dag_parent)

    maximize = bool(getattr(getattr(trace, "scen", None), "metric_direction", True))
    idx2loop_id = getattr(trace, "idx2loop_id", {}) if hasattr(trace, "idx2loop_id") else {}
    if not isinstance(idx2loop_id, dict):
        idx2loop_id = {}

    nodes: list[dict[str, object]] = []
    hist = getattr(trace, "hist", []) or []
    for idx, item in enumerate(hist):
        exp = None
        fb = None
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            exp, fb = item[0], item[1]
        ws_path = None
        try:
            ws_path = Path(exp.experiment_workspace.workspace_path).resolve() if exp is not None else None
        except Exception:
            ws_path = None

        valid_metric = _coerce_float(getattr(exp, "valid_metric", None)) if exp is not None else None
        cv_mean = _coerce_float(getattr(exp, "cv_mean", None)) if exp is not None else None
        cv_std = _coerce_float(getattr(exp, "cv_std", None)) if exp is not None else None
        cv_folds = getattr(exp, "cv_folds", None) if exp is not None else None
        if isinstance(cv_folds, (list, tuple)):
            cv_folds = [_coerce_float(v) for v in cv_folds]

        code_text = ""
        stdout_text = ""
        if ws_path is not None and ws_path.is_dir():
            code_text = _read_text(ws_path / "main.py", max_chars=200_000)
            stdout_text = _read_text(ws_path / "stdout.txt", max_chars=200_000)

        hypothesis_text = ""
        try:
            hypothesis_text = str(getattr(exp, "hypothesis", "") or "")
        except Exception:
            hypothesis_text = ""

        plan_text = ""
        try:
            plan = getattr(exp, "plan", None)
            if plan is not None:
                plan_text = json.dumps(plan, indent=2, default=str, sort_keys=True)
        except Exception:
            plan_text = ""

        feedback_text = ""
        is_buggy = False
        try:
            if fb is not None:
                feedback_text = str(fb)
                is_buggy = bool(getattr(fb, "exception", None)) or (getattr(fb, "decision", True) is False)
        except Exception:
            feedback_text = ""

        ws_rel = None
        if ws_path is not None:
            try:
                ws_rel = os.path.relpath(ws_path, log_path)
            except Exception:
                ws_rel = str(ws_path)

        nodes.append(
            {
                "id": idx,
                "loop_id": idx2loop_id.get(idx),
                "workspace_path": str(ws_path) if ws_path is not None else None,
                "workspace_rel": ws_rel,
                "valid_metric": valid_metric,
                "cv_mean": cv_mean,
                "cv_std": cv_std,
                "cv_folds": cv_folds,
                "maximize": maximize,
                "hypothesis": hypothesis_text,
                "plan": plan_text,
                "code": code_text,
                "stdout": stdout_text,
                "feedback": feedback_text,
                "is_buggy": is_buggy,
                "highlight": idx in set(highlight_indices),
            }
        )

    # AIDE-style "seen nodes": use ancestors (transitively) as a proxy.
    ancestors: list[list[int]] = []
    parents_map = {i: (dag_parent[i] if i < len(dag_parent) and isinstance(dag_parent[i], list) else []) for i in range(len(nodes))}
    for i in range(len(nodes)):
        seen: set[int] = set()
        stack = list(parents_map.get(i, []))
        while stack:
            p = stack.pop()
            if not isinstance(p, int) or p in seen:
                continue
            seen.add(p)
            stack.extend(parents_map.get(p, []))
        ancestors.append(sorted(seen))

    return {
        "meta": {"type": "trace", "maximize": maximize},
        "nodes": nodes,
        "edges": edges,
        "layout": layout,
        "seen_nodes_per_node": ancestors,
    }


def _build_tree_payload_from_workspace_scan(
    *,
    candidates_info: list[dict[str, object]],
    log_path: Path,
    maximize: bool,
    highlight_indices: list[int] | None = None,
) -> dict[str, object]:
    highlight = set(highlight_indices or [])
    nodes: list[dict[str, object]] = []
    for row in candidates_info:
        idx = int(row.get("candidate_idx") or 0)
        ws_path = Path(str(row.get("workspace_path") or ""))
        ws_rel = None
        try:
            ws_rel = os.path.relpath(ws_path, log_path)
        except Exception:
            ws_rel = str(ws_path)
        nodes.append(
            {
                "id": idx,
                "loop_id": None,
                "workspace_path": str(ws_path),
                "workspace_rel": ws_rel,
                "valid_metric": row.get("valid_metric"),
                "cv_mean": None,
                "cv_std": None,
                "cv_folds": None,
                "maximize": maximize,
                "hypothesis": "",
                "plan": "",
                "code": _read_text(ws_path / "main.py", max_chars=200_000) if ws_path.is_dir() else "",
                "stdout": _read_text(ws_path / "stdout.txt", max_chars=200_000) if ws_path.is_dir() else "",
                "feedback": "",
                "is_buggy": False,
                "highlight": idx in highlight,
            }
        )
    return {
        "meta": {"type": "workspace_scan", "maximize": maximize},
        "nodes": nodes,
        "edges": [],
        "layout": [[0.5, 0.0]] if nodes else [],
        "seen_nodes_per_node": [[] for _ in nodes],
    }


def _write_tree_plot_html(log_path: Path, trace, *, highlight_indices: list[int]) -> None:
    """
    Write an AIDE-like `tree_plot.html` for RD-Agent runs.

    This is intentionally self-contained (no extra python deps, no selenium imports).
    """
    try:
        payload = _build_tree_payload_from_trace(trace=trace, log_path=log_path, highlight_indices=highlight_indices)
    except Exception as e:
        (log_path / "tree_plot_error.txt").write_text(f"{type(e).__name__}: {e}\n")
        payload = {"meta": {"type": "error", "error": str(e)}, "nodes": [], "edges": [], "layout": []}
    _write_tree_plot_html_from_payload(log_path, payload)


def _write_tree_plot_html_from_payload(log_path: Path, payload: dict[str, object]) -> None:
    # Keep a raw JSON alongside the HTML (useful for debugging and downstream tooling).
    (log_path / "tree_plot.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    # A lighter, AIDE-like "journal" (easy to skim).
    try:
        nodes = payload.get("nodes") if isinstance(payload, dict) else None
        light_nodes: list[dict[str, object]] = []
        if isinstance(nodes, list):
            for n in nodes:
                if not isinstance(n, dict):
                    continue
                light_nodes.append(
                    {
                        "id": n.get("id"),
                        "loop_id": n.get("loop_id"),
                        "valid_metric": n.get("valid_metric"),
                        "cv_mean": n.get("cv_mean"),
                        "cv_std": n.get("cv_std"),
                        "is_buggy": n.get("is_buggy"),
                        "highlight": n.get("highlight"),
                        "workspace_rel": n.get("workspace_rel"),
                        "hypothesis": n.get("hypothesis"),
                    }
                )
        journal = {
            "meta": payload.get("meta") if isinstance(payload, dict) else None,
            "edges": payload.get("edges") if isinstance(payload, dict) else None,
            "layout": payload.get("layout") if isinstance(payload, dict) else None,
            "nodes": light_nodes,
        }
        (log_path / "journal.json").write_text(
            json.dumps(journal, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
    except Exception:
        pass

    data_json = json.dumps(payload, ensure_ascii=False, default=str)
    html = "\n".join(
        [
            "<!doctype html>",
            "<html>",
            "<head>",
            '  <meta charset="utf-8" />',
            "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />",
            "  <title>RD-Agent Tree Plot</title>",
            "  <style>",
            "    body{font-family:system-ui,Arial,sans-serif;margin:0;display:flex;height:100vh}",
            "    #sidebar{width:380px;overflow:auto;border-right:1px solid #ddd;padding:16px;background:#fafafa}",
            "    #main{flex:1;display:flex;flex-direction:column}",
            "    #svgwrap{flex:1;overflow:auto;padding:16px}",
            "    #details{height:42vh;overflow:auto;border-top:1px solid #ddd;padding:16px}",
            "    .muted{color:#666;font-size:12px}",
            "    .node-item{padding:6px 8px;border-radius:6px;cursor:pointer;margin:4px 0;font-size:12px}",
            "    .node-item:hover{background:#eee}",
            "    .node-item.sel{background:#dbeafe}",
            "    .pill{display:inline-block;padding:2px 6px;border-radius:999px;background:#eee;font-size:11px;margin-left:6px}",
            "    pre{white-space:pre-wrap;word-break:break-word;background:#0b1020;color:#e5e7eb;padding:10px;border-radius:8px;font-size:11px}",
            "    a{color:#2563eb;text-decoration:none}",
            "    a:hover{text-decoration:underline}",
            "  </style>",
            "</head>",
            "<body>",
            "  <div id=\"sidebar\">",
            "    <h2 style=\"margin:0 0 6px 0;\">RD-Agent Tree Plot</h2>",
            "    <div class=\"muted\" id=\"metaLine\"></div>",
            "    <div class=\"muted\">Click nodes to inspect hypothesis/code/stdout.</div>",
            "    <div style=\"margin-top:10px;\">",
            "      <a href=\"../submission/selections/selection_summary.html\">selection_summary.html</a>",
            "      &nbsp;|&nbsp; <a href=\"console.log\">console.log</a>",
            "    </div>",
            "    <h3 style=\"margin-top:16px;\">Nodes</h3>",
            "    <div id=\"nodeList\"></div>",
            "  </div>",
            "  <div id=\"main\">",
            "    <div id=\"svgwrap\">",
            "      <svg id=\"tree\" width=\"1400\" height=\"800\" viewBox=\"0 0 1400 800\" xmlns=\"http://www.w3.org/2000/svg\"></svg>",
            "    </div>",
            "    <div id=\"details\">",
            "      <h3 style=\"margin:0 0 8px 0;\">Details</h3>",
            "      <div class=\"muted\">Select a node on the left or in the graph.</div>",
            "      <div id=\"detailBody\"></div>",
            "    </div>",
            "  </div>",
            "  <script>",
            f"const TREE = {data_json};",
            "const metaLine = document.getElementById('metaLine');",
            "if (metaLine) {",
            "  const t = TREE.meta && TREE.meta.type ? TREE.meta.type : 'unknown';",
            "  const msg = TREE.meta && TREE.meta.message ? ` — ${TREE.meta.message}` : '';",
            "  const err = TREE.meta && TREE.meta.error ? ` — error: ${TREE.meta.error}` : '';",
            "  metaLine.textContent = `source: ${t}${msg}${err}`;",
            "}",
            "const el = (tag, attrs={}, children=[]) => {",
            "  const n = document.createElement(tag);",
            "  for (const [k,v] of Object.entries(attrs)) {",
            "    if (k === 'class') n.className = v; else if (k === 'html') n.innerHTML = v; else n.setAttribute(k, v);",
            "  }",
            "  for (const c of children) n.appendChild(c);",
            "  return n;",
            "};",
            "const fmt = (x) => (x === null || x === undefined) ? '' : String(x);",
            "const metricStr = (n) => n.valid_metric === null || n.valid_metric === undefined ? 'N/A' : fmt(n.valid_metric);",
            "let selected = null;",
            "const nodeList = document.getElementById('nodeList');",
            "const svg = document.getElementById('tree');",
            "const detailBody = document.getElementById('detailBody');",
            "",
            "function renderList() {",
            "  nodeList.innerHTML = '';",
            "  (TREE.nodes || []).forEach((n) => {",
            "    const item = el('div', {class: 'node-item', 'data-id': n.id});",
            "    item.appendChild(document.createTextNode(`#${n.id}  metric=${metricStr(n)}`));",
            "    if (n.highlight) item.appendChild(el('span', {class: 'pill'}, [document.createTextNode('selected')]));",
            "    item.onclick = () => selectNode(n.id);",
            "    nodeList.appendChild(item);",
            "  });",
            "}",
            "",
            "function renderSvg() {",
            "  const W = 1400, H = 800;",
            "  svg.innerHTML = '';",
            "  const layout = TREE.layout || [];",
            "  const pos = (id) => {",
            "    const p = layout[id] || [0.5, 0.0];",
            "    return {x: 80 + p[0] * (W - 160), y: 60 + p[1] * (H - 120)};",
            "  };",
            "  (TREE.edges || []).forEach(([a,b]) => {",
            "    const pa = pos(a), pb = pos(b);",
            "    const line = document.createElementNS('http://www.w3.org/2000/svg','line');",
            "    line.setAttribute('x1', pa.x); line.setAttribute('y1', pa.y);",
            "    line.setAttribute('x2', pb.x); line.setAttribute('y2', pb.y);",
            "    line.setAttribute('stroke', '#999'); line.setAttribute('stroke-width','2');",
            "    svg.appendChild(line);",
            "  });",
            "  (TREE.nodes || []).forEach((n) => {",
            "    const p = pos(n.id);",
            "    const g = document.createElementNS('http://www.w3.org/2000/svg','g');",
            "    g.style.cursor = 'pointer';",
            "    const c = document.createElementNS('http://www.w3.org/2000/svg','circle');",
            "    c.setAttribute('cx', p.x); c.setAttribute('cy', p.y);",
            "    c.setAttribute('r', n.highlight ? 22 : 18);",
            "    c.setAttribute('fill', n.is_buggy ? '#fca5a5' : (n.highlight ? '#60a5fa' : '#93c5fd'));",
            "    c.setAttribute('stroke', '#111827'); c.setAttribute('stroke-width','1');",
            "    const t = document.createElementNS('http://www.w3.org/2000/svg','text');",
            "    t.setAttribute('x', p.x); t.setAttribute('y', p.y + 4);",
            "    t.setAttribute('text-anchor','middle'); t.setAttribute('font-size','12');",
            "    t.setAttribute('fill','#111827');",
            "    t.textContent = String(n.id);",
            "    g.appendChild(c); g.appendChild(t);",
            "    g.onclick = () => selectNode(n.id);",
            "    svg.appendChild(g);",
            "  });",
            "}",
            "",
            "function renderDetails(n) {",
            "  if (!n) { detailBody.innerHTML = ''; return; }",
            "  const ws = n.workspace_rel || n.workspace_path || '';",
            "  const wsLink = ws ? `<a href=\"${ws}\">${ws}</a>` : '';",
            "  const rows = [];",
            "  rows.push(`<div><b>Node</b>: #${n.id} <span class='pill'>metric ${metricStr(n)}</span></div>`);",
            "  if (n.loop_id !== null && n.loop_id !== undefined) rows.push(`<div><b>Loop</b>: ${fmt(n.loop_id)}</div>`);",
            "  if (ws) rows.push(`<div><b>Workspace</b>: ${wsLink}</div>`);",
            "  if (n.hypothesis) rows.push(`<h4>Hypothesis</h4><pre>${n.hypothesis}</pre>`);",
            "  if (n.plan) rows.push(`<h4>Plan</h4><pre>${n.plan}</pre>`);",
            "  if (n.code) rows.push(`<h4>Code (main.py)</h4><pre>${n.code}</pre>`);",
            "  if (n.stdout) rows.push(`<h4>Stdout</h4><pre>${n.stdout}</pre>`);",
            "  if (n.feedback) rows.push(`<h4>Feedback</h4><pre>${n.feedback}</pre>`);",
            "  detailBody.innerHTML = rows.join('\\n');",
            "}",
            "",
            "function selectNode(id) {",
            "  selected = id;",
            "  document.querySelectorAll('.node-item').forEach((d) => {",
            "    d.classList.toggle('sel', Number(d.getAttribute('data-id')) === id);",
            "  });",
            "  const n = (TREE.nodes || []).find((x) => x.id === id);",
            "  renderDetails(n);",
            "}",
            "",
            "renderList();",
            "renderSvg();",
            "if (TREE.nodes && TREE.nodes.length) selectNode(TREE.nodes[0].id);",
            "  </script>",
            "</body>",
            "</html>",
            "",
        ]
    )

    (log_path / "tree_plot.html").write_text(html, encoding="utf-8")
    # Convenience aliases (AIDE uses tree_plot.html; some scripts expect tree.html/trace_tree.html).
    for alias in ["tree.html", "trace_tree.html"]:
        (log_path / alias).write_text(
            "\n".join(
                [
                    "<!doctype html>",
                    "<html>",
                    "<head>",
                    '  <meta charset="utf-8" />',
                    f"  <meta http-equiv=\"refresh\" content=\"0; url=tree_plot.html\" />",
                    "  <title>RD-Agent Tree Plot</title>",
                    "</head>",
                    "<body>",
                    "  <p>Redirecting to <a href=\"tree_plot.html\">tree_plot.html</a>…</p>",
                    "</body>",
                    "</html>",
                    "",
                ]
            ),
            encoding="utf-8",
        )


def _html_escape(text: object) -> str:
    s = "" if text is None else str(text)
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _write_selection_summary_html(
    *,
    selections_root: Path,
    log_path: Path,
    selection_rows: list[dict[str, object]],
) -> None:
    trace_html = log_path / "tree_plot.html"
    if not trace_html.exists():
        trace_html = log_path / "trace_tree.html"
    trace_rel = os.path.relpath(trace_html, selections_root) if trace_html.exists() else None
    has_grading = (selections_root / "grading.csv").exists()

    headers = list(selection_rows[0].keys()) if selection_rows else []
    lines: list[str] = []
    lines += [
        "<!doctype html>",
        "<html>",
        "<head>",
        '  <meta charset="utf-8" />',
        "  <title>RD-Agent Selection Summary</title>",
        "  <style>",
        "    body{font-family:system-ui,Arial,sans-serif;margin:24px}",
        "    table{border-collapse:collapse;width:100%}",
        "    th,td{border:1px solid #ddd;padding:8px;font-size:12px;vertical-align:top}",
        "    th{background:#f6f6f6;text-align:left}",
        "    code{font-size:12px}",
        "  </style>",
        "</head>",
        "<body>",
        "  <h2>RD-Agent Selection Summary</h2>",
    ]
    if trace_rel:
        lines.append(f'  <p><a href="{_html_escape(trace_rel)}">Open trace_tree.html</a></p>')
    lines += [
        "  <p>"
        '<a href="selection_candidates.csv">selection_candidates.csv</a>'
        ' &nbsp;|&nbsp; <a href="selection_results.csv">selection_results.csv</a>'
        + (' &nbsp;|&nbsp; <a href="grading.csv">grading.csv</a>' if has_grading else "")
        + "</p>",
        "  <h3>Selected Submissions</h3>",
        "  <table>",
        "    <thead>",
        "      <tr>",
    ]
    lines += [f"        <th>{_html_escape(h)}</th>" for h in headers]
    lines += [
        "      </tr>",
        "    </thead>",
        "    <tbody>",
    ]

    for row in selection_rows:
        method = str(row.get("method") or "")
        method_dir = selections_root / method
        sub_rel = f"{method}/submission.csv" if (method_dir / "submission.csv").exists() else ""
        info_rel = f"{method}/selection_info.json" if (method_dir / "selection_info.json").exists() else ""

        lines.append("      <tr>")
        for h in headers:
            v = row.get(h)
            if h == "method" and method:
                cell = _html_escape(method)
                links = []
                if sub_rel:
                    links.append(f'<a href="{_html_escape(sub_rel)}">submission.csv</a>')
                if info_rel:
                    links.append(f'<a href="{_html_escape(info_rel)}">selection_info.json</a>')
                if links:
                    cell += "<br/>" + " ".join(links)
                lines.append(f"        <td>{cell}</td>")
            else:
                lines.append(f"        <td><code>{_html_escape(v)}</code></td>")
        lines.append("      </tr>")

    lines += [
        "    </tbody>",
        "  </table>",
        "</body>",
        "</html>",
        "",
    ]

    (selections_root / "selection_summary.html").write_text("\n".join(lines))


def _grade_selection_rows(
    *,
    selection_rows: list[dict[str, object]],
    selections_root: Path,
    competition_id: str,
    mlebench_data_dir: str | None,
) -> None:
    if not selection_rows:
        return
    if not competition_id or not mlebench_data_dir:
        return

    try:
        from mlebench.grade import grade_csv
        from mlebench.registry import Registry
    except Exception as e:
        (selections_root / "grading_error.txt").write_text(f"Failed to import MLEBench grading utilities: {e}\n")
        return

    try:
        registry = Registry().set_data_dir(Path(mlebench_data_dir))
        competition = registry.get_competition(competition_id)
    except Exception as e:
        (selections_root / "grading_error.txt").write_text(f"Failed to load competition `{competition_id}`: {e}\n")
        return

    grading_dir = selections_root / "grading"
    _safe_mkdir(grading_dir)

    grading_rows: list[dict[str, object]] = []
    grade_by_method: dict[str, dict[str, object]] = {}

    for row in selection_rows:
        method = str(row.get("method") or "")
        if not method:
            continue
        submission_path = selections_root / method / "submission.csv"
        if not submission_path.is_file():
            continue

        try:
            report = grade_csv(submission_path, competition)
            payload = report.to_dict()
            payload["method"] = method
            (grading_dir / f"{method}.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

            grade_payload: dict[str, object] = {
                "grade_score": payload.get("score"),
                "grade_valid_submission": payload.get("valid_submission"),
                "grade_any_medal": payload.get("any_medal"),
                "grade_gold_medal": payload.get("gold_medal"),
                "grade_silver_medal": payload.get("silver_medal"),
                "grade_bronze_medal": payload.get("bronze_medal"),
                "grade_above_median": payload.get("above_median"),
                "grade_submission_path": payload.get("submission_path"),
                "grade_error": None,
            }
            grade_by_method[method] = grade_payload
            grading_rows.append(
                {
                    "method": method,
                    "score": payload.get("score"),
                    "valid_submission": payload.get("valid_submission"),
                    "any_medal": payload.get("any_medal"),
                    "gold_medal": payload.get("gold_medal"),
                    "silver_medal": payload.get("silver_medal"),
                    "bronze_medal": payload.get("bronze_medal"),
                    "above_median": payload.get("above_median"),
                    "submission_path": payload.get("submission_path"),
                }
            )
        except Exception as e:
            payload = {"method": method, "submission_path": str(submission_path), "error": str(e)}
            (grading_dir / f"{method}.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            grade_by_method[method] = {
                "grade_score": None,
                "grade_valid_submission": None,
                "grade_any_medal": None,
                "grade_gold_medal": None,
                "grade_silver_medal": None,
                "grade_bronze_medal": None,
                "grade_above_median": None,
                "grade_submission_path": str(submission_path),
                "grade_error": str(e),
            }
            grading_rows.append(
                {
                    "method": method,
                    "score": None,
                    "valid_submission": False,
                    "any_medal": False,
                    "gold_medal": False,
                    "silver_medal": False,
                    "bronze_medal": False,
                    "above_median": False,
                    "submission_path": str(submission_path),
                    "error": str(e),
                }
            )

    if grading_rows:
        _write_csv(selections_root / "grading.csv", grading_rows)
        (selections_root / "grading.json").write_text(json.dumps(grading_rows, indent=2, sort_keys=True) + "\n")

    # Add grading columns into selection_results (for quick scanning in one table).
    grade_keys = [
        "grade_score",
        "grade_valid_submission",
        "grade_any_medal",
        "grade_gold_medal",
        "grade_silver_medal",
        "grade_bronze_medal",
        "grade_above_median",
        "grade_submission_path",
        "grade_error",
    ]
    for row in selection_rows:
        method = str(row.get("method") or "")
        grade_payload = grade_by_method.get(method, {})
        for k in grade_keys:
            row[k] = grade_payload.get(k)


def _read_ensemble_metric(scores_csv: Path) -> float | None:
    try:
        with scores_csv.open("r", newline="") as f:
            rows = list(csv.reader(f))
        if len(rows) < 2:
            return None
        # Expected: header like ",ROC AUC" then row like "ensemble,0.95"
        for row in rows[1:]:
            if len(row) >= 2 and row[0].strip().lower() == "ensemble":
                return _coerce_float(row[1])
    except Exception:
        return None
    return None


def _scan_workspace_candidates(workspace_root: Path) -> tuple[list[Path], list[dict[str, object]]]:
    candidates: list[Path] = []
    info_rows: list[dict[str, object]] = []
    for ws in sorted(workspace_root.iterdir()):
        if not ws.is_dir():
            continue
        sub = ws / "submission.csv"
        scores = ws / "scores.csv"
        if not sub.is_file() or not scores.is_file():
            continue
        valid_metric = _read_ensemble_metric(scores)
        idx = len(candidates)
        candidates.append(ws)
        info_rows.append(
            {
                "candidate_idx": idx,
                "workspace_path": str(ws.resolve()),
                "valid_metric": valid_metric,
                "scores_csv": str(scores.resolve()),
            }
        )
    return candidates, info_rows


def _infer_maximize_from_mlebench(competition_id: str, mlebench_data_dir: str | None) -> bool:
    if not competition_id or not mlebench_data_dir:
        return True
    try:
        from mlebench.leaderboard import get_leaderboard
        from mlebench.registry import Registry
    except Exception:
        return True
    try:
        competition = Registry().set_data_dir(Path(mlebench_data_dir)).get_competition(competition_id)
        leaderboard = get_leaderboard(competition)
        is_lower_better = competition.grader.is_lower_better(leaderboard)
        return not bool(is_lower_better)
    except Exception:
        return True


def _write_placeholder_tree(log_path: Path) -> None:
    html_path = log_path / "tree_plot.html"
    if html_path.exists():
        return
    _write_tree_plot_html_from_payload(
        log_path,
        {
            "meta": {
                "type": "placeholder",
                "message": "No serialized `trace` was found in this run (loop likely stopped early).",
            },
            "nodes": [],
            "edges": [],
            "layout": [],
            "seen_nodes_per_node": [],
        },
    )


def main() -> int:
    competition_id = os.getenv("COMPETITION_ID") or ""
    mlebench_data_dir = os.getenv("MLEBENCH_DATA_DIR")
    log_path = Path(os.environ["LOG_TRACE_PATH"]).resolve()
    submission_dir = Path(os.environ["SUBMISSION_DIR"]).resolve()
    code_dir = Path(os.environ["CODE_DIR"]).resolve()
    workspace_root = Path(os.environ["WORKSPACE_PATH"]).resolve() if os.getenv("WORKSPACE_PATH") else None

    _safe_mkdir(submission_dir)
    _safe_mkdir(code_dir)

    results: list[SelectionResult] = []

    # 1) Copy the auto-selected SOTA submission (whatever the DS loop chose).
    auto_ws: Path | None = None
    try:
        exp = _load_latest_sota_exp(log_path)
        if exp is not None and getattr(exp, "experiment_workspace", None) is not None:
            ws = Path(exp.experiment_workspace.workspace_path).resolve()
            auto_ws = ws
            copied = _copy_submission_from_workspace(ws, submission_dir)
            if copied is not None:
                results.append(
                    SelectionResult(
                        method="auto",
                        selected=True,
                        submission_path=str(copied),
                        workspace_path=str(ws),
                        info={"competition_id": competition_id},
                    )
                )
                _copy_workspace_code(ws, code_dir)
                # Also store under selections/auto for consistency.
                _copy_submission_from_workspace(ws, submission_dir / "selections" / "auto")
                _copy_workspace_artifacts(ws, submission_dir / "selections" / "auto")
            else:
                results.append(
                    SelectionResult(
                        method="auto",
                        selected=False,
                        submission_path=None,
                        workspace_path=str(ws),
                        info={"competition_id": competition_id},
                        error="submission.csv not found in selected workspace",
                    )
                )
        else:
            results.append(
                SelectionResult(
                    method="auto",
                    selected=False,
                    submission_path=None,
                    workspace_path=None,
                    info={"competition_id": competition_id},
                    error="no sota_exp_to_submit found in logs",
                )
            )
    except Exception as e:
        results.append(
            SelectionResult(
                method="auto",
                selected=False,
                submission_path=None,
                workspace_path=None,
                info={"competition_id": competition_id},
                error=f"failed to extract auto selection: {e}",
            )
        )

    # 2) Also emit per-strategy "post-search" selections (offline, from the final trace).
    selections_root = submission_dir / "selections"
    _safe_mkdir(selections_root)

    trace = None
    try:
        trace = _load_latest_trace(log_path)
    except Exception as e:
        results.append(
            SelectionResult(
                method="trace",
                selected=False,
                submission_path=None,
                workspace_path=None,
                info={"competition_id": competition_id},
                error=f"failed to load latest trace: {e}",
            )
        )

    if trace is not None:
        from rdagent.app.data_science.conf import DS_RD_SETTING
        from rdagent.scenarios.data_science.utils.post_search import select_final_experiment_with_info

        maximize = bool(getattr(getattr(trace, "scen", None), "metric_direction", True))
        idx2loop_id = getattr(trace, "idx2loop_id", {}) if hasattr(trace, "idx2loop_id") else {}
        if not isinstance(idx2loop_id, dict):
            idx2loop_id = {}

        candidates = []
        candidates_info: list[dict[str, object]] = []
        workspace_to_idx: dict[str, int] = {}
        for idx, item in enumerate(getattr(trace, "hist", []) or []):
            if not item:
                continue
            exp = item[0]
            if exp is None or getattr(exp, "result", None) is None:
                continue
            candidates.append(exp)

            ws = Path(exp.experiment_workspace.workspace_path).resolve()
            workspace_to_idx[str(ws)] = idx
            valid_metric = _coerce_float(getattr(exp, "valid_metric", None))
            if valid_metric is None:
                try:
                    valid_metric = _coerce_float(exp.result.loc["ensemble"].iloc[0])
                except Exception:
                    valid_metric = None

            cv_mean = _coerce_float(getattr(exp, "cv_mean", None))
            cv_std = _coerce_float(getattr(exp, "cv_std", None))
            cv_folds = getattr(exp, "cv_folds", None)
            cv_worst = None
            if isinstance(cv_folds, list) and cv_folds:
                folds = [_coerce_float(v) for v in cv_folds]
                folds = [v for v in folds if v is not None]
                if folds:
                    cv_worst = min(folds) if maximize else max(folds)

            candidates_info.append(
                {
                    "candidate_idx": idx,
                    "loop_id": idx2loop_id.get(idx),
                    "workspace_path": str(ws),
                    "valid_metric": valid_metric,
                    "cv_mean": cv_mean,
                    "cv_std": cv_std,
                    "cv_worst": cv_worst,
                    "has_cv_folds": bool(cv_folds),
                    "cv_validation_error": getattr(exp, "cv_validation_error", None),
                }
            )

        _write_csv(selections_root / "selection_candidates.csv", candidates_info)
        (selections_root / "selection_candidates.json").write_text(
            json.dumps(candidates_info, indent=2, sort_keys=True)
        )

        selection_methods = [
            "best_valid",
            "maximin",
            "elite_maximin",
            "mean_minus_k_std",
            "maximin_no_filter",
        ]

        top_k = int(getattr(DS_RD_SETTING, "post_search_top_k", 20) or 20)
        k_std = float(getattr(DS_RD_SETTING, "post_search_k_std", 2.0) or 2.0)
        elite_top_k = int(getattr(DS_RD_SETTING, "post_search_elite_top_k", 3) or 3)
        elite_ratio = float(getattr(DS_RD_SETTING, "post_search_elite_ratio", 0.05) or 0.05)
        elite_k_std = float(getattr(DS_RD_SETTING, "post_search_elite_k_std", 2.0) or 2.0)

        selection_rows: list[dict[str, object]] = []
        highlight_indices: list[int] = []
        if auto_ws is not None:
            auto_dir = selections_root / "auto"
            _safe_mkdir(auto_dir)
            _copy_submission_from_workspace(auto_ws, auto_dir)
            _copy_workspace_artifacts(auto_ws, auto_dir)
            (auto_dir / "workspace_path.txt").write_text(str(auto_ws) + "\n")
            (auto_dir / "selection_info.json").write_text(
                json.dumps({"method": "auto", "competition_id": competition_id}, indent=2, sort_keys=True)
            )

            sel_idx = workspace_to_idx.get(str(auto_ws))
            if isinstance(sel_idx, int):
                highlight_indices.append(sel_idx)
            selection_rows.append(
                {
                    "method": "auto",
                    "selected": True,
                    "selected_candidate_idx": sel_idx,
                    "selected_loop_id": idx2loop_id.get(sel_idx) if isinstance(sel_idx, int) else None,
                    "selected_workspace_path": str(auto_ws),
                    "error": None,
                }
            )

        for method in selection_methods:
            try:
                selected, info = select_final_experiment_with_info(
                    candidates,
                    maximize=maximize,
                    selection=method,
                    top_k=top_k,
                    k_std=k_std,
                    elite_top_k=elite_top_k,
                    elite_ratio=elite_ratio,
                    elite_k_std=elite_k_std,
                )
                if selected is None:
                    results.append(
                        SelectionResult(
                            method=method,
                            selected=False,
                            submission_path=None,
                            workspace_path=None,
                            info=info,
                            error="no candidate selected",
                        )
                    )
                    selection_rows.append(
                        {
                            "method": method,
                            "selected": False,
                            "selected_candidate_idx": None,
                            "selected_loop_id": None,
                            "selected_workspace_path": None,
                            "error": "no candidate selected",
                        }
                    )
                    continue

                ws = Path(selected.experiment_workspace.workspace_path).resolve()
                out_dir = selections_root / method
                copied = _copy_submission_from_workspace(ws, out_dir)
                _copy_workspace_artifacts(ws, out_dir)
                (out_dir / "selection_info.json").write_text(json.dumps(info, indent=2, sort_keys=True))
                (out_dir / "workspace_path.txt").write_text(str(ws) + "\n")

                sel_idx = workspace_to_idx.get(str(ws))
                if isinstance(sel_idx, int):
                    highlight_indices.append(sel_idx)

                results.append(
                    SelectionResult(
                        method=method,
                        selected=copied is not None,
                        submission_path=str(copied) if copied is not None else None,
                        workspace_path=str(ws),
                        info=info,
                        error=None if copied is not None else "submission.csv not found in selected workspace",
                    )
                )
                selection_rows.append(
                    {
                        "method": method,
                        "selected": copied is not None,
                        "selected_candidate_idx": sel_idx,
                        "selected_loop_id": idx2loop_id.get(sel_idx) if isinstance(sel_idx, int) else None,
                        "selected_workspace_path": str(ws),
                        "error": None if copied is not None else "submission.csv not found in selected workspace",
                        **({"selection_info": json.dumps(info, sort_keys=True)} if isinstance(info, dict) else {}),
                    }
                )
            except Exception as e:
                results.append(
                    SelectionResult(
                        method=method,
                        selected=False,
                        submission_path=None,
                        workspace_path=None,
                        info=None,
                        error=str(e),
                    )
                )
                selection_rows.append(
                    {
                        "method": method,
                        "selected": False,
                        "selected_candidate_idx": None,
                        "selected_loop_id": None,
                        "selected_workspace_path": None,
                        "error": str(e),
                    }
                )

        if selection_rows:
            _grade_selection_rows(
                selection_rows=selection_rows,
                selections_root=selections_root,
                competition_id=competition_id,
                mlebench_data_dir=mlebench_data_dir,
            )
            _write_csv(selections_root / "selection_results.csv", selection_rows)
            (selections_root / "selection_results.json").write_text(
                json.dumps(selection_rows, indent=2, sort_keys=True)
            )
            _write_selection_summary_html(
                selections_root=selections_root,
                log_path=log_path,
                selection_rows=selection_rows,
            )
            (log_path / "final_selection.json").write_text(
                json.dumps(
                    {"competition_id": competition_id, "selection_rows": selection_rows},
                    indent=2,
                    sort_keys=True,
                    default=str,
                )
                + "\n"
            )

        # Tree visualization (AIDE-style `tree.html`).
        _write_trace_tree_artifacts(log_path, trace, sorted(set(highlight_indices)))
    elif workspace_root is not None and workspace_root.is_dir():
        # Fallback: DS loop may stop before serializing `trace`/`sota_exp_to_submit`.
        candidates, candidates_info = _scan_workspace_candidates(workspace_root)
        _write_csv(selections_root / "selection_candidates.csv", candidates_info)
        (selections_root / "selection_candidates.json").write_text(
            json.dumps(candidates_info, indent=2, sort_keys=True) + "\n"
        )

        maximize = _infer_maximize_from_mlebench(competition_id, mlebench_data_dir)
        highlight_indices: list[int] = []

        selection_methods = [
            "best_valid",
            "maximin",
            "elite_maximin",
            "mean_minus_k_std",
            "maximin_no_filter",
        ]

        selection_rows: list[dict[str, object]] = []
        if candidates:
            # Pick a single best candidate by validation metric as a reasonable default.
            def _key(i_ws: tuple[int, Path]) -> float:
                idx, ws = i_ws
                m = _read_ensemble_metric(ws / "scores.csv")
                if m is None:
                    return float("-inf") if maximize else float("inf")
                return float(m)

            best_idx, best_ws = (
                max(list(enumerate(candidates)), key=_key)
                if maximize
                else min(list(enumerate(candidates)), key=_key)
            )
            highlight_indices = [best_idx]

            # Treat this as "auto" for artifact copying.
            auto_ws = best_ws.resolve()
            copied = _copy_submission_from_workspace(auto_ws, submission_dir)
            if copied is not None:
                _copy_workspace_code(auto_ws, code_dir)
                _copy_submission_from_workspace(auto_ws, selections_root / "auto")
                _copy_workspace_artifacts(auto_ws, selections_root / "auto")
                results.append(
                    SelectionResult(
                        method="auto",
                        selected=True,
                        submission_path=str(copied),
                        workspace_path=str(auto_ws),
                        info={"fallback": "workspace_scan", "competition_id": competition_id},
                        error=None,
                    )
                )
            else:
                results.append(
                    SelectionResult(
                        method="auto",
                        selected=False,
                        submission_path=None,
                        workspace_path=str(auto_ws),
                        info={"fallback": "workspace_scan", "competition_id": competition_id},
                        error="submission.csv not found in selected workspace",
                    )
                )
            selection_rows.append(
                {
                    "method": "auto",
                    "selected": copied is not None,
                    "selected_candidate_idx": best_idx,
                    "selected_loop_id": None,
                    "selected_workspace_path": str(auto_ws),
                    "error": None if copied is not None else "submission.csv not found in selected workspace",
                    "selection_info": json.dumps(
                        {"method": "auto", "fallback": "workspace_scan", "competition_id": competition_id},
                        sort_keys=True,
                    ),
                }
            )

            for method in selection_methods:
                out_dir = selections_root / method
                copied = _copy_submission_from_workspace(auto_ws, out_dir)
                _copy_workspace_artifacts(auto_ws, out_dir)
                (out_dir / "selection_info.json").write_text(
                    json.dumps(
                        {
                            "method": method,
                            "fallback": "workspace_scan",
                            "selected_candidate_idx": best_idx,
                            "selected_workspace_path": str(auto_ws),
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n"
                )
                (out_dir / "workspace_path.txt").write_text(str(auto_ws) + "\n")
                results.append(
                    SelectionResult(
                        method=method,
                        selected=copied is not None,
                        submission_path=str(copied) if copied is not None else None,
                        workspace_path=str(auto_ws),
                        info={"fallback": "workspace_scan", "competition_id": competition_id},
                        error=None if copied is not None else "submission.csv not found in selected workspace",
                    )
                )
                selection_rows.append(
                    {
                        "method": method,
                        "selected": copied is not None,
                        "selected_candidate_idx": best_idx,
                        "selected_loop_id": None,
                        "selected_workspace_path": str(auto_ws),
                        "error": None if copied is not None else "submission.csv not found in selected workspace",
                        "selection_info": json.dumps({"fallback": "workspace_scan"}, sort_keys=True),
                    }
                )

        if selection_rows:
            _grade_selection_rows(
                selection_rows=selection_rows,
                selections_root=selections_root,
                competition_id=competition_id,
                mlebench_data_dir=mlebench_data_dir,
            )
            _write_csv(selections_root / "selection_results.csv", selection_rows)
            (selections_root / "selection_results.json").write_text(
                json.dumps(selection_rows, indent=2, sort_keys=True) + "\n"
            )
            _write_selection_summary_html(
                selections_root=selections_root,
                log_path=log_path,
                selection_rows=selection_rows,
            )
            (log_path / "final_selection.json").write_text(
                json.dumps(
                    {"competition_id": competition_id, "selection_rows": selection_rows},
                    indent=2,
                    sort_keys=True,
                    default=str,
                )
                + "\n"
            )

        if candidates_info:
            payload = _build_tree_payload_from_workspace_scan(
                candidates_info=candidates_info,
                log_path=log_path,
                maximize=maximize,
                highlight_indices=highlight_indices,
            )
            _write_tree_plot_html_from_payload(log_path, payload)
        else:
            _write_placeholder_tree(log_path)

    (submission_dir / "rdagent_selection_summary.json").write_text(
        json.dumps([r.__dict__ for r in results], indent=2, sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
