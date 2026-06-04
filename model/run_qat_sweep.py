#!/usr/bin/env python3
"""
Sweep over HGQ2 quantizer bit-width parameters (i0, b0, f0) for weight, bias,
and datalane scopes. Modifies train_quantized_cnn.ipynb, executes it via
nbclient, and records Loss, Accuracy, EBOPs, LUTs, DSPs into a .md table
and a .json state file for resume support.

Usage:
    python model/run_qat_sweep.py              # 20 random combinations
    python model/run_qat_sweep.py --trials 50  # N random combinations
    python model/run_qat_sweep.py --manual     # use MANUAL_COMBINATIONS below

Run from the project root directory.
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
import traceback
from pathlib import Path
from typing import Any

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NOTEBOOK_PATH = Path("model/train_quantized_cnn.ipynb")
TEMP_NOTEBOOK_PATH = Path("model/train_quantized_cnn_temp.ipynb")
STATE_PATH = Path("model/qat_sweep_state.json")
MD_PATH = Path("model/qat_sweep_results.md")

DEFAULT_TRIALS = 20

# Random ranges: (min, max) inclusive
PARAM_RANGES: dict[str, tuple[int, int]] = {
    "weight_i0": (1, 4),
    "weight_b0": (4, 8),
    "bias_i0": (1, 4),
    "bias_b0": (3, 6),
    "datalane_i0": (1, 4),
    "datalane_f0": (1, 4),
}

# ---------------------------------------------------------------------------
# Manual combinations  (uncomment and use: python run_qat_sweep.py --manual)
# Each entry is (weight_i0, weight_b0, bias_i0, bias_b0, datalane_i0, datalane_f0)
# ---------------------------------------------------------------------------
# MANUAL_COMBINATIONS: list[tuple[int, int, int, int, int, int]] = [
#     (2, 6, 2, 4, 3, 3),   # default baseline
#     (1, 4, 1, 3, 1, 1),
#     (4, 8, 4, 6, 4, 4),
#     (2, 5, 2, 3, 2, 2),
#     (3, 7, 3, 5, 3, 3),
# ]
MANUAL_COMBINATIONS: list[tuple[int, int, int, int, int, int]] = []


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------
def load_state() -> list[dict[str, Any]]:
    if STATE_PATH.exists():
        with open(STATE_PATH) as f:
            return json.load(f)
    return []


def save_state(state: list[dict[str, Any]]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def _params_key(params: dict[str, int]) -> tuple[int, ...]:
    return tuple(params[k] for k in sorted(params))


def is_duplicate(params: dict[str, int], state: list[dict[str, Any]]) -> bool:
    key = _params_key(params)
    return any(_params_key(e) == key for e in state)


# ---------------------------------------------------------------------------
# Markdown logger
# ---------------------------------------------------------------------------
MD_HEADER = (
    "| Run ID | weight_i0 | weight_b0 | bias_i0 | bias_b0 "
    "| datalane_i0 | datalane_f0 | Loss | Accuracy | EBOPs | LUTs | DSPs |\n"
    "|--------|-----------|-----------|---------|---------"
    "|-------------|-------------|------|----------|-------|------|------|\n"
)


def _fmt(val: Any, fmt: str = ".4f") -> str:
    if val is None:
        return "N/A"
    return f"{val:{fmt}}"


def _fmt_pct(val: Any) -> str:
    if val is None:
        return "N/A"
    return f"{val:.2%}"


def append_to_md(entry: dict[str, Any]) -> None:
    write_header = not MD_PATH.exists()
    with open(MD_PATH, "a") as f:
        if write_header:
            f.write(MD_HEADER)
        f.write(
            f"| {entry['run_id']} "
            f"| {entry['weight_i0']} | {entry['weight_b0']} "
            f"| {entry['bias_i0']} | {entry['bias_b0']} "
            f"| {entry['datalane_i0']} | {entry['datalane_f0']} "
            f"| {_fmt(entry['loss'])} | {_fmt_pct(entry['accuracy'])} "
            f"| {_fmt(entry['ebops'], 'd')} | {_fmt(entry['luts'], 'd')} "
            f"| {_fmt(entry['dsps'], 'd')} |\n"
        )


# ---------------------------------------------------------------------------
# Parameter generation
# ---------------------------------------------------------------------------
def generate_random_params() -> dict[str, int]:
    return {
        name: random.randint(lo, hi)
        for name, (lo, hi) in PARAM_RANGES.items()
    }


# ---------------------------------------------------------------------------
# Notebook manipulation
# ---------------------------------------------------------------------------
def _make_patch_pattern(anchor: str, param_names: str) -> re.Pattern:
    """Build a compiled regex that captures the text up to the target
    key=value pair, so we can substitute only the values."""
    return re.compile(
        rf"(place='{anchor}',\n"
        rf".*?(?:bc|fc)=Max\(\d+\),\n"
        rf".*?ic=Max\(\d+\),\n"
        rf".*?){param_names}",
        re.DOTALL,
    )


_WEIGHT_PATTERN = _make_patch_pattern("weight", r"i0=\d+, b0=\d+")
_BIAS_PATTERN = _make_patch_pattern("bias", r"i0=\d+, b0=\d+")
_DATALANE_PATTERN = _make_patch_pattern("datalane", r"i0=\d+, f0=\d+")


def _replace_params(source: str, params: dict[str, int]) -> str:
    source = _WEIGHT_PATTERN.sub(
        rf"\1i0={params['weight_i0']}, b0={params['weight_b0']}", source
    )
    source = _BIAS_PATTERN.sub(
        rf"\1i0={params['bias_i0']}, b0={params['bias_b0']}", source
    )
    source = _DATALANE_PATTERN.sub(
        rf"\1i0={params['datalane_i0']}, f0={params['datalane_f0']}", source
    )
    return source


def _strip_outputs(nb: nbformat.NotebookNode) -> None:
    """Remove all cell outputs and reset execution counts."""
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        cell.outputs = []
        cell.execution_count = None


def modify_notebook(params: dict[str, int]) -> None:
    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)

    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        src = cell.source
        if isinstance(src, list):
            src = "".join(src)
        if "place='weight'" not in src:
            continue
        cell.source = _replace_params(src, params)
        break

    _strip_outputs(nb)
    nbformat.write(nb, TEMP_NOTEBOOK_PATH)


# ---------------------------------------------------------------------------
# Execution & result extraction
# ---------------------------------------------------------------------------
def execute_notebook() -> nbformat.NotebookNode:
    nb = nbformat.read(TEMP_NOTEBOOK_PATH, as_version=4)
    client = NotebookClient(nb, timeout=7200)
    cwd = Path.cwd()
    # Notebook expects to run from model/ (relative paths like ../dataset/output)
    os.chdir(str(NOTEBOOK_PATH.parent))
    try:
        client.execute()
    except CellExecutionError:
        traceback.print_exc()
    finally:
        os.chdir(str(cwd))
    nbformat.write(nb, TEMP_NOTEBOOK_PATH)
    return nb


def parse_results(nb: nbformat.NotebookNode) -> dict[str, Any]:
    results: dict[str, Any] = {
        "loss": None, "accuracy": None,
        "ebops": None, "luts": None, "dsps": None,
        "error": None,
    }

    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        for out in cell.get("outputs", []):
            if out.output_type != "stream" or out.name != "stdout":
                continue
            text = "".join(out.text)

            m = re.search(r"EBOPs:\s*([\d.]+)", text)
            if m: results["ebops"] = int(float(m.group(1)))
            m = re.search(r"LUTs:\s*([\d.]+)", text)
            if m: results["luts"] = int(float(m.group(1)))
            m = re.search(r"DSPs:\s*([\d.]+)", text)
            if m: results["dsps"] = int(float(m.group(1)))

            m = re.search(r"Loss:\s*([\d.]+)", text)
            if m: results["loss"] = float(m.group(1))
            m = re.search(r"Accuracy:\s*([\d.]+)%", text)
            if m: results["accuracy"] = float(m.group(1)) / 100.0

    if results["loss"] is None:
        results["error"] = "Could not parse Loss/Accuracy from outputs"
    if results["ebops"] is None:
        err = results.get("error") or ""
        results["error"] = (err + "; Could not parse EBOPs/LUTs/DSPs").strip("; ")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    argv = [a.lower() for a in sys.argv[1:]]
    use_manual = "--manual" in argv
    trials = DEFAULT_TRIALS
    for i, a in enumerate(argv):
        if a == "--trials" and i + 1 < len(argv):
            trials = int(argv[i + 1])

    state = load_state()
    seen_params: set[tuple[int, ...]] = {_params_key(e) for e in state}

    # Build combination list
    if use_manual and MANUAL_COMBINATIONS:
        combinations = [
            {
                "weight_i0": w_i0, "weight_b0": w_b0,
                "bias_i0": b_i0, "bias_b0": b_b0,
                "datalane_i0": d_i0, "datalane_f0": d_f0,
            }
            for w_i0, w_b0, b_i0, b_b0, d_i0, d_f0 in MANUAL_COMBINATIONS
        ]
    else:
        combinations = []
        while len(combinations) < trials:
            p = generate_random_params()
            key = _params_key(p)
            if key not in seen_params:
                seen_params.add(key)
                combinations.append(p)

    total = len(combinations)
    already = sum(1 for p in combinations if is_duplicate(p, state))
    print(f"Total: {total}, already completed: {already}")

    for idx, params in enumerate(combinations, 1):
        if is_duplicate(params, state):
            print(f"[{idx}/{total}] Skipping (already done): {params}")
            continue

        run_id = f"run_{len(state) + 1:04d}"
        print(f"[{idx}/{total}] {run_id}: {params}")

        modify_notebook(params)
        try:
            nb = execute_notebook()
            results = parse_results(nb)
        except Exception:
            traceback.print_exc()
            entry: dict[str, Any] = {
                "run_id": run_id, **params,
                "loss": None, "accuracy": None,
                "ebops": None, "luts": None, "dsps": None,
                "error": traceback.format_exc(),
            }
            state.append(entry)
            save_state(state)
            continue

        entry = {"run_id": run_id, **params, **results}
        state.append(entry)
        save_state(state)
        append_to_md(entry)

        status = "OK" if results["error"] is None else "PARTIAL"
        print(
            f"  [{status}] Loss={results['loss']}, Acc={results['accuracy']}, "
            f"EBOPs={results['ebops']}, LUTs={results['luts']}, DSPs={results['dsps']}"
        )

    # Cleanup
    if TEMP_NOTEBOOK_PATH.exists():
        TEMP_NOTEBOOK_PATH.unlink()

    print(f"\nDone. {len(state)} results saved to {STATE_PATH} and {MD_PATH}")

    if any(e.get("error") for e in state):
        print("\nRuns with errors:")
        for e in state:
            if e.get("error"):
                print(f"  {e['run_id']}: {e['error']}")


if __name__ == "__main__":
    main()
