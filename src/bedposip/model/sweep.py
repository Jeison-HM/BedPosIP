"""HGQ2 quantizer bit-width sweep utilities.

This module provides functions to sweep over HGQ2 quantizer bit-width
parameters (i0, b0, f0) for weight, bias, and datalane scopes, execute
a target notebook for each configuration, and collect Loss, Accuracy,
EBOPs, LUTs, DSPs for Pareto analysis.
"""

from __future__ import annotations

import os
import random
import re
import sys
import traceback
from pathlib import Path
from typing import Any

import nbformat
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
QAT_PARAM_RANGES: dict[str, tuple[int, int]] = {
    "weight_i0": (1, 4),
    "weight_b0": (4, 8),
    "bias_i0": (1, 4),
    "bias_b0": (3, 6),
    "datalane_i0": (1, 4),
    "datalane_f0": (1, 4),
}

DEFAULT_TRIALS = 20

_MANUAL_COMBINATIONS: list[tuple[int, int, int, int, int, int]] = [
    # (2, 6, 2, 4, 3, 3),   # default baseline
    # (1, 4, 1, 3, 1, 1),
    # (4, 8, 4, 6, 4, 4),
    # (2, 5, 2, 3, 2, 2),
    # (3, 7, 3, 5, 3, 3),
]


# ---------------------------------------------------------------------------
# Parameter generation
# ---------------------------------------------------------------------------
def generate_qat_params() -> dict[str, int]:
    """Generate a random set of quantizer parameters within ranges."""
    return {
        name: random.randint(lo, hi)
        for name, (lo, hi) in QAT_PARAM_RANGES.items()
    }


def _params_key(params: dict[str, int]) -> tuple[int, ...]:
    return tuple(params[k] for k in sorted(params))


def _build_combinations(trials: int, manual: bool = False) -> list[dict[str, int]]:
    """Build a list of parameter combinations."""
    if manual and _MANUAL_COMBINATIONS:
        return [
            {
                "weight_i0": w_i0,
                "weight_b0": w_b0,
                "bias_i0": b_i0,
                "bias_b0": b_b0,
                "datalane_i0": d_i0,
                "datalane_f0": d_f0,
            }
            for w_i0, w_b0, b_i0, b_b0, d_i0, d_f0 in _MANUAL_COMBINATIONS
        ]

    combos: list[dict[str, int]] = []
    seen: set[tuple[int, ...]] = set()
    while len(combos) < trials:
        p = generate_qat_params()
        key = _params_key(p)
        if key not in seen:
            seen.add(key)
            combos.append(p)
    return combos


# ---------------------------------------------------------------------------
# Notebook manipulation
# ---------------------------------------------------------------------------
def _make_patch_pattern(anchor: str, param_names: str) -> re.Pattern:
    """Build a compiled regex that locates the target key=value pair."""
    template = (
        f"(place='{anchor}',\n"
        f".*?(?:bc|fc)=Max\\(\\d+\\),\n"
        f".*?ic=Max\\(\\d+\\),\n"
        f".*?){param_names}"
    )
    return re.compile(template, re.DOTALL)


_WEIGHT_PATTERN = _make_patch_pattern("weight", r"i0=\d+, b0=\d+")
_BIAS_PATTERN = _make_patch_pattern("bias", r"i0=\d+, b0=\d+")
_DATALANE_PATTERN = _make_patch_pattern("datalane", r"i0=\d+, f0=\d+")


def _replace_params(source: str, params: dict[str, int]) -> str:
    source = _WEIGHT_PATTERN.sub(
        f"\\1i0={params['weight_i0']}, b0={params['weight_b0']}",
        source,
    )
    source = _BIAS_PATTERN.sub(
        f"\\1i0={params['bias_i0']}, b0={params['bias_b0']}",
        source,
    )
    source = _DATALANE_PATTERN.sub(
        f"\\1i0={params['datalane_i0']}, f0={params['datalane_f0']}",
        source,
    )
    return source


def _strip_outputs(nb: nbformat.NotebookNode) -> None:
    """Remove all cell outputs and reset execution counts."""
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        cell.outputs = []
        cell.execution_count = None


def patch_notebook(notebook_path: Path, params: dict[str, int]) -> Path:
    """Patch a notebook with new quantizer parameters and write a temp copy.

    Args:
        notebook_path: Path to the source notebook.
        params: Quantizer parameter dict.

    Returns:
        Path to the patched temporary notebook.
    """
    nb = nbformat.read(notebook_path, as_version=4)
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
    temp_path = notebook_path.with_name(notebook_path.stem + "_temp.ipynb")
    nbformat.write(nb, temp_path)
    return temp_path


# ---------------------------------------------------------------------------
# Execution & result extraction
# ---------------------------------------------------------------------------
def run_notebook(temp_path: Path, timeout: int = 7200) -> nbformat.NotebookNode:
    """Execute a temporary notebook and return the executed node.

    Args:
        temp_path: Path to the patched notebook.
        timeout: Execution timeout in seconds.

    Returns:
        Executed notebook node.
    """
    nb = nbformat.read(temp_path, as_version=4)
    client = NotebookClient(nb, timeout=timeout)
    cwd = Path.cwd()
    # The notebook expects to run from model/ (relative paths like ../dataset/output)
    os.chdir(str(temp_path.parent))
    try:
        client.execute()
    except CellExecutionError:
        traceback.print_exc()
    finally:
        os.chdir(str(cwd))
    nbformat.write(nb, temp_path)
    return nb


def parse_notebook_results(nb: nbformat.NotebookNode) -> dict[str, Any]:
    """Extract Loss, Accuracy, EBOPs, LUTs, DSPs from notebook outputs.

    Args:
        nb: Executed notebook node.

    Returns:
        Dictionary with extracted metrics.
    """
    results: dict[str, Any] = {
        "loss": None,
        "accuracy": None,
        "ebops": None,
        "luts": None,
        "dsps": None,
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
            if m:
                results["ebops"] = int(float(m.group(1)))
            m = re.search(r"LUTs:\s*([\d.]+)", text)
            if m:
                results["luts"] = int(float(m.group(1)))
            m = re.search(r"DSPs:\s*([\d.]+)", text)
            if m:
                results["dsps"] = int(float(m.group(1)))

            m = re.search(r"Loss:\s*([\d.]+)", text)
            if m:
                results["loss"] = float(m.group(1))
            m = re.search(r"Accuracy:\s*([\d.]+)%", text)
            if m:
                results["accuracy"] = float(m.group(1)) / 100.0

    if results["loss"] is None:
        results["error"] = "Could not parse Loss/Accuracy from outputs"
    if results["ebops"] is None:
        err = results.get("error") or ""
        results["error"] = (
            err + "; Could not parse EBOPs/LUTs/DSPs"
        ).strip("; ")

    return results


# ---------------------------------------------------------------------------
# Sweep orchestration
# ---------------------------------------------------------------------------
def run_sweep(
    trials: int = DEFAULT_TRIALS,
    manual: bool = False,
    notebook_path: Path | str = "model/train_quantized_cnn.ipynb",
) -> pd.DataFrame:
    """Run a QAT bit-width sweep by executing the target notebook for each config.

    Args:
        trials: Number of random combinations to try.
        manual: If True, use manual combinations instead of random.
        notebook_path: Path to the source notebook.

    Returns:
        DataFrame with one row per trial.
    """
    notebook_path = Path(notebook_path)
    combos = _build_combinations(trials, manual=manual)
    total = len(combos)
    records: list[dict[str, Any]] = []

    print(f"Total combinations: {total}")

    for idx, params in enumerate(combos, 1):
        run_id = f"run_{idx:04d}"
        print(f"[{idx}/{total}] {run_id}: {params}")

        temp_path = patch_notebook(notebook_path, params)
        try:
            nb = run_notebook(temp_path)
            results = parse_notebook_results(nb)
        except Exception:
            traceback.print_exc()
            entry = {
                "run_id": run_id,
                "loss": None,
                "accuracy": None,
                "ebops": None,
                "luts": None,
                "dsps": None,
                "error": traceback.format_exc(),
            }
            entry.update(params)
            records.append(entry)
            if temp_path.exists():
                temp_path.unlink()
            continue

        entry = {
            "run_id": run_id,
            "loss": results["loss"],
            "accuracy": results["accuracy"],
            "ebops": results["ebops"],
            "luts": results["luts"],
            "dsps": results["dsps"],
            "error": results["error"],
        }
        entry.update(params)
        records.append(entry)

        status = "OK" if results["error"] is None else "PARTIAL"
        print(
            f"  [{status}] Loss={results['loss']}, "
            f"Acc={results['accuracy']:.4f}, EBOPs={results['ebops']}, "
            f"LUTs={results['luts']}, DSPs={results['dsps']}"
        )

        if temp_path.exists():
            temp_path.unlink()

    df = pd.DataFrame(records)
    print(f"\nDone. {len(df)} results collected.")
    if df["error"].notna().any():
        print("\nRuns with errors:")
        for _, row in df[df["error"].notna()].iterrows():
            print(f"  {row['run_id']}: {row['error']}")
    return df


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def plot_pareto(
    df: pd.DataFrame,
    resource: str = "ebops",
    filename: str = "pareto",
    color_column: str = "weight_b0",
    color_label: str = "Weight bits",
) -> None:
    """Scatter plot with Pareto frontier overlay for a given resource.

    Args:
        df: DataFrame from :func:`run_sweep`.
        resource: Column to plot on the x-axis (ebops, luts, dsps).
        filename: Output PDF filename (without extension).
        color_column: Column used for scatter point coloring.
        color_label: Label for the colorbar.
    """
    x = df[resource].dropna().values
    y = df.loc[df[resource].notna(), "accuracy"].values

    # Sort by x and compute Pareto frontier (max y so far)
    idx = np.argsort(x)
    x_sorted = x[idx]
    y_sorted = y[idx]

    pareto_x, pareto_y = [], []
    max_y = -1.0
    for i in range(len(x_sorted)):
        if y_sorted[i] > max_y:
            max_y = y_sorted[i]
            pareto_x.append(x_sorted[i])
            pareto_y.append(y_sorted[i])

    # Color by selected column
    valid_df = df[df[resource].notna()]
    colors = valid_df[color_column].values

    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(
        valid_df[resource],
        valid_df["accuracy"] * 100,
        c=colors,
        cmap="viridis",
        alpha=0.7,
        edgecolors="k",
        linewidth=0.5,
    )
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label(color_label)

    # Pareto frontier
    ax.plot(
        pareto_x,
        np.array(pareto_y) * 100,
        "r--",
        linewidth=1.5,
        label="Pareto frontier",
    )
    ax.scatter(pareto_x, np.array(pareto_y) * 100, c="red", marker="s", s=40, zorder=5)

    # Labels
    ax.set_xlabel(resource.upper() + r" $\downarrow$")
    ax.set_ylabel(r"Accuracy (\%) $\uparrow$")
    ax.legend()
    ax.set_title(f"{resource.upper()} vs Accuracy")

    fig.tight_layout()
    fig.savefig(f"figures/{filename}.pdf")
    plt.show()


def print_pareto_summary(
    df: pd.DataFrame,
    resource: str = "ebops",
    threshold: float = 0.80,
) -> None:
    """Print a summary table of Pareto-optimal configurations above a threshold.

    Args:
        df: DataFrame from :func:`run_sweep`.
        resource: Resource column to evaluate Pareto frontier on.
        threshold: Minimum accuracy required (default 0.80).
    """
    valid = df[df["accuracy"].notna()]
    high = valid[valid["accuracy"] >= threshold]

    if high.empty:
        print(f"No configurations meet the {threshold:.0%} accuracy threshold.")
        return

    print(
        f"Runs with accuracy >= {threshold:.0%}: {len(high)} out of {len(valid)}\n"
    )

    # Pareto frontier computation on filtered subset
    sub = high.sort_values(resource)
    pareto = []
    max_y = -1.0
    for _, row in sub.iterrows():
        if row["accuracy"] > max_y:
            max_y = row["accuracy"]
            pareto.append(row)

    if not pareto:
        print(f"No Pareto-optimal points found for {resource}.")
        return

    print(f"Pareto-optimal configurations (accuracy >= {threshold:.0%}):\n")
    print(
        f"{'Run ID':<12} {'Acc':>6} {'EBOPs':>8} {'LUTs':>8} {'DSPs':>6}  "
        f"{'w_i0':>4} {'w_b0':>4} {'b_i0':>4} {'b_b0':>4} {'d_i0':>4} {'d_f0':>4}"
    )
    print("-" * 75)

    for row in pareto:
        print(
            f"{row['run_id']:<12} {row['accuracy']*100:>5.1f}% "
            f"{int(row['ebops']):>8} {int(row['luts']):>8} {int(row['dsps']):>6}  "
            f"{row['weight_i0']:>4} {row['weight_b0']:>4} "
            f"{row['bias_i0']:>4} {row['bias_b0']:>4} "
            f"{row['datalane_i0']:>4} {row['datalane_f0']:>4}"
        )
