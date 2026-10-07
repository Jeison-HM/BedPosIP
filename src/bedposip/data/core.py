# Copyright 2026 Jeison Hernández
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Core spatial transformation primitives for pressure map augmentation.

This module provides the low-level utilities used by all augmentation
stages.  Every function is an exact replication of its MATLAB counterpart
so that the overall pipeline remains numerically consistent with the
original NeuroPosASIC reference implementation.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np


def row_to_mat(row: np.ndarray) -> np.ndarray:
    """Reshape a 64-element row vector into an 8×8 matrix.

    This is the exact equivalent of MATLAB ``Row2Mat``.  The CSV stores
    pressure maps in row-major order (Cell_0_0 … Cell_0_7, Cell_1_0 …),
    which matches NumPy's default C-order reshape.

    Args:
        row: 1-D array of shape ``(64,)``.

    Returns:
        2-D array of shape ``(8, 8)``.
    """
    return row.reshape(8, 8, order="C")


def mat_to_row(mat: np.ndarray) -> np.ndarray:
    """Flatten an 8×8 matrix into a 64-element row vector.

    Exact equivalent of MATLAB ``Mat2Row``.

    Args:
        mat: 2-D array of shape ``(8, 8)``.

    Returns:
        1-D array of shape ``(64,)``.
    """
    return mat.flatten(order="C")


def calculate_centroid(mat: np.ndarray) -> Tuple[float, float]:
    """Compute the intensity-weighted centroid of an 8×8 matrix.

    Replication of MATLAB ``calculateCentroid``.  Coordinates are returned
    as **1-based** ``(x, y)`` pairs to stay compatible with the rest of the
    MATLAB-derived pipeline.  If the total mass is zero, ``(NaN, NaN)`` is
    returned.

    Args:
        mat: 2-D array of shape ``(8, 8)``.

    Returns:
        ``(x_centroid, y_centroid)`` as 1-based floats.
    """
    mat = mat.astype(np.float64)
    total_mass = float(np.sum(mat))

    if total_mass == 0.0:
        return np.nan, np.nan

    x_coords = np.arange(1, 9)  # columns
    y_coords = np.arange(1, 9)  # rows
    X, Y = np.meshgrid(x_coords, y_coords)

    # MATLAB round rounds .5 away from zero; for positive values this is
    # equivalent to floor(x + 0.5).
    xc = int(np.floor(np.sum(X * mat) / total_mass + 0.5))
    yc = int(np.floor(np.sum(Y * mat) / total_mass + 0.5))
    return xc, yc


def _logic_box_impl(
    mat: np.ndarray,
    *,
    ensure_centroid: bool = False,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Internal implementation of ``logicBox`` shared by both stages.

    Args:
        mat: 2-D array of shape ``(8, 8)``.
        ensure_centroid: If ``True``, force the centroid pixel to be ``True``
            after outlier removal (used by Stage 3).

    Returns:
        ``(logic_mat, displacement)`` where ``logic_mat`` is an 8×8 boolean
        array and ``displacement`` is a dict with keys ``"x"`` and ``"y"``,
        each mapping to a 2-element NumPy array ``[min, max]``.
    """
    xc, yc = calculate_centroid(mat)

    if np.isnan(xc) or np.isnan(yc):
        logic_mat = np.zeros((8, 8), dtype=bool)
        disp: Dict[str, np.ndarray] = {
            "x": np.array([0, 0]),
            "y": np.array([0, 0]),
        }
        return logic_mat, disp

    # MATLAB: mat(xc, yc)  →  row xc, column yc  →  Python: mat[xc-1, yc-1]
    threshold = 0.5 * mat[xc - 1, yc - 1]
    logic_mat = mat > threshold

    # Outlier removal using a 10×10 padded copy.
    padded_logic = np.zeros((10, 10), dtype=bool)
    padded_logic[1:9, 1:9] = logic_mat

    for i in range(8):
        for j in range(8):
            if np.count_nonzero(padded_logic[i : i + 3, j : j + 3]) == 1:
                logic_mat[i, j] = False

    if ensure_centroid:
        # Present only in DataDistributionAugmentation.m
        logic_mat[xc - 1, yc - 1] = True

    # Bounding box over the remaining True pixels.
    x_i, y_i = np.meshgrid(np.arange(1, 9), np.arange(1, 9))
    x_logic = x_i[logic_mat]
    y_logic = y_i[logic_mat]

    if x_logic.size == 0:
        # Fallback – should not happen with real pressure data.
        x_min = x_max = int(xc)
        y_min = y_max = int(yc)
    else:
        x_max = int(np.max(x_logic))
        x_min = int(np.min(x_logic))
        y_max = int(np.max(y_logic))
        y_min = int(np.min(y_logic))

    # Fill the bounding box.
    # MATLAB: logicMat(y_min:y_max, x_min:x_max) = true;
    logic_mat[y_min - 1 : y_max, x_min - 1 : x_max] = True

    disp = {
        "x": np.array([-(x_min - 1), (8 - x_max)]),
        "y": np.array([-(y_min - 1), (8 - y_max)]),
    }
    return logic_mat, disp


def logic_box(mat: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Compute the logic mask and displacement bounds for a pressure map.

    Exact replication of the ``logicBox`` function from
    ``DataAugmentation.m`` (Stage 1).

    Args:
        mat: 2-D array of shape ``(8, 8)``.

    Returns:
        ``(logic_mat, displacement)`` – see :func:`_logic_box_impl`.
    """
    return _logic_box_impl(mat, ensure_centroid=False)


def gen_trans(
    mat: np.ndarray,
    logic: np.ndarray,
    disp: Dict[str, np.ndarray],
    rng: Optional[np.random.Generator] = None,
) -> List[np.ndarray]:
    """Generate all valid spatial translations for a single pressure map.

    Exact replication of MATLAB ``genTrans``.

    Background (non-pressure) cells are filled with Gaussian noise whose
    mean and standard deviation are derived from the cells where
    ``logic == False``.  Negative noise values are clipped to ``0``.  The
    original pressure values are then copied into their shifted positions.

    Args:
        mat: Original 8×8 pressure map.
        logic: Boolean 8×8 mask from :func:`logic_box`.
        disp: Displacement dict with ``"x"`` and ``"y"`` keys.
        rng: Optional :class:`numpy.random.Generator`.  If ``None``, a
            generator seeded with ``42`` is created.

    Returns:
        List of translated 8×8 matrices.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    x_trans = np.arange(disp["x"][0], disp["x"][1] + 1)
    y_trans = np.arange(disp["y"][0], disp["y"][1] + 1)

    output: List[np.ndarray] = []

    bg_mask = ~logic
    bg_values = mat[bg_mask]

    if bg_values.size == 0:
        bg_std = 0.0
        bg_mean = 0.0
    else:
        # Population standard deviation per project mapping instructions.
        bg_std = float(np.std(bg_values, ddof=0))
        bg_mean = float(np.mean(bg_values))

    for dx in x_trans:
        for dy in y_trans:
            trans_mat = bg_std * rng.standard_normal((8, 8)) + bg_mean
            trans_mat[trans_mat < 0] = 0.0

            # Copy original values for all valid shift positions.
            # MATLAB:  for i = 1:8    (column index)
            #              for j = 1:8 (row    index)
            #                  if i+dx in [1,8] and j+dy in [1,8]
            #                      trans_mat(j+dy, i+dx) = mat(j,i)
            # In 0-based Python the same condition is 0 <= new_* < 8.
            for j in range(8):
                for i in range(8):
                    new_j = j + dy
                    new_i = i + dx
                    if 0 <= new_j < 8 and 0 <= new_i < 8:
                        trans_mat[new_j, new_i] = mat[j, i]

            output.append(trans_mat)

    return output
