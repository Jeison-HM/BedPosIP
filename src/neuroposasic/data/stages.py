"""Augmentation pipeline stages for posture classification.

This module replicates the three-stage MATLAB augmentation workflow:

* Stage 1 – ``DataAugmentation.m`` via :func:`spatial_augmentation`
* Stage 2 – ``DataDistribution.m`` via :func:`distribution_generation`
* Stage 3 – ``DataDistributionAugmentation.m`` via :func:`dist_spatial_augmentation`

All functions use an optional ``rng`` argument so that results are fully
reproducible when a seed is supplied.
"""

from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from neuroposasic.data.core import (
    _logic_box_impl,
    calculate_centroid,
    logic_box,
    mat_to_row,
    row_to_mat,
)


def spatial_augmentation(
    df: pd.DataFrame,
    posicion_col: str = "POSICION",
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """Stage 1 spatial augmentation – replicates ``DataAugmentation.m``.

    The function accepts a DataFrame whose ``POSICION`` column may be the
    first or the last column.  The returned DataFrame is guaranteed to have
    ``POSICION`` as the **last** column.

    Args:
        df: Input DataFrame with 64 feature columns and one ``POSICION``
            column.
        posicion_col: Name of the target label column.  Defaults to
            ``"POSICION"``.
        rng: Optional random generator.  Defaults to seed ``42``.

    Returns:
        Augmented DataFrame with ``POSICION`` as the last column.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    if posicion_col not in df.columns:
        raise ValueError(f"Column '{posicion_col}' not found in DataFrame")

    feature_cols = [c for c in df.columns if c != posicion_col]
    labels = df[posicion_col].values
    features = df[feature_cols].to_numpy(dtype=np.float64)

    new_rows: List[np.ndarray] = []
    new_labels: List[str] = []

    for idx in range(len(df)):
        row = features[idx]
        mat = row_to_mat(row).copy()
        logic, disp = logic_box(mat)
        from neuroposasic.data.core import gen_trans

        new_mats = gen_trans(mat, logic, disp, rng=rng)
        for new_mat in new_mats:
            new_rows.append(mat_to_row(new_mat))
            new_labels.append(labels[idx])

    out_df = pd.DataFrame(new_rows, columns=feature_cols)
    out_df[posicion_col] = new_labels
    return out_df


def dist_spatial_augmentation(
    df: pd.DataFrame,
    posicion_col: str = "POSICION",
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """Stage 3 spatial augmentation – replicates ``DataDistributionAugmentation.m``.

    Identical to :func:`spatial_augmentation` except that:

    1. Negative values in the input matrix are clipped to ``0`` before
       ``logicBox`` is called.
    2. The centroid pixel is forced to ``True`` inside ``logicBox``.

    Args:
        df: Input DataFrame (``POSICION`` may be first or last).
        posicion_col: Name of the label column.  Defaults to ``"POSICION"``.
        rng: Optional random generator.  Defaults to seed ``42``.

    Returns:
        Augmented DataFrame with ``POSICION`` as the last column.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    if posicion_col not in df.columns:
        raise ValueError(f"Column '{posicion_col}' not found in DataFrame")

    feature_cols = [c for c in df.columns if c != posicion_col]
    labels = df[posicion_col].values
    features = df[feature_cols].to_numpy(dtype=np.float64)

    new_rows: List[np.ndarray] = []
    new_labels: List[str] = []

    for idx in range(len(df)):
        row = features[idx]
        mat = row_to_mat(row).copy()
        mat[mat < 0] = 0.0
        logic, disp = _logic_box_impl(mat, ensure_centroid=True)
        from neuroposasic.data.core import gen_trans

        new_mats = gen_trans(mat, logic, disp, rng=rng)
        for new_mat in new_mats:
            new_rows.append(mat_to_row(new_mat))
            new_labels.append(labels[idx])

    out_df = pd.DataFrame(new_rows, columns=feature_cols)
    out_df[posicion_col] = new_labels
    return out_df


def sample_from_kde(data: np.ndarray, N: int, rng: np.random.Generator) -> np.ndarray:
    """Replicate MATLAB ``sampleFromKDE``.

    Fit a kernel-density estimate to 1-D data, compute the empirical CDF,
    and draw *N* samples via the inverse-CDF (probability-integral)
    transform.  The bandwidth follows MATLAB's ``ksdensity`` normal-rule
    default for Gaussian kernels.

    Args:
        data: 1-D observations.
        N: Number of samples to generate.
        rng: Random generator instance.

    Returns:
        1-D array of *N* synthetic samples.
    """
    data = np.asarray(data).ravel()
    n = len(data)
    if n == 0:
        return np.zeros(N, dtype=float)
    if n < 2 or np.allclose(data, data[0]):
        return rng.choice(data, size=N, replace=True).astype(float)
    std = np.std(data, ddof=1)
    q75, q25 = np.percentile(data, [75.0, 25.0])
    iqr = q75 - q25
    bw = 0.9 * min(std, iqr / 1.34) * n ** (-1.0 / 5.0)
    if bw <= 0 or not np.isfinite(bw):
        return rng.choice(data, size=N, replace=True).astype(float)
    factor = bw / std if std > 0 else 1.0
    kde = stats.gaussian_kde(data, bw_method=factor)
    pad = 3 * max(bw, 1e-6)
    x_vals = np.linspace(data.min() - pad, data.max() + pad, 1000)
    pdf_vals = kde(x_vals)
    cdf_vals = np.cumsum(pdf_vals)
    cdf_vals = cdf_vals / cdf_vals[-1]
    cdf_vals, unique_idx = np.unique(cdf_vals, return_index=True)
    x_vals = x_vals[unique_idx]
    u = rng.random(N)
    samples = np.interp(u, cdf_vals, x_vals, left=x_vals[0], right=x_vals[-1])
    return samples


def matrix_samples(mat_set: np.ndarray, N: int, rng: np.random.Generator) -> np.ndarray:
    """Replicate MATLAB ``matrix_samples``.

    For each cell ``(i, j)`` in the 8×8 grid, call :func:`sample_from_kde`
    on the slice ``mat_set[:, i, j]``.

    Args:
        mat_set: Array of shape ``(n_real, 8, 8)``.
        N: Number of synthetic matrices to generate.
        rng: Random generator instance.

    Returns:
        Array of shape ``(N, 8, 8)``.
    """
    if mat_set.ndim != 3 or mat_set.shape[1:] != (8, 8):
        raise ValueError(f"mat_set must have shape (n_samples, 8, 8), got {mat_set.shape}")
    samples = np.empty((N, 8, 8), dtype=float)
    for i in range(8):
        for j in range(8):
            samples[:, i, j] = sample_from_kde(mat_set[:, i, j], N, rng)
    return samples


def no_low_count_cats(
    data: np.ndarray, cats: np.ndarray, thr: int
) -> tuple[np.ndarray, np.ndarray]:
    """Remove centroid categories with fewer than ``thr`` samples.

    Replicates MATLAB ``no_low_count_cats``.

    Args:
        data: Array of shape ``(n_samples, 8, 8)``.
        cats: Array of centroid category strings, shape ``(n_samples,)``.
        thr: Minimum count threshold.

    Returns:
        ``(filtered_data, filtered_cats)``.
    """
    unique_cats, counts = np.unique(cats, return_counts=True)
    valid_cats = unique_cats[counts >= thr]
    mask = np.isin(cats, valid_cats)
    return data[mask], cats[mask]


def distribution_generation(
    df: pd.DataFrame,
    n_samples: int = 1000,
    cat_thr: int = 10,
    posicion_col: str = "POSICION",
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """Stage 2: Generate synthetic samples from per-cell KDE distributions.

    Replicates ``DataDistribution.m``.  The default ``n_samples=1000``
    yields roughly 25 000 total samples (across all centroid categories
    and classes), which is a convenient scale for quick validation.

    Args:
        df: Input DataFrame (Stage 1 output).
        n_samples: Samples to generate **per centroid category**.
            Defaults to ``1000``.
        cat_thr: Minimum observations required to retain a centroid
            category.  Defaults to ``10``.
        posicion_col: Name of the label column.  Defaults to
            ``"POSICION"``.
        rng: Optional random generator.  Defaults to seed ``42``.

    Returns:
        DataFrame with ``POSICION`` as the last column.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    feature_cols = [c for c in df.columns if c != posicion_col]
    all_rows: List[np.ndarray] = []
    all_labels: List[str] = []

    for class_name, group in df.groupby(posicion_col, sort=False):
        features = group[feature_cols].to_numpy(dtype=np.float64)
        n_real = len(features)

        # Convert to 8×8 matrices and compute centroids
        mats = np.empty((n_real, 8, 8), dtype=np.float64)
        centroids = []
        for idx in range(n_real):
            mats[idx] = row_to_mat(features[idx])
            xc, yc = calculate_centroid(mats[idx])
            centroids.append(f"{xc},{yc}")

        centroids = np.array(centroids)

        # Remove low-count categories
        mats_filtered, cents_filtered = no_low_count_cats(mats, centroids, cat_thr)

        if len(mats_filtered) == 0:
            continue

        # Generate samples per centroid category
        unique_cents = np.unique(cents_filtered)
        for cent in unique_cents:
            cat_mats = mats_filtered[cents_filtered == cent]
            new_mats = matrix_samples(cat_mats, n_samples, rng)
            for k in range(n_samples):
                all_rows.append(mat_to_row(new_mats[k]))
                all_labels.append(class_name)

    out_df = pd.DataFrame(all_rows, columns=feature_cols)
    out_df[posicion_col] = all_labels
    return out_df
