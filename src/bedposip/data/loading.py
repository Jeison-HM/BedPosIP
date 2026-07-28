"""Data loading utilities for posture classification.

This module provides :func:`load_dataset` to load ``.npy`` arrays produced
by ``dataset/data_preparation.ipynb``. Labels are already integer-encoded
(``(N,)`` with values ``{0, 1, 2}``), so no further preprocessing is required.

Example:
    >>> from bedposip.data import load_dataset
    >>> X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
    >>> # Data is shaped (N, 8, 8, 1); labels are integers (N,)
"""

from pathlib import Path

import numpy as np


def load_dataset(
    dataset_dir: str | Path = "input/",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load training, validation, and test datasets.

    Files are expected as individual ``.npy`` arrays named ``X_train.npy``,
    ``y_train.npy``, ``X_val.npy``, ``y_val.npy``, ``X_test.npy``, and
    ``y_test.npy``. Data arrays are shaped ``(N, 8, 8, 1)`` and label
    arrays are integer-encoded ``(N,)`` with values ``{0, 1, 2}``. The
    default path points to ``../dataset/output`` relative to the notebook.

    Args:
        dataset_dir: Directory containing the ``.npy`` files. Can be a string
            or a :class:`pathlib.Path` object.

    Returns:
        A tuple with six arrays in the following order:
        ``(X_train, y_train, X_val, y_val, X_test, y_test)``.

    Raises:
        FileNotFoundError: If any of the expected files does not exist.
    """
    dataset_path = Path(dataset_dir)

    files = {
        "X_train": dataset_path / "X_train.npy",
        "y_train": dataset_path / "y_train.npy",
        "X_val": dataset_path / "X_val.npy",
        "y_val": dataset_path / "y_val.npy",
        "X_test": dataset_path / "X_test.npy",
        "y_test": dataset_path / "y_test.npy",
    }

    for name, file in files.items():
        if not file.exists():
            raise FileNotFoundError(
                f"Data file not found: {file}. "
                "Verify that dataset preprocessing has been executed."
            )

    X_train = np.load(files["X_train"]).astype("float32")
    y_train = np.load(files["y_train"])
    X_val = np.load(files["X_val"]).astype("float32")
    y_val = np.load(files["y_val"])
    X_test = np.load(files["X_test"]).astype("float32")
    y_test = np.load(files["y_test"])

    print("\nDataset shapes:")
    print(f"  Train: {X_train.shape}, labels: {y_train.shape}")
    print(f"  Val:   {X_val.shape}, labels: {y_val.shape}")
    print(f"  Test:  {X_test.shape}, labels: {y_test.shape}")

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    )
