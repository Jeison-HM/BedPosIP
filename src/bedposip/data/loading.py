"""Data loading and preprocessing utilities for posture classification.

This module centralizes data loading operations from `.npz` files and their
preprocessing for CNN training. This way, both the base and quantized notebooks
can share the same logic without duplicating code.

Example:
    >>> from bedposip.data import load_dataset, preprocess_data
    >>> X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
    >>> (X_train, y_train, X_val, y_val, X_test, y_test, class_names) = preprocess_data(
    ...     X_train, y_train, X_val, y_val, X_test, y_test
    ... )
"""

from pathlib import Path
from typing import Tuple

import numpy as np
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder


def load_dataset(
    dataset_dir: str | Path = "../dataset/output",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load training, validation, and test datasets.

    Files are expected in `.npz` format with keys `data` and `labels`.
    The default path points to `../dataset/output` relative to the notebook.

    Args:
        dataset_dir: Directory containing the `.npz` files. Can be a string
            or a :class:`pathlib.Path` object.

    Returns:
        A tuple with six arrays in the following order:
        ``(X_train, y_train, X_val, y_val, X_test, y_test)``.

    Raises:
        FileNotFoundError: If any of the expected files does not exist.
    """
    dataset_path = Path(dataset_dir)
    train_file = dataset_path / "train.npz"
    val_file = dataset_path / "val.npz"
    test_file = dataset_path / "test.npz"

    for file in (train_file, val_file, test_file):
        if not file.exists():
            raise FileNotFoundError(
                f"Data file not found: {file}. "
                "Verify that dataset preprocessing has been executed."
            )

    train_data = np.load(train_file, allow_pickle=True)
    val_data = np.load(val_file, allow_pickle=True)
    test_data = np.load(test_file, allow_pickle=True)

    X_train = train_data["data"]
    y_train = train_data["labels"]
    X_val = val_data["data"]
    y_val = val_data["labels"]
    X_test = test_data["data"]
    y_test = test_data["labels"]

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


def preprocess_data(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    num_classes: int = 3,
) -> Tuple[np.ndarray, ...]:
    """Preprocess data for CNN training.

    The operations performed are:
        1. Label encoding with :class:`LabelEncoder`.
        2. One-hot conversion with :func:`keras.utils.to_categorical`.
        3. Reshape from ``(N, 64)`` to ``(N, 8, 8, 1)``.
        4. Cast to ``float32``.

    Args:
        X_train: Training data in flat format ``(N, 64)``.
        y_train: Training labels.
        X_val: Validation data in flat format ``(N, 64)``.
        y_val: Validation labels.
        X_test: Test data in flat format ``(N, 64)``.
        y_test: Test labels.
        num_classes: Number of output classes. Defaults to 3.

    Returns:
        Tuple of six preprocessed arrays plus class names:
        ``(X_train, y_train, X_val, y_val, X_test, y_test, class_names)``.
        ``class_names`` is an array of strings with the original label names
        in the order used by :class:`LabelEncoder`.
    """
    # Encode labels as integers
    label_encoder = LabelEncoder()
    y_train_int = label_encoder.fit_transform(y_train)
    y_val_int = label_encoder.transform(y_val)
    y_test_int = label_encoder.transform(y_test)

    print("Label encoding:")
    for i, label in enumerate(label_encoder.classes_):
        print(f"  {label} -> {i}")

    # One-hot encode labels
    y_train = to_categorical(y_train_int, num_classes=num_classes)
    y_val = to_categorical(y_val_int, num_classes=num_classes)
    y_test = to_categorical(y_test_int, num_classes=num_classes)

    print(f"\nOne-hot encoded shape: {y_train.shape[1]}")

    # Reshape data from (N, 64) to (N, 8, 8, 1)
    X_train = X_train.reshape(-1, 8, 8, 1).astype("float32")
    X_val = X_val.reshape(-1, 8, 8, 1).astype("float32")
    X_test = X_test.reshape(-1, 8, 8, 1).astype("float32")

    print("\nReshaped data:")
    print(f"  Train: {X_train.shape}")
    print(f"  Val:   {X_val.shape}")
    print(f"  Test:  {X_test.shape}")
    print(f"\nData range: [{X_train.min():.4f}, {X_train.max():.4f}]")
    print(f"Data dtype: {X_train.dtype}")

    return X_train, y_train, X_val, y_val, X_test, y_test, label_encoder.classes_
