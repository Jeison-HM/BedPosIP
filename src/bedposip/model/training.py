"""Training utilities for posture classification models.

This module encapsulates training logic and callback configuration.
By centralizing these functions, both FP32 and QAT training are guaranteed
to use the same interface, facilitating reproducibility and maintenance.

Example:
    >>> from bedposip.model import get_callbacks, train_model
    >>> callbacks = get_callbacks(model_name='base', patience=10)
    >>> history = train_model(
    ...     model, X_train, y_train, X_val, y_val, callbacks,
    ...     batch_size=32, max_epochs=50
    ... )
"""

import time
from typing import List, Optional

import keras

# HGQ2 callbacks (optional, only used for quantized models)
try:
    from hgq.utils.sugar import BetaPID, FreeEBOPs
except ImportError:
    BetaPID = None
    FreeEBOPs = None


def get_callbacks(
    model_name: str = "base",
    patience: int = 10,
    track_ebops: bool = False,
    target_ebops: Optional[float] = None,
    init_beta: Optional[float] = None,
) -> List[keras.callbacks.Callback]:
    """Configure and return the list of callbacks for training.

    Included callbacks:
        - :class:`EarlyStopping`: Early stopping with best weights restoration.
        - :class:`ReduceLROnPlateau`: Adaptive learning rate reduction.
        - :class:`ModelCheckpoint`: Save best model based on ``val_loss``.
        - :class:`FreeEBOPs` (optional): EBOP tracking for FPGA resource
          estimation (QAT only).
        - :class:`BetaPID` (optional): PID controller that dynamically adjusts
          ``beta`` to steer the model toward a target EBOPs budget.

    Args:
        model_name: Base name for the checkpoint file.
            Generates ``'best_{model_name}_model.keras'``.
        patience: Patience for :class:`EarlyStopping` and
            :class:`ReduceLROnPlateau` (patience//3 for LR).
        track_ebops: If ``True``, includes the :class:`FreeEBOPs` callback
            for HGQ2 resource monitoring.
        target_ebops: Target EBOPs budget for :class:`BetaPID`. If provided,
            ``BetaPID`` is added to dynamically adjust ``beta`` during training.
        init_beta: Initial ``beta`` value for :class:`BetaPID`. If ``None``,
            the average ``beta`` of the model is used.

    Returns:
        List of callback instances ready to pass to :meth:`keras.Model.fit`.
    """
    filepath = f"best_{model_name}_model.keras"

    callbacks: List[keras.callbacks.Callback] = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(1, patience // 3),
            verbose=1,
        ),
        # keras.callbacks.ModelCheckpoint(
        #     filepath=filepath,
        #     monitor="val_loss",
        #     save_best_only=True,
        #     verbose=1,
        # ),
    ]

    if track_ebops and FreeEBOPs is not None:
        ebops = FreeEBOPs()  # EBOPs tracking for FPGA resource estimation
        callbacks.extend([ebops])

    if target_ebops is not None and BetaPID is not None:
        beta_pid = BetaPID(
            target_ebops=target_ebops,
            init_beta=init_beta,
            warmup=10,
            log=True,
            max_beta=1e-4,
            min_beta=1e-9,
            damp_beta_on_target=0.1,
        )
        callbacks.extend([beta_pid])

    print("Callbacks configured:")
    for cb in callbacks:
        print(f"  - {cb.__class__.__name__}")

    return callbacks


def train_model(
    model: keras.Model,
    X_train,
    y_train,
    X_val,
    y_val,
    callbacks: List[keras.callbacks.Callback],
    batch_size: int = 32,
    epochs: int = 50,
) -> keras.callbacks.History:
    """Train a Keras model with the provided data and callbacks.

    This function is agnostic to model type (FP32 or quantized);
    it only orchestrates the call to :meth:`keras.Model.fit`.

    Args:
        model: Compiled :class:`keras.Model` instance.
        X_train: Training data.
        y_train: Training labels (integer array or one-hot).
        X_val: Validation data.
        y_val: Validation labels (integer array or one-hot).
        callbacks: List of callbacks (see :func:`get_callbacks`).
        batch_size: Batch size. Defaults to 32.
        epochs: Maximum number of epochs. Defaults to 50.

    Returns:
        :class:`History` object with training metrics.
    """
    start_time = time.time()

    history = model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1,
    )

    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time / 60:.2f} minutes")

    return history
