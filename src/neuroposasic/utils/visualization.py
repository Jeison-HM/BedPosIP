"""General-purpose utilities for model inspection, visualization, and persistence.

Helper functions to inspect trainable layer sizes, visualize training curves,
generate classification reports, and save models. Useful for both the base and
quantized models.

Example:
    >>> from neuroposasic.utils import (
    ...     check_layer_trainable_params, plot_training_history,
    ...     class_report_metric, save_model,
    ... )
    >>> check_layer_trainable_params(model)
    >>> plot_training_history(history, model_name='base', save=True)
    >>> class_report_metric(model, X_test, y_test, class_names)
    >>> save_model(model, output_dir='output')
"""

import os
from typing import Sequence

import keras
import matplotlib.pyplot as plt
import numpy as np
from keras import Model
from sklearn.metrics import classification_report


def check_layer_trainable_params(model: Model) -> None:
    """Print the number of trainable parameters per layer.

    Iterates over all model layers and, for :class:`Conv2D` or :class:`Dense`
    layers, displays the weight count. If a layer exceeds 4096 parameters,
    a warning is emitted relevant for hls4ml synthesis.

    Args:
        model: :class:`keras.Model` instance to inspect.
    """
    # Taken from part6_cnns.ipynb (also used in Scr_1_TensorFlowCNN.py)
    for layer in model.layers:
        if layer.__class__.__name__ in ["Conv2D", "Dense"]:
            w = layer.get_weights()[0]
            layersize = np.prod(w.shape)
            print("{}: {}".format(layer.name, layersize))  # 0 = weights, 1 = biases
            if layersize > 4096:  # assuming that shape[0] is batch, i.e., 'None'
                print(
                    "Layer {} is too large ({}), are you sure you want to train?".format(
                        layer.name, layersize
                    )
                )


def plot_training_history(
    history,
    model_name: str = "model",
    save: bool = False,
) -> None:
    """Plot training loss and accuracy curves.

    Generates a figure with two subplots: loss (left) and accuracy (right),
    for both training and validation sets. If ``save=True``, exports the
    figure to PDF in the ``figures/`` folder.

    Args:
        history: :class:`History` object returned by :meth:`keras.Model.fit`.
        model_name: Base name for the generated PDF file. Defaults to
            ``'model'``.
        save: If ``True``, saves the figure as
            ``figures/{model_name}_training.pdf``.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot loss
    axes[0].plot(history.history["loss"], label="Training Loss", linewidth=2)
    axes[0].plot(history.history["val_loss"], label="Validation Loss", linewidth=2)
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].set_title("Training and Validation Loss", fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Plot accuracy
    axes[1].plot(history.history["accuracy"], label="Training Accuracy", linewidth=2)
    axes[1].plot(
        history.history["val_accuracy"],
        label="Validation Accuracy",
        linewidth=2,
    )
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Accuracy", fontsize=12)
    axes[1].set_title("Training and Validation Accuracy", fontsize=14)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        filepath = f"figures/{model_name}_training.pdf"
        plt.savefig(filepath, dpi=300, bbox_inches="tight", format="pdf")
        print(f"Saved: {filepath}")
    plt.show()


def class_report_metric(
    model: Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: Sequence[str],
) -> None:
    """Print a classification report for the given model on test data.

    Args:
        model: Compiled :class:`keras.Model` to evaluate.
        X_test: Test input data.
        y_test: Test labels in one-hot encoded format.
        class_names: Ordered sequence of class names corresponding to the
            label encoding used during preprocessing.
    """
    # Evaluate on test set for accuracy and loss
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

    print(f"\nAccuracy Report for `{model.name}`:")
    print("=" * 70)
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_accuracy * 100:.2f}%")

    # Get predictions
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred_proba, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    # Classification report
    print(f"\nClassification Report for `{model.name}`:")
    print("=" * 70)
    print(
        classification_report(
            y_true_classes,
            y_pred_classes,
            target_names=class_names,
            digits=4,
        )
    )


def inspect_quantization_bits(model: Model, report: bool = False) -> dict:
    """Extract estimated quantization bit-widths from HGQ2 quantized layers.

    Iterates over the model and collects the learned bit-widths for each
    quantizer (input, kernel, bias) in QConv2D and QDense layers.
    For the input layer, the precision is inferred from the input quantizer
    (``iq``) of the first quantized layer.

    Args:
        model: Compiled quantized :class:`keras.Model` instance.

    Returns:
        Dictionary mapping layer names to their quantization metadata.
    """
    bits_info: dict = {}
    first_quantized_layer = None

    for layer in model.layers:
        if hasattr(layer, "iq") and hasattr(layer, "kq"):
            if first_quantized_layer is None:
                first_quantized_layer = layer

            layer_info: dict = {"type": layer.__class__.__name__}
            for q_name, quantizer in (
                ("iq", getattr(layer, "iq", None)),
                ("kq", getattr(layer, "kq", None)),
                ("bq", getattr(layer, "bq", None)),
            ):
                if quantizer is not None:
                    try:
                        bits = float(keras.ops.mean(quantizer.bits))
                        fbits = float(keras.ops.mean(quantizer.fbits))
                        q_type = quantizer.q_type
                        layer_info[q_name] = {
                            "bits": bits,
                            "fbits": fbits,
                            "q_type": q_type,
                        }
                    except Exception:
                        pass
            bits_info[layer.name] = layer_info
        else:
            bits_info[layer.name] = {
                "type": layer.__class__.__name__,
                "quantizers": None,
            }

    if first_quantized_layer is not None:
        bits_info["_input_precision"] = {
            "inferred_from": first_quantized_layer.name,
            "iq_bits": bits_info[first_quantized_layer.name].get("iq", {}).get("bits"),
            "iq_fbits": bits_info[first_quantized_layer.name]
            .get("iq", {})
            .get("fbits"),
        }

    if report:
        print("\nQuantization Bit-width Report:")
        print("=" * 60)
        for layer_name, info in bits_info.items():
            if layer_name.startswith("_"):
                continue
            if info.get("quantizers") is None and "iq" not in info:
                print(f"{layer_name}: {info['type']} (no quantizers)")
                continue
            print(f"\n{layer_name} ({info['type']}):")
            for q_name in ["iq", "kq", "bq"]:
                q = info.get(q_name)
                if q:
                    print(
                        f"  {q_name}: bits={q['bits']:.2f}, fbits={q['fbits']:.2f}, type={q['q_type']}"
                    )

    return bits_info


def plot_quantization_bits(
    model: Model,
    bits_info: dict | None = None,
    save: bool = False,
) -> None:
    """Plot a bar chart of estimated quantization bit-widths per layer.

    Generates a grouped bar chart showing ``bits`` for the input quantizer
    (``iq``), kernel quantizer (``kq``), and bias quantizer (``bq``) of
    each quantized layer in the model.

    If ``bits_info`` is provided, it is used directly. Otherwise, the
    function calls :func:`inspect_quantization_bits` internally.

    Args:
        model: Compiled quantized :class:`keras.Model` instance.
        bits_info: Optional pre-computed quantization metadata from
            :func:`inspect_quantization_bits`. Defaults to ``None``.
        save: If ``True``, saves the figure as
            ``figures/{model.name}_quantization_bits.pdf``.
    """
    if bits_info is None:
        bits_info = inspect_quantization_bits(model)

    layers_data = []
    for layer_name, info in bits_info.items():
        if layer_name.startswith("_"):
            continue
        if info.get("quantizers") is None and "iq" not in info:
            continue

        layer_data: dict = {"name": layer_name}
        for q_name in ("iq", "kq", "bq"):
            q = info.get(q_name)
            layer_data[q_name] = q["bits"] if q else 0.0
        layers_data.append(layer_data)

    if not layers_data:
        print("No quantized layers found in the model.")
        return

    names = [d["name"] for d in layers_data]
    iq_bits = [d.get("iq", 0.0) for d in layers_data]
    kq_bits = [d.get("kq", 0.0) for d in layers_data]
    bq_bits = [d.get("bq", 0.0) for d in layers_data]

    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.5), 6))
    ax.bar(x - width, iq_bits, width, label="iq (activations)", color="#4472C4")
    ax.bar(x, kq_bits, width, label="kq (weights)", color="#ED7D31")
    ax.bar(x + width, bq_bits, width, label="bq (bias)", color="#70AD47")

    ax.set_ylabel("Bits", fontsize=12)
    ax.set_title(f"Learned Quantization Bit-widths: {model.name}", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    if save:
        os.makedirs("figures", exist_ok=True)
        filepath = f"figures/{model.name}_quantization_bits.pdf"
        plt.savefig(filepath, dpi=300, bbox_inches="tight", format="pdf")
        print(f"Saved: {filepath}")
    plt.show()


def save_model(
    model: Model,
    output_dir: str = "output",
) -> None:
    """Save a Keras model and verify it can be reloaded.

    The model is saved in the native Keras 3 ``.keras`` format under
    ``{output_dir}/{model.name}.keras``.

    Args:
        model: :class:`keras.Model` instance to save.
        output_dir: Directory where the model file will be written.
            Created if it does not exist. Defaults to ``"output"``.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save model in Keras 3 format
    model_path = f"{output_dir}/{model.name}.keras"
    model.save(model_path)

    print(f"\nModel saved to: {model_path}")

    # Verify saved model
    loaded_model = keras.models.load_model(model_path)
    print("\nVerification:")
    print(f"  Loaded successfully: {loaded_model is not None}")
    print(f"  Model name: {loaded_model.name}")
    print(f"  Total parameters: {loaded_model.count_params():,}")
