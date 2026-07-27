"""General-purpose utilities for model inspection, visualization, and persistence.

Helper functions to inspect trainable layer sizes, visualize training curves,
generate classification reports, and save models. Useful for both the base and
quantized models.

Example:
    >>> from bedposip.utils import (
    ...     check_layer_trainable_params, plot_training_history,
    ...     class_report_metric, save_model,
    ... )
    >>> check_layer_trainable_params(model)
    >>> plot_training_history(history, model_name='base', save=True)
    >>> class_report_metric(model, X_test, y_test, class_names)
    >>> save_model(model, output_dir='output')
"""

import os
from typing import Any, Sequence

import hls4ml
import keras
import matplotlib.pyplot as plt
import numpy as np
from keras import Model
from scipy.special import softmax
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    confusion_matrix,
    log_loss,
)


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
        if layer.__class__.__name__ in [
            "Conv2D",
            "Dense",
            "QConv2D",
            "QDense",
            "QBatchNormalization",
        ]:
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


def _compute_classification_metrics(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    from_logits: bool = True,
):
    """Run a single forward pass and compute loss, accuracy, and labels.

    Works with any object that implements ``.predict()``, including
    :class:`keras.Model` and hls4ml models.

    Args:
        model: Any object with a ``.predict()`` method.
        X_test: Test input data.
        y_test: Test labels, integer ``(N,)`` or one-hot ``(N, C)``.
        from_logits: If ``True``, apply softmax before computing log_loss.

    Returns:
        Tuple of (loss, accuracy, y_true_classes, y_pred_classes, y_pred_proba).
    """
    # From part4.1_HG_quantization and larger_jet_tagger
    X_test = np.ascontiguousarray(X_test)

    # From Scr_2_HLS4ML_Vivado_generator
    try:
        y_pred = model.predict(X_test, verbose=0)
    except TypeError:
        # hls4ml predict does not accept the verbose keyword
        y_pred = model.predict(X_test)

    # Support both integer labels (N,) and one-hot labels (N, C)
    if y_test.ndim == 1:
        y_true_classes = y_test
    else:
        y_true_classes = np.argmax(y_test, axis=1)

    y_pred_classes = np.argmax(y_pred, axis=1)

    if from_logits:
        y_pred_proba = softmax(y_pred, axis=1)
    else:
        y_pred_proba = y_pred

    loss = log_loss(y_true=y_true_classes, y_pred=y_pred_proba)
    accuracy = accuracy_score(y_true_classes, y_pred_classes)

    return loss, accuracy, y_true_classes, y_pred_classes, y_pred_proba, y_pred


def class_report_metric(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: Sequence[str],
    cmap: str = "inferno",
    save: bool = False,
    from_logits: bool = True,
    show_report: bool = True,
    show_cm: bool = True,
) -> tuple[np.ndarray, float, float]:
    """Print a classification report for the given model on test data.

    Works with any object that implements ``.predict()``, including
    :class:`keras.Model` and hls4ml models.

    Args:
        model: Compiled :class:`keras.Model` or hls4ml model to evaluate.
        X_test: Test input data.
        y_test: Test labels. Can be integer array ``(N,)`` or one-hot
            encoded ``(N, num_classes)``.
        class_names: Ordered sequence of class names corresponding to the
            label encoding used during preprocessing.
        cmap: Colormap for the confusion matrix display.
        save: If ``True``, saves the confusion matrix figure.
        from_logits: If ``True``, applies softmax before computing log_loss.
        show_report: If ``True``, prints the full classification report.
        show_cm: If ``True``, displays the confusion matrix.

    Returns:
        Tuple of (y_pred, accuracy, loss).
    """
    loss, accuracy, y_true, y_pred_classes, _, y_pred = _compute_classification_metrics(
        model, X_test, y_test, from_logits=from_logits
    )

    name = getattr(model, "name", "model")
    print(f"\nAccuracy Report for `{name}`:")
    print("=" * 70)
    print(f"  Loss: {loss:.4f}")
    print(f"  Accuracy: {accuracy * 100:.2f}%")

    if show_report:
        print(f"\nClassification Report for `{name}`:")
        print("=" * 70)
        print(
            classification_report(
                y_true,
                y_pred_classes,
                target_names=class_names,
                digits=4,
            )
        )

    if show_cm:
        cm = confusion_matrix(y_true, y_pred_classes)
        ConfusionMatrixDisplay(cm, display_labels=class_names).plot(cmap=cmap)
        if save:
            filepath = f"figures/cm_{name}.pdf"
            plt.savefig(filepath, dpi=300, bbox_inches="tight", format="pdf")
            print(f"Saved: {filepath}")
        plt.show()

    return y_pred, accuracy, loss


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


def check_quantizer_homogeneity(model: Model) -> bool:
    """Inspect quantization homogeneity for all quantizers in the model.

    Reports the homogeneity status of input (``iq``), kernel (``kq``),
    and bias (``bq``) quantizers for each quantized layer. A quantizer
    is homogeneous when its bit-width tensor has exactly one element
    (``size == 1``), meaning the entire tensor shares the same precision.

    Activation (``iq``) homogeneity is required for ``io_stream`` support
    in hls4ml. Weight and bias heterogeneity is expected and does not
    affect ``io_stream`` compatibility.

    Args:
        model: Quantized :class:`keras.Model` instance (e.g. HGQ2 model).

    Returns:
        ``True`` if all activation (``iq``) quantizers are homogeneous,
        ``False`` otherwise.

    Example:
        >>> from bedposip.utils import check_quantizer_homogeneity
        >>> ok = check_quantizer_homogeneity(model_qat)
        Quantizer Homogeneity Check
        ============================================================
        Layer: qconv_1 (QConv2D)
          iq  | shape=(1, 1, 1, 1)  size=   1 | homogeneous
          kq  | shape=(3, 3, 1, 12) size= 108 | heterogeneous
          bq  | shape=(12,)         size=  12 | heterogeneous
        Layer: output (QDense)
          iq  | shape=(1, 1)        size=   1 | homogeneous
          kq  | shape=(20, 3)       size=  60 | heterogeneous
          bq  | shape=(3,)          size=   3 | heterogeneous
        ------------------------------------------------------------
        Result: All activation (iq) quantizers are homogeneous.
                Kernel and bias heterogeneity is expected and does not
                affect io_stream compatibility.
    """
    print("\nQuantizer Homogeneity Check")
    print("=" * 60)

    iq_homogeneous = True
    for layer in model.layers:
        if not hasattr(layer, "iq") and not hasattr(layer, "kq"):
            continue

        print(f"Layer: {layer.name} ({layer.__class__.__name__})")
        has_quantizer = False
        for q_name in ("iq", "kq", "bq"):
            quantizer = getattr(layer, q_name, None)
            if quantizer is None:
                continue

            has_quantizer = True
            try:
                bits = quantizer.bits
                shape = tuple(bits.shape)
                size = int(bits.size)
                is_homogeneous = size == 1
            except Exception:
                print(f"  {q_name:3s} | (unable to inspect quantizer)")
                if q_name == "iq":
                    iq_homogeneous = False
                continue

            status = "homogeneous" if is_homogeneous else "heterogeneous"
            print(f"  {q_name:3s} | shape={shape!s:18s} size={size:4d} | {status}")

            if q_name == "iq" and not is_homogeneous:
                iq_homogeneous = False

        if not has_quantizer:
            print("  (no quantizers)")

    print("-" * 60)
    if iq_homogeneous:
        print("Result: All activation (iq) quantizers are homogeneous.")
        print("        Kernel and bias heterogeneity is expected and does not")
        print("        affect io_stream compatibility.")
    else:
        print("Result: At least one activation (iq) quantizer is heterogeneous.")
        print("        hls4ml io_stream may not be supported.")

    return iq_homogeneous


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


def inspect_hls4ml_precision(
    hls_model,
    report: bool = False,
    save: bool = False,
    model_name: str = "hls_model",
) -> dict:
    """Inspect layer precision of an hls4ml HLS model.

    Iterates over all layers of the hls4ml model and collects the
    precision strings for accumulators, weights, biases, and outputs.
    Optionally prints a report and/or saves a diagram of the model
    with precision annotations.

    Args:
        hls_model: hls4ml HLS model instance.
        report: If ``True``, prints a per-layer precision report to console.
        save: If ``True``, generates a diagram via
            :func:`hls4ml.utils.plot_model` and saves it as
            ``figures/{model_name}_precision.pdf``.
        model_name: Base name for the generated PDF file. Defaults to
            ``'hls_model'``.

    Returns:
        Dictionary mapping layer names to their precision metadata.
    """
    precision_info: dict = {}
    for layer in hls_model.get_layers():
        layer_info: dict = {
            "class_name": layer.class_name,
            "accum_t": None,
            "weights": {},
            "variables": {},
        }

        accum_t = layer.get_attr("accum_t")
        if accum_t is not None:
            layer_info["accum_t"] = str(accum_t.precision)

        for w_name, w_var in layer.weights.items():
            layer_info["weights"][w_name] = str(w_var.type.precision)

        for v_name, v_var in layer.variables.items():
            layer_info["variables"][v_name] = str(v_var.type.precision)

        precision_info[layer.name] = layer_info

    if report:
        print("\nhls4ml Layer Precision Report:")
        print("=" * 70)
        for layer_name, info in precision_info.items():
            print(f"\nLayer: {layer_name} ({info['class_name']})")
            if info["accum_t"] is not None:
                print(f"  accum_t:  {info['accum_t']}")
            for w_name, w_prec in info["weights"].items():
                print(f"  {w_name}: {w_prec}")
            for v_name, v_prec in info["variables"].items():
                print(f"  {v_name}: {v_prec}")

    if save:
        os.makedirs("figures", exist_ok=True)

        png_path = "figures/hls_model.png"
        hls4ml.utils.plot_model(
            hls_model,
            to_file=png_path,
            show_shapes=True,
            show_precision=True,
            dpi=300,
        )

        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(plt.imread(png_path))
        ax.axis("off")
        plt.tight_layout()

        pdf_path = f"figures/{model_name}_precision.pdf"
        plt.savefig(pdf_path, dpi=300, bbox_inches="tight", format="pdf")
        print(f"Saved: {pdf_path}")
        plt.show()

    return precision_info


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
