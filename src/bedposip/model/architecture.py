"""Model architecture definitions for posture classification.

This module defines the CNN architectures in their two variants:

- :func:`create_model`: Base floating-point (FP32) model for Keras.
- :func:`create_quantized_model`: Quantized model for quantization-aware
  training (QAT) using HGQ2.

Both architectures maintain the same spatial topology to facilitate direct
comparison and subsequent HLS conversion via hls4ml.

Example:
    >>> from bedposip.model import create_model
    >>> model = create_model(input_shape=(8, 8, 1), num_classes=3)
    >>> model.summary()
"""

from typing import Optional

import keras
from keras import layers, optimizers

# HGQ2 imports (only needed for quantized model)
from hgq.constraints import MinMax
from hgq.layers import QConv2D, QDense, QBatchNormalization, QMaxPooling2D
from hgq.quantizer.config import QuantizerConfig


def create_model(
    hp: Optional[object] = None,
    input_shape: tuple[int, int, int] = (8, 8, 1),
    num_classes: int = 3,
) -> keras.Model:
    """Create the base floating-point (FP32) CNN model.

    The architecture consists of 3 convolutional blocks followed by
    :class:`MaxPooling2D`, :class:`GlobalAveragePooling2D`, and a dense
    output layer with softmax activation.

    When ``hp`` is provided (Keras Tuner object), the function operates
    in hyperparameter search mode, varying the number of filters and the
    learning rate.

    Args:
        hp: Optional ``keras_tuner.HyperParameters`` object. If ``None``,
            the fixed architecture is used.
        input_shape: Input tensor shape ``(H, W, C)``.
            Defaults to ``(8, 8, 1)`` for pressure maps.
        num_classes: Number of output classes. Defaults to 3.

    Returns:
        Compiled :class:`keras.Model` instance.
    """
    if hp is None:
        # Fixed architecture mode
        filters_list = [16, 16, 24]
        densefilters_list = [42, 64]
        kernel_size = 3
        learning_rate = 0.001
        model_name = "posture_classifier_base"
    else:
        # Hyperparameter tuning mode
        filters_list = [
            hp.Int("filters_1", min_value=8, max_value=24, step=4),
            hp.Int("filters_2", min_value=8, max_value=24, step=4),
            hp.Int("filters_3", min_value=8, max_value=32, step=4),
        ]
        densefilters_list = [
            hp.Int("densefilters_1", min_value=8, max_value=32, step=4),
            hp.Int("densefilters_2", min_value=8, max_value=32, step=4),
        ]
        kernel_size = 3
        learning_rate = hp.Choice("learning_rate", values=[0.001, 0.0001])
        model_name = "posture_classifier_tuned"

    # Input layer
    inputs = keras.Input(shape=input_shape, name="pressure_map")

    # Conv Block 1
    x = layers.Conv2D(
        filters=filters_list[0],
        kernel_size=(kernel_size, kernel_size),
        use_bias=False,
        name="conv_1",
    )(inputs)
    x = layers.BatchNormalization(name="bn_conv_1")(x)
    x = layers.Activation("relu", name="conv_act_1")(x)

    # Conv Block 2
    x = layers.Conv2D(
        filters=filters_list[1],
        kernel_size=(kernel_size, kernel_size),
        use_bias=False,
        name="conv_2",
    )(x)
    x = layers.BatchNormalization(name="bn_conv_2")(x)
    x = layers.Activation("relu", name="conv_act_2")(x)

    # Conv Block 3
    x = layers.Conv2D(
        filters=filters_list[2],
        kernel_size=(kernel_size, kernel_size),
        use_bias=False,
        name="conv_3",
    )(x)
    x = layers.BatchNormalization(name="bn_conv_3")(x)
    x = layers.Activation("relu", name="conv_act_3")(x)

    # Pooling
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = keras.layers.Flatten()(x)

    x = layers.Dense(densefilters_list[0], use_bias=False, name="dense_1")(x)
    x = layers.BatchNormalization(name="bn_dense_1")(x)
    x = layers.Activation("relu")(x)

    x = layers.Dense(densefilters_list[1], use_bias=False, name="dense_2")(x)
    x = layers.BatchNormalization(name="bn_dense_2")(x)
    x = layers.Activation("relu")(x)

    # Output layer
    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)

    # Create and compile model
    model = keras.Model(inputs=inputs, outputs=outputs, name=model_name)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def create_quantized_model(
    input_shape: tuple[int, int, int] = (8, 8, 1),
    num_classes: int = 3,
) -> keras.Model:
    """Create the quantized CNN model for QAT with HGQ2.

    The spatial topology is identical to the FP32 model, but uses quantized
    layers (:class:`QConv2D`, :class:`QDense`) instead of their Keras
    equivalents. Weight and activation quantization is handled automatically
    through HGQ2 configuration contexts (``QuantizerConfigScope``,
    ``LayerConfigScope``).

    The output layer returns **logits** (no softmax activation). The caller
    should therefore compile the model with a loss that accepts logits, e.g.
    ``keras.losses.SparseCategoricalCrossentropy(from_logits=True)``.

    Args:
        input_shape: Input tensor shape ``(H, W, C)``.
            Defaults to ``(8, 8, 1)``.
        num_classes: Number of output classes. Defaults to 3.

    Returns:
        :class:`keras.Model` instance with quantized layers.
        **Note:** The model must be compiled **inside** an active
        ``QuantizerConfigScope`` so that HGQ2 properly registers all
        quantizers.
    """
    # Fixed architecture parameters
    filters_list = [20, 20, 28]
    qfilters_list = [28, 24]
    kernel_size = 3
    model_name = "posture_classifier_qat"

    # Input layer
    inputs = keras.Input(shape=input_shape, name="pressure_map")

    # QConv Block 1
    x = QConv2D(
        filters=filters_list[0],
        kernel_size=(kernel_size, kernel_size),
        use_bias=False,
        name="qconv_1",
    )(inputs)
    x = QBatchNormalization(name="qbn_conv_1")(x)
    x = layers.Activation("relu", name="qconv_act_1")(x)

    # QConv Block 2
    x = QConv2D(
        filters=filters_list[1],
        kernel_size=(kernel_size, kernel_size),
        use_bias=False,
        name="qconv_2",
    )(x)
    x = QBatchNormalization(name="qbn_conv_2")(x)
    x = layers.Activation("relu", name="qconv_act_2")(x)

    # QConv Block 3
    x = QConv2D(
        filters=filters_list[2],
        kernel_size=(kernel_size, kernel_size),
        use_bias=False,
        name="qconv_3",
    )(x)
    x = QBatchNormalization(name="qbn_conv_3")(x)
    x = layers.Activation("relu", name="qconv_act_3")(x)

    # QPooling
    x = QMaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Flatten()(x)

    x = QDense(qfilters_list[0], use_bias=False, name="qdense_1")(x)
    x = QBatchNormalization(name="qbn_dense_1")(x)
    x = layers.Activation("relu")(x)

    x = QDense(qfilters_list[1], use_bias=False, name="qdense_2")(x)
    x = QBatchNormalization(name="qbn_dense_2")(x)
    x = layers.Activation("relu")(x)

    # Output layer
    outputs = QDense(num_classes, name="output")(x)

    # Create model (compile inside an active QuantizerConfigScope)
    model = keras.Model(inputs=inputs, outputs=outputs, name=model_name)

    return model
