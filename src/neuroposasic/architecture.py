"""Model architecture definitions for posture classification.

This module defines the CNN architectures in their two variants:

- :func:`create_model`: Base floating-point (FP32) model for Keras.
- :func:`create_quantized_model`: Quantized model for quantization-aware
  training (QAT) using HGQ2.

Both architectures maintain the same spatial topology to facilitate direct
comparison and subsequent HLS conversion via hls4ml.

Example:
    >>> from neuroposasic.architecture import create_model
    >>> model = create_model(input_shape=(8, 8, 1), num_classes=3)
    >>> model.summary()
"""

from typing import Optional

import keras
from keras import layers, optimizers, regularizers

# HGQ2 imports (only needed for quantized model)
from hgq.layers import QConv2D, QDense


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
        kernel_size = 3
        learning_rate = 0.001
        model_name = "posture_classifier_base"
    else:
        # Hyperparameter tuning mode
        filters_list = [
            hp.Int("filters_1", min_value=4, max_value=64, step=4),
            hp.Int("filters_2", min_value=4, max_value=64, step=4),
            hp.Int("filters_3", min_value=8, max_value=128, step=4),
        ]
        kernel_size = 3
        learning_rate = hp.Choice("learning_rate", values=[0.001, 0.0001])
        model_name = "posture_classifier_tuned"

    # Input layer
    inputs = keras.Input(shape=input_shape, name="pressure_map")

    # Build 3 conv blocks using loop
    x = inputs
    for i, filters in enumerate(filters_list, 1):
        x = layers.Conv2D(
            filters=filters,
            kernel_size=(kernel_size, kernel_size),
            kernel_initializer="lecun_uniform",
            kernel_regularizer=regularizers.l1(1e-4),
            use_bias=False,
            name=f"conv_{i}",
        )(x)
        x = layers.BatchNormalization(name=f"bn_conv_{i}")(x)
        x = layers.Activation("relu", name=f"conv_act_{i}")(x)

    # MaxPooling (after all conv blocks)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool_3")(x)

    # Global Average Pooling
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)

    # Output layer
    outputs = layers.Dense(
        num_classes, activation="softmax", name="output"
    )(x)

    # Create and compile model
    model = keras.Model(inputs=inputs, outputs=outputs, name=model_name)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
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

    Args:
        input_shape: Input tensor shape ``(H, W, C)``.
            Defaults to ``(8, 8, 1)``.
        num_classes: Number of output classes. Defaults to 3.

    Returns:
        :class:`keras.Model` instance with quantized layers.
        **Note:** The model must be compiled *outside* this function,
        preferably within the appropriate quantization scope.
    """
    # Input layer
    inputs = keras.Input(shape=input_shape, name="pressure_map")

    # Conv Block 1: 16 filters, 3x3 kernel
    x = QConv2D(
        filters=16,
        kernel_size=(3, 3),
        kernel_initializer="lecun_uniform",
        kernel_regularizer=regularizers.l1(1e-4),
        use_bias=False,
        name="qconv_1",
    )(inputs)
    x = keras.layers.Activation("relu", name="conv_act_1")(x)

    # Conv Block 2: 16 filters, 3x3 kernel
    x = QConv2D(
        filters=16,
        kernel_size=(3, 3),
        kernel_initializer="lecun_uniform",
        kernel_regularizer=regularizers.l1(1e-4),
        use_bias=False,
        name="qconv_2",
    )(x)
    x = keras.layers.Activation("relu", name="conv_act_2")(x)

    # Conv Block 3: 24 filters, 3x3 kernel
    x = QConv2D(
        filters=24,
        kernel_size=(3, 3),
        kernel_initializer="lecun_uniform",
        kernel_regularizer=regularizers.l1(1e-4),
        use_bias=False,
        name="qconv_3",
    )(x)
    x = keras.layers.Activation("relu", name="conv_act_3")(x)

    # Pooling
    x = keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool_3")(x)
    x = keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)

    # Output layer
    outputs = QDense(num_classes, activation="softmax", name="output")(x)

    # Create model (compile outside with HGQ2 scopes)
    model = keras.Model(
        inputs=inputs, outputs=outputs, name="posture_classifier_qat"
    )

    return model
