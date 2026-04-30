"""Dataset balancing utility.

Provides :func:`balance_to_target` so that downstream training pipelines
can request an exact number of samples per class.  The balancer operates
as a **post-processing** step on an already-augmented dataset; it does
not generate new data itself.
"""

from typing import Dict

import numpy as np
import pandas as pd


def balance_to_target(
    df: pd.DataFrame,
    targets: Dict[str, int],
    posicion_col: str = "POSICION",
    random_state: int = 42,
) -> pd.DataFrame:
    """Balance dataset to exact counts per class.

    For classes whose available count exceeds the target, random
    **undersampling** is applied.  If a class has *fewer* samples than
    requested, a :class:`ValueError` is raised so the caller knows to
    increase ``n_samples`` in :func:`neuroposasic.data.stages.distribution_generation`
    and re-run the pipeline.

    Args:
        df: Augmented dataset DataFrame.
        targets: Mapping from class name to desired count.
            Example: ``{"trunk_L": 10000, "supine": 10000, ...}``.
        posicion_col: Name of the class label column.
        random_state: Seed for reproducible undersampling.

    Returns:
        Balanced DataFrame with exact target counts.

    Raises:
        ValueError: If any class has fewer samples than its target.
    """
    rng = np.random.default_rng(random_state)
    balanced_frames: list[pd.DataFrame] = []

    for class_name, target in targets.items():
        class_df = df[df[posicion_col] == class_name]
        n_available = len(class_df)

        if n_available < target:
            short = target - n_available
            raise ValueError(
                f"Class '{class_name}' has {n_available} samples, but target is "
                f"{target}. Short by {short}. Increase n_samples in "
                f"distribution_generation() and re-run pipeline."
            )

        if n_available > target:
            class_df = class_df.sample(n=target, random_state=rng)

        balanced_frames.append(class_df)

    return pd.concat(balanced_frames, ignore_index=True)
