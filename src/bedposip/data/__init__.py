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

"""NeuroPosASIC data submodule.

Exports:
    - :func:`load_dataset`: Load `.npy` datasets.
    - :func:`row_to_mat` / :func:`mat_to_row`: Row ↔ matrix conversions.
    - :func:`calculate_centroid`: Intensity-weighted centroid.
    - :func:`logic_box`: Bounding-box / displacement logic.
    - :func:`gen_trans`: Spatial translation generator.
    - :func:`spatial_augmentation`: Stage 1 augmentation.
    - :func:`distribution_generation`: Stage 2 KDE-based generation.
    - :func:`dist_spatial_augmentation`: Stage 3 augmentation.
    - :func:`balance_to_target`: Exact-count balancing.
"""

from bedposip.data.balancer import balance_to_target
from bedposip.data.core import (
    calculate_centroid,
    gen_trans,
    logic_box,
    mat_to_row,
    row_to_mat,
)
from bedposip.data.loading import load_dataset
from bedposip.data.stages import (
    dist_spatial_augmentation,
    distribution_generation,
    spatial_augmentation,
)

__all__ = [
    "load_dataset",
    "row_to_mat",
    "mat_to_row",
    "calculate_centroid",
    "logic_box",
    "gen_trans",
    "spatial_augmentation",
    "distribution_generation",
    "dist_spatial_augmentation",
    "balance_to_target",
]
