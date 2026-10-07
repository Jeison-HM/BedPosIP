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

"""Comprehensive tests for the data augmentation module.

These tests cover the exact NumPy/MATLAB-replication functions in
``src/bedposip/data/core.py``, ``stages.py``, and ``balancer.py``.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bedposip.data.balancer import balance_to_target
from bedposip.data.core import (
    calculate_centroid,
    gen_trans,
    logic_box,
    mat_to_row,
    row_to_mat,
)
from bedposip.data.stages import (
    distribution_generation,
    matrix_samples,
    no_low_count_cats,
    sample_from_kde,
    spatial_augmentation,
)

DATASET_DIR = Path(__file__).resolve().parents[1] / "dataset"


@pytest.fixture
def rng() -> np.random.Generator:
    """Deterministic random generator for reproducible tests."""
    return np.random.default_rng(42)


class TestRowToMat:
    """Tests for row_to_mat / mat_to_row round-trip conversions."""

    def test_row_to_mat_roundtrip(self, rng: np.random.Generator) -> None:
        """mat_to_row(row_to_mat(arr)) must equal the original 64-element array."""
        original = rng.random(64)
        mat = row_to_mat(original)
        assert mat.shape == (8, 8)
        recovered = mat_to_row(mat)
        np.testing.assert_array_equal(original, recovered)


class TestCalculateCentroid:
    """Tests for calculate_centroid."""

    def test_calculate_centroid_known_matrix(self) -> None:
        """A single non-zero pixel at (row=3, col=3) 0-based yields centroid (4,4) 1-based."""
        mat = np.zeros((8, 8), dtype=float)
        mat[3, 3] = 1.0
        xc, yc = calculate_centroid(mat)
        assert xc == 4
        assert yc == 4

    def test_calculate_centroid_zero_mass_returns_nan(self) -> None:
        """An all-zeros matrix must return (NaN, NaN)."""
        mat = np.zeros((8, 8), dtype=float)
        xc, yc = calculate_centroid(mat)
        assert np.isnan(xc)
        assert np.isnan(yc)


class TestLogicBox:
    """Tests for logic_box."""

    def test_logic_box_returns_boolean_mask(self, rng: np.random.Generator) -> None:
        """logic_box must return an 8x8 boolean mask with at least one True for real data."""
        df = pd.read_csv(DATASET_DIR / "DatosEntrenamiento.csv")
        row = df.iloc[0].drop("POSICION").to_numpy(dtype=float)
        mat = row_to_mat(row)
        logic, disp = logic_box(mat)

        assert logic.shape == (8, 8)
        assert logic.dtype == bool
        assert np.any(logic), "Expected at least one True pixel in the logic mask"
        assert "x" in disp
        assert "y" in disp
        assert disp["x"].shape == (2,)
        assert disp["y"].shape == (2,)


class TestGenTrans:
    """Tests for gen_trans."""

    def test_gen_trans_count_matches_displacement(self, rng: np.random.Generator) -> None:
        """For a centered 2x2 block, the number of translations equals dx_range * dy_range."""
        mat = np.zeros((8, 8), dtype=float)
        mat[3:5, 3:5] = 1.0  # 2x2 block at centre
        logic, disp = logic_box(mat)
        translations = gen_trans(mat, logic, disp, rng=rng)

        dx_range = int(disp["x"][1] - disp["x"][0] + 1)
        dy_range = int(disp["y"][1] - disp["y"][0] + 1)
        expected_count = dx_range * dy_range

        assert len(translations) == expected_count
        for t in translations:
            assert t.shape == (8, 8)


class TestSpatialAugmentation:
    """Tests for spatial_augmentation (Stage 1)."""

    def test_spatial_augmentation_row_count(self, rng: np.random.Generator) -> None:
        """Running Stage 1 on the original dataset must yield exactly 626 rows."""
        df = pd.read_csv(DATASET_DIR / "DatosEntrenamiento.csv")
        aug = spatial_augmentation(df, rng=rng)
        assert len(aug) == 626

    def test_spatial_augmentation_class_counts(self, rng: np.random.Generator) -> None:
        """Class counts after Stage 1 must match the known reference values."""
        df = pd.read_csv(DATASET_DIR / "DatosEntrenamiento.csv")
        aug = spatial_augmentation(df, rng=rng)
        counts = aug["POSICION"].value_counts().to_dict()

        assert counts["fetus_R"] == 156
        assert counts["trunk_L"] == 122
        assert counts["trunk_R"] == 97
        assert counts["fetus_L"] == 97
        assert counts["supine"] == 80
        assert counts["prone"] == 74


class TestSampleFromKde:
    """Tests for sample_from_kde degenerate cases."""

    def test_sample_from_kde_degenerate_empty(self, rng: np.random.Generator) -> None:
        """Empty input data must return an array of zeros."""
        result = sample_from_kde(np.array([]), N=10, rng=rng)
        assert result.shape == (10,)
        np.testing.assert_array_equal(result, np.zeros(10))

    def test_sample_from_kde_degenerate_constant(self, rng: np.random.Generator) -> None:
        """Constant input data must return that same constant."""
        const = 5.0
        result = sample_from_kde(np.full(20, const), N=10, rng=rng)
        assert result.shape == (10,)
        np.testing.assert_allclose(result, np.full(10, const))


class TestMatrixSamples:
    """Tests for matrix_samples shape validation."""

    def test_matrix_samples_wrong_shape_raises(self, rng: np.random.Generator) -> None:
        """Passing a (5, 7, 7) array must raise ValueError."""
        bad_input = np.zeros((5, 7, 7))
        with pytest.raises(ValueError, match="must have shape \\(n_samples, 8, 8\\)"):
            matrix_samples(bad_input, N=5, rng=rng)


class TestNoLowCountCats:
    """Tests for no_low_count_cats filtering."""

    def test_no_low_count_cats_filters_correctly(self) -> None:
        """Categories below the threshold must be removed."""
        data = np.arange(24).reshape(6, 2, 2)
        cats = np.array(["A", "A", "B", "B", "C", "C"])
        # With thr=3 all have count 2 -> all removed.
        filtered_data, filtered_cats = no_low_count_cats(data, cats, thr=3)
        assert len(filtered_cats) == 0

        # With thr=2 all have count >=2 -> all kept.
        filtered_data, filtered_cats = no_low_count_cats(data, cats, thr=2)
        assert len(filtered_cats) == 6
        np.testing.assert_array_equal(filtered_cats, cats)

        # With thr=1 all kept.
        filtered_data, filtered_cats = no_low_count_cats(data, cats, thr=1)
        assert len(filtered_cats) == 6


class TestDistributionGeneration:
    """Tests for distribution_generation (Stage 2)."""

    def test_distribution_generation_runs(self, rng: np.random.Generator) -> None:
        """Running with tiny n_samples=5 must execute and return expected columns."""
        df = pd.read_csv(DATASET_DIR / "DatosEntrenamiento.csv")
        result = distribution_generation(df, n_samples=5, cat_thr=10, rng=rng)

        assert isinstance(result, pd.DataFrame)
        assert "POSICION" in result.columns
        # Should have at least one row per class that survived centroid filtering.
        assert len(result) > 0
        # Verify all 64 feature columns plus POSICION are present.
        assert len(result.columns) == 65


class TestBalanceToTarget:
    """Tests for balance_to_target."""

    def test_balance_to_target_undersample_success(self) -> None:
        """Undersampling all classes to 7000 must yield exact counts."""
        df = pd.read_csv(DATASET_DIR / "DatosEntrenamiento_DistProbTransEspacial.csv")
        targets = {
            "fetus_L": 7000,
            "trunk_L": 7000,
            "supine": 7000,
            "trunk_R": 7000,
            "fetus_R": 7000,
            "prone": 7000,
        }
        balanced = balance_to_target(df, targets)
        counts = balanced["POSICION"].value_counts().to_dict()

        for cls, target in targets.items():
            assert counts[cls] == target, f"Class {cls} expected {target}, got {counts[cls]}"

    def test_balance_to_target_oversample_raises(self) -> None:
        """Requesting 10000 samples for 'prone' (which has 7741) must raise ValueError."""
        df = pd.read_csv(DATASET_DIR / "DatosEntrenamiento_DistProbTransEspacial.csv")
        targets = {"prone": 10000}

        with pytest.raises(ValueError) as exc_info:
            balance_to_target(df, targets)

        msg = str(exc_info.value)
        assert "prone" in msg
        assert "7741" in msg
        assert "10000" in msg
        assert "Short by 2259" in msg


class TestNoNegativeValues:
    """Invariant: no augmented dataset may contain negative pressure values."""

    def test_reference_dataset_no_negatives(self) -> None:
        """All pressure cells in the existing final CSV must be >= 0."""
        df = pd.read_csv(DATASET_DIR / "DatosEntrenamiento_DistProbTransEspacial.csv")
        feature_cols = [c for c in df.columns if c != "POSICION"]
        min_val = df[feature_cols].min().min()
        assert min_val >= 0.0, f"Found negative value: {min_val}"
        neg_count = (df[feature_cols] < 0).sum().sum()
        assert neg_count == 0, f"Found {neg_count} negative cells"


