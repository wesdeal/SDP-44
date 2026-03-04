"""tests/unit/test_splitter.py — Unit tests for core/splitter.py.

Covers:
- random: determinism, no overlap, all rows accounted for
- stratified: class balance preserved, custom stratify_on
- chronological: ordering preserved by time_col, no shuffling
- group_kfold: no group leakage across any pair of splits
- time_series_cv: deterministic folds, frozen test set is last segment,
  folds do not include test rows, rolling vs expanding behaviour, gap,
  n_folds validation
- validation: invalid strategy, bad fractions, missing columns
"""

import numpy as np
import pandas as pd
import pytest

from Pipeline.core.splitter import split_data


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_df():
    """100-row DataFrame with a numeric target."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "feature_a": rng.normal(size=100),
        "feature_b": rng.normal(size=100),
        "target": rng.normal(size=100),
    })


@pytest.fixture
def classified_df():
    """100-row DataFrame with a balanced binary class target (50/50)."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "feature_a": rng.normal(size=100),
        "feature_b": rng.normal(size=100),
        "label": ["A"] * 50 + ["B"] * 50,
    })


@pytest.fixture
def time_df():
    """100-row DataFrame with a monotonically increasing integer time column."""
    return pd.DataFrame({
        "time": range(100),
        "feature": range(100),
        "target": range(100),
    })


@pytest.fixture
def group_df():
    """100-row DataFrame with 10 distinct groups, each containing 10 rows."""
    groups = [f"g{i}" for i in range(10) for _ in range(10)]
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "group": groups,
        "feature": rng.normal(size=100),
        "target": rng.normal(size=100),
    })


@pytest.fixture
def ts_df():
    """200-row time series DataFrame with consecutive integer time values."""
    return pd.DataFrame({
        "time": range(200),
        "feature": range(200),
        "target": range(200),
    })


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _shape_a_keys():
    return {"X_train", "X_val", "X_test", "y_train", "y_val", "y_test"}


def _no_overlap(r):
    """Assert train/val/test index sets are pairwise disjoint."""
    idx_train = set(r["X_train"].index)
    idx_val = set(r["X_val"].index)
    idx_test = set(r["X_test"].index)
    assert idx_train.isdisjoint(idx_val), "train and val share indices"
    assert idx_train.isdisjoint(idx_test), "train and test share indices"
    assert idx_val.isdisjoint(idx_test), "val and test share indices"


def _all_rows(r, n):
    total = len(r["X_train"]) + len(r["X_val"]) + len(r["X_test"])
    assert total == n, f"Expected {n} total rows, got {total}"


# ---------------------------------------------------------------------------
# random strategy
# ---------------------------------------------------------------------------


class TestRandom:
    def test_returns_shape_a(self, simple_df):
        r = split_data(simple_df, "target", "random", 0.7, 0.15, 0.15, 42)
        assert set(r.keys()) == _shape_a_keys()

    def test_deterministic(self, simple_df):
        r1 = split_data(simple_df, "target", "random", 0.7, 0.15, 0.15, 42)
        r2 = split_data(simple_df, "target", "random", 0.7, 0.15, 0.15, 42)
        pd.testing.assert_frame_equal(r1["X_train"], r2["X_train"])
        pd.testing.assert_series_equal(r1["y_test"], r2["y_test"])

    def test_different_seeds_differ(self, simple_df):
        r1 = split_data(simple_df, "target", "random", 0.7, 0.15, 0.15, 0)
        r2 = split_data(simple_df, "target", "random", 0.7, 0.15, 0.15, 99)
        assert not r1["X_train"].index.equals(r2["X_train"].index)

    def test_no_overlap(self, simple_df):
        r = split_data(simple_df, "target", "random", 0.7, 0.15, 0.15, 42)
        _no_overlap(r)

    def test_all_rows_accounted_for(self, simple_df):
        r = split_data(simple_df, "target", "random", 0.7, 0.15, 0.15, 42)
        _all_rows(r, len(simple_df))

    def test_target_not_in_X(self, simple_df):
        r = split_data(simple_df, "target", "random", 0.7, 0.15, 0.15, 42)
        assert "target" not in r["X_train"].columns
        assert "target" not in r["X_test"].columns

    def test_approximate_test_size(self, simple_df):
        r = split_data(simple_df, "target", "random", 0.7, 0.15, 0.15, 42)
        assert abs(len(r["X_test"]) / len(simple_df) - 0.15) < 0.05


# ---------------------------------------------------------------------------
# stratified strategy
# ---------------------------------------------------------------------------


class TestStratified:
    def test_returns_shape_a(self, classified_df):
        r = split_data(classified_df, "label", "stratified", 0.7, 0.15, 0.15, 42)
        assert set(r.keys()) == _shape_a_keys()

    def test_class_balance_in_train(self, classified_df):
        r = split_data(classified_df, "label", "stratified", 0.7, 0.15, 0.15, 42)
        counts = r["y_train"].value_counts(normalize=True)
        assert abs(counts["A"] - 0.5) < 0.1

    def test_class_balance_in_test(self, classified_df):
        r = split_data(classified_df, "label", "stratified", 0.7, 0.15, 0.15, 42)
        counts = r["y_test"].value_counts(normalize=True)
        assert abs(counts["A"] - 0.5) < 0.1

    def test_deterministic(self, classified_df):
        r1 = split_data(classified_df, "label", "stratified", 0.7, 0.15, 0.15, 42)
        r2 = split_data(classified_df, "label", "stratified", 0.7, 0.15, 0.15, 42)
        pd.testing.assert_frame_equal(r1["X_train"], r2["X_train"])

    def test_custom_stratify_on(self, simple_df):
        df = simple_df.copy()
        df["strat"] = (df["feature_a"] > 0).astype(int)
        r = split_data(df, "target", "stratified", 0.7, 0.15, 0.15, 42, stratify_on="strat")
        assert set(r.keys()) == _shape_a_keys()
        _no_overlap(r)

    def test_no_overlap(self, classified_df):
        r = split_data(classified_df, "label", "stratified", 0.7, 0.15, 0.15, 42)
        _no_overlap(r)

    def test_all_rows_accounted_for(self, classified_df):
        r = split_data(classified_df, "label", "stratified", 0.7, 0.15, 0.15, 42)
        _all_rows(r, len(classified_df))


# ---------------------------------------------------------------------------
# chronological strategy
# ---------------------------------------------------------------------------


class TestChronological:
    def test_returns_shape_a(self, time_df):
        r = split_data(time_df, "target", "chronological", 0.7, 0.15, 0.15, 42, time_col="time")
        assert set(r.keys()) == _shape_a_keys()

    def test_ordering_preserved(self, time_df):
        r = split_data(time_df, "target", "chronological", 0.7, 0.15, 0.15, 42, time_col="time")
        assert r["X_train"]["time"].max() < r["X_val"]["time"].min()
        assert r["X_val"]["time"].max() < r["X_test"]["time"].min()

    def test_sort_applied_regardless_of_input_order(self, time_df):
        """Input rows in reverse order must still produce correct ordering."""
        df_rev = time_df.iloc[::-1].reset_index(drop=True)
        r = split_data(df_rev, "target", "chronological", 0.7, 0.15, 0.15, 42, time_col="time")
        assert r["X_train"]["time"].max() < r["X_val"]["time"].min()
        assert r["X_val"]["time"].max() < r["X_test"]["time"].min()

    def test_all_rows_accounted_for(self, time_df):
        r = split_data(time_df, "target", "chronological", 0.7, 0.15, 0.15, 42, time_col="time")
        _all_rows(r, len(time_df))

    def test_deterministic(self, time_df):
        r1 = split_data(time_df, "target", "chronological", 0.7, 0.15, 0.15, 42, time_col="time")
        r2 = split_data(time_df, "target", "chronological", 0.7, 0.15, 0.15, 42, time_col="time")
        pd.testing.assert_frame_equal(r1["X_train"], r2["X_train"])

    def test_requires_time_col(self, time_df):
        with pytest.raises(ValueError, match="time_col"):
            split_data(time_df, "target", "chronological", 0.7, 0.15, 0.15, 42)


# ---------------------------------------------------------------------------
# group_kfold strategy
# ---------------------------------------------------------------------------


class TestGroupKFold:
    def test_returns_shape_a(self, group_df):
        r = split_data(group_df, "target", "group_kfold", 0.7, 0.15, 0.15, 42, group_col="group")
        assert set(r.keys()) == _shape_a_keys()

    def test_no_group_leakage_train_test(self, group_df):
        r = split_data(group_df, "target", "group_kfold", 0.7, 0.15, 0.15, 42, group_col="group")
        train_groups = set(group_df.loc[r["X_train"].index, "group"])
        test_groups = set(group_df.loc[r["X_test"].index, "group"])
        assert train_groups.isdisjoint(test_groups)

    def test_no_group_leakage_val_test(self, group_df):
        r = split_data(group_df, "target", "group_kfold", 0.7, 0.15, 0.15, 42, group_col="group")
        val_groups = set(group_df.loc[r["X_val"].index, "group"])
        test_groups = set(group_df.loc[r["X_test"].index, "group"])
        assert val_groups.isdisjoint(test_groups)

    def test_no_group_leakage_train_val(self, group_df):
        r = split_data(group_df, "target", "group_kfold", 0.7, 0.15, 0.15, 42, group_col="group")
        train_groups = set(group_df.loc[r["X_train"].index, "group"])
        val_groups = set(group_df.loc[r["X_val"].index, "group"])
        assert train_groups.isdisjoint(val_groups)

    def test_deterministic(self, group_df):
        r1 = split_data(group_df, "target", "group_kfold", 0.7, 0.15, 0.15, 42, group_col="group")
        r2 = split_data(group_df, "target", "group_kfold", 0.7, 0.15, 0.15, 42, group_col="group")
        pd.testing.assert_frame_equal(r1["X_train"], r2["X_train"])

    def test_requires_group_col(self, group_df):
        with pytest.raises(ValueError, match="group_col"):
            split_data(group_df, "target", "group_kfold", 0.7, 0.15, 0.15, 42)


# ---------------------------------------------------------------------------
# time_series_cv strategy
# ---------------------------------------------------------------------------


class TestTimeSeriesCV:
    def test_returns_shape_b(self, ts_df):
        r = split_data(
            ts_df, "target", "time_series_cv", 0.6, 0.2, 0.2, 42,
            time_col="time", cv_method="rolling",
        )
        assert "folds" in r
        assert "X_test" in r
        assert "y_test" in r
        assert "meta" in r
        assert _shape_a_keys().isdisjoint(r.keys() - {"X_test", "y_test"})

    def test_meta_fields(self, ts_df):
        r = split_data(
            ts_df, "target", "time_series_cv", 0.6, 0.2, 0.2, 42,
            time_col="time", cv_method="rolling",
        )
        assert r["meta"]["cv_method"] == "rolling"
        assert isinstance(r["meta"]["n_folds"], int)
        assert r["meta"]["n_folds"] == len(r["folds"])
        assert r["meta"]["gap"] == 0

    def test_deterministic_folds(self, ts_df):
        kwargs = dict(time_col="time", cv_method="rolling")
        r1 = split_data(ts_df, "target", "time_series_cv", 0.6, 0.2, 0.2, 42, **kwargs)
        r2 = split_data(ts_df, "target", "time_series_cv", 0.6, 0.2, 0.2, 42, **kwargs)
        assert len(r1["folds"]) == len(r2["folds"])
        for f1, f2 in zip(r1["folds"], r2["folds"]):
            pd.testing.assert_frame_equal(f1["X_train"], f2["X_train"])
            pd.testing.assert_frame_equal(f1["X_val"], f2["X_val"])

    def test_frozen_test_set_is_last_by_time(self, ts_df):
        r = split_data(
            ts_df, "target", "time_series_cv", 0.6, 0.2, 0.2, 42,
            time_col="time", cv_method="rolling",
        )
        test_min_time = r["X_test"]["time"].min()
        for fold in r["folds"]:
            assert fold["X_train"]["time"].max() < test_min_time
            assert fold["X_val"]["time"].max() < test_min_time

    def test_folds_do_not_include_test_rows(self, ts_df):
        r = split_data(
            ts_df, "target", "time_series_cv", 0.6, 0.2, 0.2, 42,
            time_col="time", cv_method="rolling",
        )
        test_indices = set(r["X_test"].index)
        for fold in r["folds"]:
            assert test_indices.isdisjoint(set(fold["X_train"].index))
            assert test_indices.isdisjoint(set(fold["X_val"].index))

    def test_rolling_fixed_train_window(self, ts_df):
        initial = 60
        val_win = 20
        r = split_data(
            ts_df, "target", "time_series_cv", 0.6, 0.2, 0.2, 42,
            time_col="time", cv_method="rolling",
            initial_train_size=initial, val_window_size=val_win,
        )
        for fold in r["folds"]:
            assert len(fold["X_train"]) == initial
            assert len(fold["X_val"]) == val_win

    def test_expanding_train_grows(self, ts_df):
        initial = 40
        step = 20
        val_win = 20
        r = split_data(
            ts_df, "target", "time_series_cv", 0.6, 0.2, 0.2, 42,
            time_col="time", cv_method="expanding",
            initial_train_size=initial, step_size=step, val_window_size=val_win,
        )
        sizes = [len(f["X_train"]) for f in r["folds"]]
        assert len(sizes) >= 2, "Need at least 2 folds to verify expansion"
        for i in range(1, len(sizes)):
            assert sizes[i] > sizes[i - 1], f"Fold {i}: train did not grow ({sizes})"

    def test_n_folds_exact(self, ts_df):
        r = split_data(
            ts_df, "target", "time_series_cv", 0.6, 0.2, 0.2, 42,
            time_col="time", cv_method="rolling", n_folds=3,
            initial_train_size=60, val_window_size=20,
        )
        assert len(r["folds"]) == 3
        assert r["meta"]["n_folds"] == 3

    def test_n_folds_unachievable_raises(self, ts_df):
        with pytest.raises(ValueError, match="Cannot produce"):
            split_data(
                ts_df, "target", "time_series_cv", 0.6, 0.2, 0.2, 42,
                time_col="time", cv_method="rolling", n_folds=999,
                initial_train_size=100, val_window_size=40,
            )

    def test_gap_shifts_val_start(self, ts_df):
        """Val window must start exactly `gap` rows later when gap > 0."""
        gap = 5
        initial = 40
        val_win = 20
        r_no_gap = split_data(
            ts_df, "target", "time_series_cv", 0.6, 0.2, 0.2, 42,
            time_col="time", cv_method="rolling",
            initial_train_size=initial, val_window_size=val_win, gap=0,
        )
        r_gap = split_data(
            ts_df, "target", "time_series_cv", 0.6, 0.2, 0.2, 42,
            time_col="time", cv_method="rolling",
            initial_train_size=initial, val_window_size=val_win, gap=gap,
        )
        # time values are consecutive integers, so the positional gap maps 1:1
        diff = (
            r_gap["folds"][0]["X_val"]["time"].min()
            - r_no_gap["folds"][0]["X_val"]["time"].min()
        )
        assert diff == gap

    def test_gap_stored_in_meta(self, ts_df):
        r = split_data(
            ts_df, "target", "time_series_cv", 0.6, 0.2, 0.2, 42,
            time_col="time", cv_method="rolling", gap=3,
        )
        assert r["meta"]["gap"] == 3

    def test_folds_in_chronological_order(self, ts_df):
        r = split_data(
            ts_df, "target", "time_series_cv", 0.6, 0.2, 0.2, 42,
            time_col="time", cv_method="rolling",
        )
        if len(r["folds"]) > 1:
            for i in range(1, len(r["folds"])):
                assert (
                    r["folds"][i]["X_val"]["time"].min()
                    > r["folds"][i - 1]["X_val"]["time"].min()
                )

    def test_step_size_defaults_to_val_window_size(self, ts_df):
        """With no step_size, adjacent fold val windows must be exactly val_win apart."""
        val_win = 20
        r = split_data(
            ts_df, "target", "time_series_cv", 0.6, 0.2, 0.2, 42,
            time_col="time", cv_method="rolling",
            initial_train_size=60, val_window_size=val_win,
        )
        if len(r["folds"]) > 1:
            diff = (
                r["folds"][1]["X_val"]["time"].min()
                - r["folds"][0]["X_val"]["time"].min()
            )
            assert diff == val_win

    def test_requires_time_col(self, ts_df):
        with pytest.raises(ValueError, match="time_col"):
            split_data(ts_df, "target", "time_series_cv", 0.6, 0.2, 0.2, 42, cv_method="rolling")

    def test_invalid_cv_method_raises(self, ts_df):
        with pytest.raises(ValueError, match="cv_method"):
            split_data(
                ts_df, "target", "time_series_cv", 0.6, 0.2, 0.2, 42,
                time_col="time", cv_method="invalid_method",
            )

    def test_unsorted_input_still_freezes_last_time_as_test(self, ts_df):
        """Even if input rows are shuffled, test set must contain the largest time values."""
        df_shuffled = ts_df.sample(frac=1, random_state=0).reset_index(drop=True)
        r = split_data(
            df_shuffled, "target", "time_series_cv", 0.6, 0.2, 0.2, 42,
            time_col="time", cv_method="rolling",
        )
        # Test set should hold the top-time rows
        test_max_time = r["X_test"]["time"].max()
        assert test_max_time == ts_df["time"].max()


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestValidation:
    def test_invalid_strategy_raises(self, simple_df):
        with pytest.raises(ValueError, match="Invalid strategy"):
            split_data(simple_df, "target", "bad_strategy", 0.7, 0.15, 0.15, 42)

    def test_fractions_not_summing_to_one_raises(self, simple_df):
        with pytest.raises(ValueError, match="1.0"):
            split_data(simple_df, "target", "random", 0.7, 0.2, 0.2, 42)

    def test_zero_fraction_raises(self, simple_df):
        with pytest.raises(ValueError, match="val_fraction"):
            split_data(simple_df, "target", "random", 0.85, 0.0, 0.15, 42)

    def test_missing_target_col_raises(self, simple_df):
        with pytest.raises(ValueError, match="target_col"):
            split_data(simple_df, "nonexistent", "random", 0.7, 0.15, 0.15, 42)

    def test_chronological_missing_time_col_arg_raises(self, time_df):
        with pytest.raises(ValueError, match="time_col"):
            split_data(time_df, "target", "chronological", 0.7, 0.15, 0.15, 42)

    def test_chronological_bad_time_col_name_raises(self, time_df):
        with pytest.raises(ValueError, match="time_col"):
            split_data(
                time_df, "target", "chronological", 0.7, 0.15, 0.15, 42,
                time_col="no_such_col",
            )

    def test_group_kfold_missing_group_col_arg_raises(self, group_df):
        with pytest.raises(ValueError, match="group_col"):
            split_data(group_df, "target", "group_kfold", 0.7, 0.15, 0.15, 42)

    def test_group_kfold_bad_group_col_name_raises(self, group_df):
        with pytest.raises(ValueError, match="group_col"):
            split_data(
                group_df, "target", "group_kfold", 0.7, 0.15, 0.15, 42,
                group_col="no_such_col",
            )

    def test_stratified_bad_stratify_on_raises(self, simple_df):
        with pytest.raises(ValueError, match="stratify_on"):
            split_data(
                simple_df, "target", "stratified", 0.7, 0.15, 0.15, 42,
                stratify_on="no_such_col",
            )
