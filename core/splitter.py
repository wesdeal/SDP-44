"""core/splitter.py — Deterministic data splitting utilities.

Supported strategies:
    random           — shuffled train/val/test via sklearn train_test_split
    stratified       — class-balanced train/val/test
    chronological    — ordered split by time_col; no shuffling
    group_kfold      — group-leakage-free split via GroupShuffleSplit
    time_series_cv   — rolling/expanding CV folds with a frozen test tail

Authority: AGENT_ARCHITECTURE.md §3 and TEAM_IMPLEMENTATION_PLAN.md §2.1.
"""

from __future__ import annotations

import math
from typing import Optional

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

_VALID_STRATEGIES = frozenset(
    {"random", "stratified", "chronological", "group_kfold", "time_series_cv"}
)
_FRACTION_EPSILON = 1e-9


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def split_data(
    df: pd.DataFrame,
    target_col: str,
    strategy: str,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    random_seed: int,
    stratify_on: Optional[str] = None,
    group_col: Optional[str] = None,
    time_col: Optional[str] = None,
    cv_method: Optional[str] = None,
    n_folds: Optional[int] = None,
    initial_train_size: Optional[int] = None,
    step_size: Optional[int] = None,
    val_window_size: Optional[int] = None,
    gap: int = 0,
) -> dict:
    """Split *df* according to *strategy* and return a split dict.

    Returns shape A for random / stratified / chronological / group_kfold::

        {
            "X_train": DataFrame, "X_val": DataFrame, "X_test": DataFrame,
            "y_train": Series,    "y_val": Series,    "y_test": Series,
        }

    Returns shape B for time_series_cv::

        {
            "folds": [{"X_train", "y_train", "X_val", "y_val"}, ...],
            "X_test": DataFrame,
            "y_test": Series,
            "meta": {"cv_method": str, "n_folds": int, "gap": int},
        }

    Args:
        df: Input DataFrame. Must contain *target_col* and any column
            referenced by *time_col*, *group_col*, or *stratify_on*.
        target_col: Name of the target column. Excluded from all X splits.
        strategy: One of "random", "stratified", "chronological",
            "group_kfold", "time_series_cv".
        train_fraction: Fraction of data for training (0 < f < 1).
        val_fraction: Fraction of data for validation (0 < f < 1).
        test_fraction: Fraction of data for test (0 < f < 1).
            The three fractions must sum to 1.0 within 1e-9.
        random_seed: RNG seed used wherever shuffling occurs.
        stratify_on: Column used for stratified splitting. Defaults to
            *target_col* when not provided. Only used by "stratified".
        group_col: Column whose values define groups. Required for
            "group_kfold".
        time_col: Column used for ordering. Required for "chronological"
            and "time_series_cv".
        cv_method: "rolling" or "expanding". Required for "time_series_cv".
        n_folds: Exact number of CV folds to return. If the data cannot
            produce this many folds, a ValueError is raised.  When None,
            all achievable folds are returned.
        initial_train_size: Length of the first training window in
            time_series_cv. Defaults to max(1, floor(|pre_test| *
            train_fraction)).
        step_size: How many rows to advance the window per fold. Defaults
            to *val_window_size*.
        val_window_size: Number of rows per validation window. Defaults to
            max(1, floor(|pre_test| * val_fraction)).
        gap: Number of rows to skip between the end of the training window
            and the start of the validation window (embargo). Default 0.

    Returns:
        Split dict (shape A or B as described above).

    Raises:
        ValueError: On invalid strategy, inconsistent fractions, or missing
            required columns.  Also raised by time_series_cv when
            *n_folds* cannot be achieved or *cv_method* is unrecognised.
    """
    _validate_strategy(strategy)
    _validate_fractions(train_fraction, val_fraction, test_fraction)
    _validate_columns(df, target_col, strategy, stratify_on, group_col, time_col)

    if strategy == "random":
        return _split_random(
            df, target_col, train_fraction, val_fraction, test_fraction, random_seed
        )
    if strategy == "stratified":
        return _split_stratified(
            df, target_col, train_fraction, val_fraction, test_fraction,
            random_seed, stratify_on,
        )
    if strategy == "chronological":
        return _split_chronological(
            df, target_col, train_fraction, val_fraction, test_fraction, time_col
        )
    if strategy == "group_kfold":
        return _split_group_kfold(
            df, target_col, train_fraction, val_fraction, test_fraction,
            random_seed, group_col,
        )
    # strategy == "time_series_cv"
    return _split_time_series_cv(
        df, target_col, train_fraction, val_fraction, test_fraction,
        time_col, cv_method, n_folds, initial_train_size, step_size,
        val_window_size, gap,
    )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_strategy(strategy: str) -> None:
    if strategy not in _VALID_STRATEGIES:
        raise ValueError(
            f"Invalid strategy {strategy!r}. "
            f"Must be one of {sorted(_VALID_STRATEGIES)}"
        )


def _validate_fractions(
    train_fraction: float, val_fraction: float, test_fraction: float
) -> None:
    for name, val in (
        ("train_fraction", train_fraction),
        ("val_fraction", val_fraction),
        ("test_fraction", test_fraction),
    ):
        if val <= 0.0:
            raise ValueError(f"{name} must be > 0, got {val}")

    total = train_fraction + val_fraction + test_fraction
    if abs(total - 1.0) > _FRACTION_EPSILON:
        raise ValueError(
            f"train_fraction + val_fraction + test_fraction must equal 1.0, "
            f"got {total:.10f}"
        )


def _validate_columns(
    df: pd.DataFrame,
    target_col: str,
    strategy: str,
    stratify_on: Optional[str],
    group_col: Optional[str],
    time_col: Optional[str],
) -> None:
    if target_col not in df.columns:
        raise ValueError(f"target_col {target_col!r} not found in DataFrame columns")

    if strategy in ("chronological", "time_series_cv"):
        if time_col is None:
            raise ValueError(
                f"strategy={strategy!r} requires time_col to be specified"
            )
        if time_col not in df.columns:
            raise ValueError(f"time_col {time_col!r} not found in DataFrame columns")

    if strategy == "group_kfold":
        if group_col is None:
            raise ValueError("strategy='group_kfold' requires group_col to be specified")
        if group_col not in df.columns:
            raise ValueError(f"group_col {group_col!r} not found in DataFrame columns")

    if strategy == "stratified" and stratify_on is not None:
        if stratify_on not in df.columns:
            raise ValueError(
                f"stratify_on {stratify_on!r} not found in DataFrame columns"
            )


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------


def _split_random(
    df: pd.DataFrame,
    target_col: str,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    random_seed: int,
) -> dict:
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_fraction, shuffle=True, random_state=random_seed
    )
    val_relative = val_fraction / (train_fraction + val_fraction)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_relative, shuffle=True, random_state=random_seed
    )
    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
    }


def _split_stratified(
    df: pd.DataFrame,
    target_col: str,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    random_seed: int,
    stratify_on: Optional[str],
) -> dict:
    X = df.drop(columns=[target_col])
    y = df[target_col]

    use_custom = stratify_on is not None and stratify_on != target_col
    stratify1 = df[stratify_on] if use_custom else y

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_fraction, stratify=stratify1, random_state=random_seed
    )

    stratify2 = df.loc[X_temp.index, stratify_on] if use_custom else y_temp
    val_relative = val_fraction / (train_fraction + val_fraction)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_relative, stratify=stratify2, random_state=random_seed
    )
    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
    }


def _split_chronological(
    df: pd.DataFrame,
    target_col: str,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    time_col: str,
) -> dict:
    df_sorted = df.sort_values(time_col, kind="stable")
    n = len(df_sorted)
    n_train = math.floor(n * train_fraction)
    n_val = math.floor(n * val_fraction)

    X = df_sorted.drop(columns=[target_col])
    y = df_sorted[target_col]

    return {
        "X_train": X.iloc[:n_train],
        "X_val": X.iloc[n_train:n_train + n_val],
        "X_test": X.iloc[n_train + n_val:],
        "y_train": y.iloc[:n_train],
        "y_val": y.iloc[n_train:n_train + n_val],
        "y_test": y.iloc[n_train + n_val:],
    }


def _split_group_kfold(
    df: pd.DataFrame,
    target_col: str,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    random_seed: int,
    group_col: str,
) -> dict:
    X = df.drop(columns=[target_col])
    y = df[target_col]
    groups = df[group_col]

    # Pass 1: separate test groups from train+val
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=random_seed)
    train_val_pos, test_pos = next(gss1.split(X, y, groups=groups))

    X_tv = X.iloc[train_val_pos]
    y_tv = y.iloc[train_val_pos]
    groups_tv = groups.iloc[train_val_pos]
    X_test = X.iloc[test_pos]
    y_test = y.iloc[test_pos]

    # Pass 2: separate val groups from train (proportionally)
    val_relative = val_fraction / (train_fraction + val_fraction)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_relative, random_state=random_seed)
    train_pos, val_pos = next(gss2.split(X_tv, y_tv, groups=groups_tv))

    return {
        "X_train": X_tv.iloc[train_pos],
        "X_val": X_tv.iloc[val_pos],
        "X_test": X_test,
        "y_train": y_tv.iloc[train_pos],
        "y_val": y_tv.iloc[val_pos],
        "y_test": y_test,
    }


def _split_time_series_cv(
    df: pd.DataFrame,
    target_col: str,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    time_col: str,
    cv_method: Optional[str],
    n_folds: Optional[int],
    initial_train_size: Optional[int],
    step_size: Optional[int],
    val_window_size: Optional[int],
    gap: int,
) -> dict:
    if cv_method not in ("rolling", "expanding"):
        raise ValueError(
            f"cv_method must be 'rolling' or 'expanding', got {cv_method!r}"
        )

    df_sorted = df.sort_values(time_col, kind="stable")
    n = len(df_sorted)

    # Frozen test set: last floor(n * test_fraction) rows (no shuffling)
    n_test = math.floor(n * test_fraction)
    if n_test < 1:
        raise ValueError(
            f"test_fraction={test_fraction} yields 0 test rows for n={n}"
        )
    n_pre = n - n_test  # length of pre-test segment

    # Apply defaults
    _initial = (
        initial_train_size
        if initial_train_size is not None
        else max(1, math.floor(n_pre * train_fraction))
    )
    _val_win = (
        val_window_size
        if val_window_size is not None
        else max(1, math.floor(n_pre * val_fraction))
    )
    _step = step_size if step_size is not None else _val_win

    # Generate all achievable folds within the pre-test segment
    folds_raw: list[tuple[int, int, int, int]] = []
    fold_k = 0
    while True:
        if cv_method == "rolling":
            tr_start = fold_k * _step
            tr_end = tr_start + _initial
        else:  # expanding
            tr_start = 0
            tr_end = _initial + fold_k * _step

        val_start = tr_end + gap
        val_end = val_start + _val_win

        if tr_end > n_pre or val_end > n_pre:
            break

        folds_raw.append((tr_start, tr_end, val_start, val_end))
        fold_k += 1

    if n_folds is not None:
        if n_folds <= 0:
            raise ValueError(f"n_folds must be a positive integer, got {n_folds}")
        if len(folds_raw) < n_folds:
            raise ValueError(
                f"Cannot produce {n_folds} CV fold(s) with the given parameters. "
                f"Only {len(folds_raw)} fold(s) are achievable "
                f"(n_pre={n_pre}, initial_train_size={_initial}, "
                f"val_window_size={_val_win}, step_size={_step}, gap={gap})."
            )
        folds_raw = folds_raw[:n_folds]

    X_all = df_sorted.drop(columns=[target_col])
    y_all = df_sorted[target_col]

    folds = [
        {
            "X_train": X_all.iloc[tr_start:tr_end],
            "y_train": y_all.iloc[tr_start:tr_end],
            "X_val": X_all.iloc[val_start:val_end],
            "y_val": y_all.iloc[val_start:val_end],
        }
        for tr_start, tr_end, val_start, val_end in folds_raw
    ]

    return {
        "folds": folds,
        "X_test": X_all.iloc[n_pre:],
        "y_test": y_all.iloc[n_pre:],
        "meta": {
            "cv_method": cv_method,
            "n_folds": len(folds),
            "gap": gap,
        },
    }
