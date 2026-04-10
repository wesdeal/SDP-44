/**
 * Mock pipeline run.
 *
 * Mirrors the union of artifacts written under runs/{run_id}/artifacts/:
 *   - task_spec.json
 *   - dataset_profile.json
 *   - eval_protocol.json
 *   - selected_models.json
 *   - training_results.json
 *   - evaluation_report.json
 *   - comparison_table.json
 *   - preprocessing_plan.json
 *   - preprocessing_manifest.json
 *
 * Field names match the real artifact schemas exactly so swapping in
 * `fetch('/api/runs/:id')` later is a one-line change in api.js.
 *
 * Numbers reflect the user's current real run:
 *   - Target: OT, time series forecasting
 *   - Split: chronological (12177 / 1739 / 3480)
 *   - Primary metric: MAE
 *   - Ranking: XGBoost > RandomForest > DummyRegressor
 */

const RUN_ID = "299970a3-0f86-4c91-bc05-4e0f3995ee43";
const TIME_COL = "date";
const TARGET = "OT";
const PRIMARY_METRIC = "MAE";

// --- training history for XGBoost (RMSE per boosting round) ---
// validation_1: held-out validation set RMSE (the convergence curve we care about)
const xgbValRmse = [
  3.81, 3.42, 3.05, 2.78, 2.55, 2.36, 2.21, 2.09, 1.99, 1.91, 1.85, 1.80, 1.76,
  1.72, 1.69, 1.66, 1.63, 1.61, 1.59, 1.57, 1.55, 1.54, 1.53, 1.52, 1.51, 1.50,
  1.495, 1.49, 1.485, 1.482, 1.479, 1.476, 1.473, 1.471, 1.469, 1.468, 1.467,
  1.466, 1.465, 1.4645, 1.464, 1.4636, 1.4633, 1.463, 1.4628, 1.4627, 1.4626,
  1.4625, 1.4625, 1.4624,
];
// validation_0: training-set RMSE — always lower than validation, converges faster
const xgbTrainRmse = xgbValRmse.map((v) => +(v * 0.82).toFixed(4));

export const mockRun = {
  run_id: RUN_ID,
  status: "completed",
  created_at: "2026-04-08T14:21:03Z",
  completed_at: "2026-04-08T14:24:51Z",

  // ----- task_spec.json -----
  task_spec: {
    run_id: RUN_ID,
    task_type: "time_series_forecasting",
    target_col: TARGET,
    target_dtype: "float64",
    target_cardinality: null,
    time_col: TIME_COL,
    group_col: null,
    forecast_horizon: 24,
    is_multivariate_ts: true,
    classification_subtype: null,
    regression_subtype: null,
    task_confidence: "high",
    task_reasoning:
      "Dataset has a monotonic datetime column and a continuous numeric target. LLM analysis confirmed seasonality and a stable trend. Univariate target with multivariate exogenous features.",
    warnings: [],
  },

  // ----- dataset_profile.json -----
  dataset_profile: {
    dataset_name: "ETTh1",
    file_format: "csv",
    num_rows: 17396,
    num_columns: 8,
    columns: [
      {
        name: "date",
        dtype_pandas: "datetime64[ns]",
        inferred_type: "datetime",
        missing_count: 0,
        missing_fraction: 0,
        unique_count: 17396,
        sample_values: [
          "2016-07-01 00:00:00",
          "2016-07-01 01:00:00",
          "2016-07-01 02:00:00",
        ],
        is_temporal: true,
        is_monotonically_increasing: true,
      },
      ...["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"].map((n) => ({
        name: n,
        dtype_pandas: "float64",
        inferred_type: "numeric",
        missing_count: 0,
        missing_fraction: 0,
        unique_count: 8000 + Math.floor(Math.random() * 4000),
        sample_values: [1.23, 2.34, 3.45],
        min: -5.2,
        max: 18.4,
        mean: 6.1,
        std: 3.4,
        has_negative_values: true,
        is_temporal: false,
      })),
      {
        name: TARGET,
        dtype_pandas: "float64",
        inferred_type: "numeric",
        missing_count: 0,
        missing_fraction: 0,
        unique_count: 10421,
        sample_values: [30.5312, 27.7896, 27.7896],
        min: -4.08,
        max: 46.01,
        mean: 17.13,
        std: 9.42,
        has_negative_values: true,
        is_temporal: false,
      },
    ],
    llm_analysis: {
      dataset_description:
        "Hourly electricity transformer temperature dataset (ETTh1). Includes load and oil-temperature signals over multiple years.",
      suggested_target_variable: TARGET,
      target_confidence: "high",
      target_reasoning:
        "OT (oil temperature) is the canonical forecasting target for the ETT benchmark family.",
      known_quality_issues: [],
      has_trend: true,
      has_seasonality: true,
      is_multivariate: true,
      data_source_hint: "ETT benchmark (Zhou et al.)",
      ingestion_date: "2026-04-08",
    },
    profiling_completed_at: "2026-04-08T14:21:18Z",
  },

  // ----- eval_protocol.json -----
  eval_protocol: {
    task_type: "time_series_forecasting",
    split_strategy: "chronological",
    train_fraction: 0.7,
    val_fraction: 0.1,
    test_fraction: 0.2,
    shuffle: false,
    time_col: TIME_COL,
    group_col: null,
    stratify_on: null,
    cv: { enabled: false, n_splits: null, scheme: null },
    prediction_type: "point",
    quantiles: [],
    primary_metric: PRIMARY_METRIC,
    metrics: [
      { name: "MAE", display_name: "MAE", higher_is_better: false, applicable_tasks: ["time_series_forecasting", "tabular_regression"] },
      { name: "RMSE", display_name: "RMSE", higher_is_better: false, applicable_tasks: ["time_series_forecasting", "tabular_regression"] },
      { name: "MAPE", display_name: "MAPE (%)", higher_is_better: false, applicable_tasks: ["time_series_forecasting"] },
      { name: "sMAPE", display_name: "sMAPE (%)", higher_is_better: false, applicable_tasks: ["time_series_forecasting"] },
    ],
    metrics_for_this_run: [
      { name: "MAE", display_name: "MAE", higher_is_better: false },
      { name: "RMSE", display_name: "RMSE", higher_is_better: false },
      { name: "MAPE", display_name: "MAPE (%)", higher_is_better: false },
      { name: "sMAPE", display_name: "sMAPE (%)", higher_is_better: false },
    ],
    split_counts: { train: 12177, validation: 1739, test: 3480 },
  },

  // ----- selected_models.json -----
  selected_models: {
    task_type: "time_series_forecasting",
    selection_strategy: "tier_gated_deterministic",
    selected_models: [
      {
        name: "XGBoost",
        tier: "classical",
        rationale:
          "Strong default for tabular regression with engineered lag features; supports early stopping on validation RMSE.",
        substituted_from: null,
        substitution_reason: null,
      },
      {
        name: "RandomForest",
        tier: "classical",
        rationale:
          "Robust nonparametric regressor that works well as a comparison baseline against gradient boosting.",
        substituted_from: null,
        substitution_reason: null,
      },
      {
        name: "DummyRegressor",
        tier: "baseline",
        rationale:
          "Mean predictor establishes the floor any real model must beat.",
        substituted_from: null,
        substitution_reason: null,
      },
    ],
    models_considered: ["DummyRegressor", "LinearModel", "ARIMA", "XGBoost", "RandomForest", "LightGBM", "Chronos", "LSTM"],
    models_rejected: [
      { name: "Chronos", reason: "specialized tier already filled by candidate; deferred to next run" },
      { name: "LSTM", reason: "specialized tier substitution; chronos preferred when available" },
    ],
  },

  // ----- training_results.json -----
  training_results: {
    run_id: RUN_ID,
    models: [
      {
        name: "XGBoost",
        tier: "classical",
        status: "success",
        hyperparameters: {
          n_estimators: 500,
          max_depth: 6,
          learning_rate: 0.05,
          subsample: 0.9,
          colsample_bytree: 0.9,
          objective: "reg:squarederror",
        },
        hyperparameter_source: "default",
        best_val_score: 0.6914823416764021,
        training_duration_seconds: 0.3694,
        training_history: {
          metric: "RMSE",
          validation_0: xgbTrainRmse,
          validation_1: xgbValRmse,
        },
        model_path: `runs/${RUN_ID}/trained_models/XGBoost/model.json`,
        error: null,
      },
      {
        name: "RandomForest",
        tier: "classical",
        status: "success",
        hyperparameters: {
          n_estimators: 300,
          max_depth: null,
          min_samples_split: 2,
          n_jobs: -1,
        },
        hyperparameter_source: "default",
        best_val_score: 2.354816550378462,
        training_duration_seconds: 0.6508,
        training_history: null,
        model_path: `runs/${RUN_ID}/trained_models/RandomForest/model.pkl`,
        error: null,
      },
      {
        name: "DummyRegressor",
        tier: "baseline",
        status: "success",
        hyperparameters: { strategy: "mean" },
        hyperparameter_source: "default",
        best_val_score: 12.540675257536053,
        training_duration_seconds: 0.0024,
        training_history: null,
        model_path: `runs/${RUN_ID}/trained_models/DummyRegressor/model.pkl`,
        error: null,
      },
    ],
  },

  // ----- evaluation_report.json -----
  // MAPE/sMAPE are stored as fractions, matching the real artifact shape.
  // The UI layer (formatMetricByName) converts to percent for display.
  evaluation_report: {
    run_id: RUN_ID,
    primary_metric: PRIMARY_METRIC,
    test_split_size: 3480,
    models: [
      {
        name: "XGBoost",
        tier: "classical",
        status: "evaluated",
        metrics: {
          MAE: 0.59299601516392,
          RMSE: 0.8350010945608423,
          MAPE: 0.12195013106485181,
          sMAPE: 0.11764802826158739,
        },
        n_test_samples: 3480,
        inference_time_ms: 15.85,
        plot_path: `runs/${RUN_ID}/artifacts/plots/xgboost_pred_vs_actual.png`,
        error: null,
      },
      {
        name: "RandomForest",
        tier: "classical",
        status: "evaluated",
        metrics: {
          MAE: 1.1837800064684616,
          RMSE: 1.5797236280744236,
          MAPE: 0.33077781460594047,
          sMAPE: 0.21719123931062417,
        },
        n_test_samples: 3480,
        inference_time_ms: 15.023,
        plot_path: `runs/${RUN_ID}/artifacts/plots/randomforest_pred_vs_actual.png`,
        error: null,
      },
      {
        name: "DummyRegressor",
        tier: "baseline",
        status: "evaluated",
        metrics: {
          MAE: 8.556194920265558,
          RMSE: 9.220466521279265,
          MAPE: 1.940273661253985,
          sMAPE: 0.7714918672734278,
        },
        n_test_samples: 3480,
        inference_time_ms: 0.096,
        plot_path: `runs/${RUN_ID}/artifacts/plots/dummy_pred_vs_actual.png`,
        error: null,
      },
    ],
    evaluated_at: "2026-04-08T14:24:48Z",
  },

  // ----- comparison_table.json -----
  comparison_table: {
    run_id: RUN_ID,
    ranked_by: PRIMARY_METRIC,
    ranking: [
      {
        rank: 1,
        model_name: "XGBoost",
        tier: "classical",
        primary_metric_value: 0.59299601516392,
        is_best: true,
        all_metrics: {
          MAE: 0.59299601516392,
          RMSE: 0.8350010945608423,
          MAPE: 0.12195013106485181,
          sMAPE: 0.11764802826158739,
        },
        interpretation: {
          role_label: "Recommended",
          summary:
            "XGBoost delivers the lowest MAE of 0.593 across all three models, outperforming RandomForest by 2.0× and the baseline by 14.4×. Gradient boosting with engineered lag features is a strong match for this hourly time-series forecasting task.",
          strengths: [
            "Lowest MAE (0.593) on 3,480 test samples",
            "Validation RMSE converged from 3.81 → 1.46 over 50 boosting rounds",
            "Fast inference at 15.9 ms — suitable for near-real-time use",
          ],
          weaknesses: [],
          recommendation: "deploy",
        },
      },
      {
        rank: 2,
        model_name: "RandomForest",
        tier: "classical",
        primary_metric_value: 1.1837800064684616,
        is_best: false,
        all_metrics: {
          MAE: 1.1837800064684616,
          RMSE: 1.5797236280744236,
          MAPE: 0.33077781460594047,
          sMAPE: 0.21719123931062417,
        },
        interpretation: {
          role_label: "Strong Comparator",
          summary:
            "RandomForest performs solidly with a MAE of 1.184 — roughly 2.0× higher than XGBoost. As a nonparametric tree ensemble it provides a reliable reference point with stable, one-shot fitting and no gradient-based training risk.",
          strengths: [
            "Robust to outliers and correlated features",
            "No iterative fitting — no convergence or overfitting risk",
          ],
          weaknesses: [
            "MAE 2.0× higher than XGBoost (1.184 vs 0.593)",
            "MAPE of 33.1% vs XGBoost's 12.2% — larger proportional error",
          ],
          recommendation: "fallback",
        },
      },
      {
        rank: 3,
        model_name: "DummyRegressor",
        tier: "baseline",
        primary_metric_value: 8.556194920265558,
        is_best: false,
        all_metrics: {
          MAE: 8.556194920265558,
          RMSE: 9.220466521279265,
          MAPE: 1.940273661253985,
          sMAPE: 0.7714918672734278,
        },
        interpretation: {
          role_label: "Baseline Floor",
          summary:
            "The mean predictor establishes the performance floor that any real model must surpass. XGBoost beats it by 14.4× on MAE. Its inclusion confirms both real models add substantial predictive signal beyond naive averaging.",
          strengths: [
            "Ultrafast inference at 0.096 ms",
            "Zero training cost — establishes a concrete lower bound in under 3 ms",
          ],
          weaknesses: [
            "No predictive capability — constant mean output regardless of input",
            "sMAPE of 77.1% confirms near-baseline accuracy",
          ],
          recommendation: "floor",
        },
      },
    ],
  },

  // ----- preprocessing_plan.json -----
  preprocessing_plan: {
    run_id: RUN_ID,
    preserve_temporal_order: true,
    exclude_columns_from_features: [TIME_COL],
    plan_source: "deterministic",
    steps: [
      {
        order: 1,
        method: "imputation",
        parameters: { strategy: "forward_fill" },
        applies_to: "all_numeric",
        reason: "Time-aware imputation preserves temporal continuity.",
        skip_columns: [TIME_COL],
      },
      {
        order: 2,
        method: "lag_features",
        parameters: { lags: [1, 2, 3, 6, 12, 24] },
        applies_to: [TARGET, "HUFL", "MUFL", "LUFL"],
        reason: "Hourly forecasting benefits from short and daily-cycle lags.",
        skip_columns: [],
      },
      {
        order: 3,
        method: "rolling_stats",
        parameters: { windows: [6, 24], statistics: ["mean", "std"] },
        applies_to: [TARGET, "HUFL", "MUFL"],
        reason: "Rolling statistics expose local trend and volatility.",
        skip_columns: [],
      },
      {
        order: 4,
        method: "z_norm",
        parameters: { with_mean: true, with_std: true },
        applies_to: "features_only",
        reason: "Stabilizes magnitude across heterogeneous sensor channels.",
        skip_columns: [TARGET, TIME_COL],
      },
    ],
  },

  // ----- preprocessing_manifest.json -----
  preprocessing_manifest: {
    run_id: RUN_ID,
    steps_applied: [
      {
        order: 1,
        method: "imputation",
        parameters_used: { strategy: "forward_fill" },
        fitted_params: {},
        rows_before: 17396,
        rows_after: 17396,
        columns_affected: ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", TARGET],
      },
      {
        order: 2,
        method: "lag_features",
        parameters_used: { lags: [1, 2, 3, 6, 12, 24] },
        fitted_params: { generated_columns: 24 },
        rows_before: 17396,
        rows_after: 17372,
        columns_affected: [TARGET, "HUFL", "MUFL", "LUFL"],
      },
      {
        order: 3,
        method: "rolling_stats",
        parameters_used: { windows: [6, 24], statistics: ["mean", "std"] },
        fitted_params: { generated_columns: 12 },
        rows_before: 17372,
        rows_after: 17372,
        columns_affected: [TARGET, "HUFL", "MUFL"],
      },
      {
        order: 4,
        method: "z_norm",
        parameters_used: { with_mean: true, with_std: true },
        fitted_params: { mean: "per-column", std: "per-column" },
        rows_before: 17372,
        rows_after: 17372,
        columns_affected: ["all_features"],
      },
    ],
  },

  // ----- feature_engineering (derived from preprocessing steps) -----
  // Structured for direct rendering; mirrors what preprocessing actually produced.
  feature_engineering: {
    raw_signals: ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"],
    lag_config: {
      columns: [TARGET, "HUFL", "MUFL", "LUFL"],
      lags: [1, 2, 3, 6, 12, 24],
      total_generated: 24,
    },
    rolling_config: {
      columns: [TARGET, "HUFL", "MUFL"],
      windows: [6, 24],
      statistics: ["mean", "std"],
      total_generated: 12,
    },
    feature_groups: {
      raw: ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"],
      rolling_mean: [
        "OT_rolling_mean_6", "OT_rolling_mean_24",
        "HUFL_rolling_mean_6", "HUFL_rolling_mean_24",
        "MUFL_rolling_mean_6", "MUFL_rolling_mean_24",
      ],
      rolling_std: [
        "OT_rolling_std_6", "OT_rolling_std_24",
        "HUFL_rolling_std_6", "HUFL_rolling_std_24",
        "MUFL_rolling_std_6", "MUFL_rolling_std_24",
      ],
      lag_features: [
        "HUFL_lag_1", "HUFL_lag_2", "HUFL_lag_3", "HUFL_lag_6", "HUFL_lag_12", "HUFL_lag_24",
        "MUFL_lag_1", "MUFL_lag_2", "MUFL_lag_3", "MUFL_lag_6", "MUFL_lag_12", "MUFL_lag_24",
        "LUFL_lag_1", "LUFL_lag_2", "LUFL_lag_3", "LUFL_lag_6", "LUFL_lag_12", "LUFL_lag_24",
      ],
      target_lags: [
        "OT_lag_1", "OT_lag_2", "OT_lag_3", "OT_lag_6", "OT_lag_12", "OT_lag_24",
      ],
    },
    // 6 raw + 6 rolling_mean + 6 rolling_std + 18 lag_features + 6 target_lags = 42 total
    total_features: 42,
    rows_dropped_for_lags: 24, // max lag window; chronological head of series removed
  },

  // ----- pipeline_metadata (run-level configuration summary) -----
  pipeline_metadata: {
    run_id: RUN_ID,
    task_type: "time_series_forecasting",
    target_col: TARGET,
    time_col: TIME_COL,
    primary_metric: PRIMARY_METRIC,
    models_tested: 3,
    hyperparameter_source: "default",
    using_mock_data: true,
    evaluation_timestamp: "2026-04-08T14:24:48Z",
    pipeline_version: "0.1.0",
    dataset_name: "ETTh1",
    api_wiring_note:
      "Replace getRun() body in src/data/api.js with fetch('/api/runs/:id') when backend is ready. Response shape must match this mock object.",
  },
};

export default mockRun;
