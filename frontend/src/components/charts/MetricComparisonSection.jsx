import { useMemo, useState } from "react";
import styles from "./MetricComparisonSection.module.css";
import ChartContainer from "./ChartContainer.jsx";
import MetricBarChart from "./MetricBarChart.jsx";
import MetricSelector from "./MetricSelector.jsx";
import {
  EFFICIENCY_METRICS,
  METRICS,
  QUALITY_METRICS,
  buildModelMetricRows,
  getMetric,
} from "./metricsConfig.js";
import { colorForModel } from "./chartTheme.js";

/**
 * The Phase 3 metric comparison section.
 *
 * Layout (top to bottom, all panels framed in the dark console aesthetic):
 *   1. Focused chart — selectable single metric, large bar chart, default MAE.
 *   2. Prediction quality strip — small multiples for MAE / RMSE / MAPE / sMAPE.
 *   3. Efficiency strip — small multiples for training duration / inference time.
 *
 * The same MetricBarChart component powers all four chart slots, with a
 * "mini" variant for the small multiples. Model colors are stable across
 * every chart so XGBoost = cyan everywhere, RandomForest = amber, etc.
 */
export default function MetricComparisonSection({ run }) {
  const rows = useMemo(() => buildModelMetricRows(run), [run]);
  const [selectedKey, setSelectedKey] = useState("MAE");

  if (!rows.length) return null;

  const selected = getMetric(selectedKey) || METRICS[0];
  const primary = run?.eval_protocol?.primary_metric || "MAE";

  return (
    <div className={styles.section}>
      <ModelLegend rows={rows} />

      <ChartContainer
        eyebrow="Focused comparison"
        title={`Compare models on ${selected.label}`}
        subtitle={`${selected.sublabel}${
          selected.key === primary ? " · primary ranking metric" : ""
        }`}
        direction={selected.lower_is_better ? "lower_is_better" : "higher_is_better"}
      >
        <div className={styles.focusedHead}>
          <MetricSelector
            metrics={METRICS}
            value={selected.key}
            onChange={setSelectedKey}
          />
        </div>
        <MetricBarChart rows={rows} metric={selected} height={300} />
      </ChartContainer>

      <ChartContainer
        eyebrow="Prediction quality"
        title="All error metrics, side by side"
        subtitle="MAE, RMSE, MAPE, and sMAPE across the 3 models"
        direction="lower_is_better"
      >
        <div className={styles.smallMultiples}>
          {QUALITY_METRICS.map((m) => (
            <ChartContainer
              key={m.key}
              size="compact"
              title={m.label}
              subtitle={m.sublabel}
              direction="lower_is_better"
            >
              <MetricBarChart
                rows={rows}
                metric={m}
                height={150}
                variant="mini"
              />
            </ChartContainer>
          ))}
        </div>
      </ChartContainer>

      <ChartContainer
        eyebrow="Efficiency"
        title="Training cost vs inference cost"
        subtitle="Wall-clock training time and per-batch test inference time"
        direction="lower_is_better"
      >
        <div className={styles.efficiencyGrid}>
          {EFFICIENCY_METRICS.map((m) => (
            <ChartContainer
              key={m.key}
              size="compact"
              title={m.label}
              subtitle={m.sublabel}
              direction="lower_is_better"
            >
              <MetricBarChart
                rows={rows}
                metric={m}
                height={170}
                variant="mini"
              />
            </ChartContainer>
          ))}
        </div>
      </ChartContainer>
    </div>
  );
}

/**
 * Tiny inline legend so the model→color mapping is visible above the
 * charts even though the bar charts themselves don't render a legend.
 */
function ModelLegend({ rows }) {
  return (
    <div className={styles.legend}>
      <span className={styles.legendLabel}>Models</span>
      {rows.map((r) => (
        <span key={r.name} className={styles.legendItem}>
          <span
            className={styles.legendSwatch}
            style={{ background: colorForModel(r.name) }}
          />
          <span
            className={`${styles.legendName} ${
              r.is_best ? styles.legendBest : ""
            }`}
          >
            #{r.rank} {r.name}
          </span>
        </span>
      ))}
    </div>
  );
}
