import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import styles from "./MetricBarChart.module.css";
import {
  AXIS_PROPS,
  CHART_COLORS,
  GRID_PROPS,
  TOOLTIP_CONTENT_STYLE,
  TOOLTIP_ITEM_STYLE,
  TOOLTIP_LABEL_STYLE,
  TOOLTIP_WRAPPER_STYLE,
  colorForModel,
} from "./chartTheme.js";
import { formatMetricByName } from "../../utils/format.js";

/**
 * One metric, three bars (one per model). The data array is the canonical
 * row built by metricsConfig.buildModelMetricRows; this component just
 * picks the metric.key field off each row.
 *
 * Variants:
 *   - "default" — full chart with axes, grid, tooltip, value labels.
 *   - "mini"    — compact small-multiple version with no Y axis or grid.
 */
export default function MetricBarChart({
  rows,
  metric,
  height = 260,
  variant = "default",
}) {
  if (!rows || !metric) return null;

  const data = rows.map((r) => ({
    name: r.name,
    value: r[metric.key],
    color: colorForModel(r.name),
    is_best: r.is_best,
  }));

  const isMini = variant === "mini";
  const max = Math.max(...data.map((d) => Number(d.value) || 0));
  const yDomain = [0, max > 0 ? max * 1.18 : 1];

  return (
    <div className={styles.wrap}>
      <ResponsiveContainer width="100%" height={height}>
        <BarChart
          data={data}
          margin={
            isMini
              ? { top: 16, right: 8, bottom: 8, left: 8 }
              : { top: 18, right: 16, bottom: 8, left: 8 }
          }
          barCategoryGap={isMini ? "28%" : "24%"}
        >
          {!isMini && <CartesianGrid {...GRID_PROPS} />}
          <XAxis
            dataKey="name"
            {...AXIS_PROPS}
            tick={{
              ...AXIS_PROPS.tick,
              fontSize: isMini ? 10 : 11,
            }}
            interval={0}
            height={isMini ? 22 : 28}
          />
          {!isMini && (
            <YAxis
              {...AXIS_PROPS}
              domain={yDomain}
              width={48}
              tickFormatter={(v) => formatMetricByName(v, metric.key)}
            />
          )}
          <Tooltip
            cursor={{ fill: "rgba(34,211,238,0.06)" }}
            wrapperStyle={TOOLTIP_WRAPPER_STYLE}
            contentStyle={TOOLTIP_CONTENT_STYLE}
            labelStyle={TOOLTIP_LABEL_STYLE}
            itemStyle={TOOLTIP_ITEM_STYLE}
            formatter={(value) => [
              formatMetricByName(value, metric.key),
              metric.label,
            ]}
          />
          <Bar
            dataKey="value"
            radius={[3, 3, 0, 0]}
            isAnimationActive={false}
            label={
              isMini
                ? false
                : {
                    position: "top",
                    fill: CHART_COLORS.textHi,
                    fontSize: 11,
                    fontFamily:
                      '"JetBrains Mono","SF Mono","Menlo",ui-monospace,"Cascadia Code",monospace',
                    formatter: (v) => formatMetricByName(v, metric.key),
                  }
            }
          >
            {data.map((d) => (
              <Cell
                key={d.name}
                fill={d.color}
                fillOpacity={d.is_best ? 1 : 0.65}
                stroke={d.is_best ? d.color : "transparent"}
                strokeWidth={d.is_best ? 1 : 0}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
