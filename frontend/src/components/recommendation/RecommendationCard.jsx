import styles from "./RecommendationCard.module.css";
import TierBadge from "../primitives/TierBadge.jsx";
import { formatMetric, formatPctImprovement } from "../../utils/format.js";
import { deriveWinner } from "../../services/runService.js";

export default function RecommendationCard({ run }) {
  if (!run) return null;

  const winner = deriveWinner(run);
  if (!winner) return null;

  const { name, tier, primaryMetric, primaryValue, rationale, interpretation, deltas } = winner;

  return (
    <div className={styles.card}>
      {/* ── Left: winner identity + metric + why ──────────────────── */}
      <div className={styles.left}>
        <div className={styles.eyebrow}>
          <span className={styles.eyebrowDot} />
          Recommended model
        </div>

        <div className={styles.nameRow}>
          <h3 className={styles.name}>{name}</h3>
          <TierBadge tier={tier} />
          <span className={styles.winnerBadge}>WINNER</span>
        </div>

        {/* Primary metric hero */}
        <div className={styles.metricBlock}>
          <div className={styles.metricLabel}>{primaryMetric}</div>
          <div className={styles.metricValue}>
            {formatMetric(primaryValue, { digits: 3 })}
          </div>
          <div className={styles.metricHint}>
            ranked by {primaryMetric} · lower is better
          </div>
        </div>

        {/* Summary sentence from interpretation */}
        {interpretation?.summary && (
          <p className={styles.summary}>{interpretation.summary}</p>
        )}

        {/* Strengths list */}
        {interpretation?.strengths?.length > 0 && (
          <ul className={styles.strengthsList}>
            {interpretation.strengths.map((s) => (
              <li key={s} className={styles.strengthItem}>
                <span className={styles.strengthMark} aria-hidden="true">✓</span>
                {s}
              </li>
            ))}
          </ul>
        )}

        {/* Fallback rationale if no interpretation */}
        {!interpretation?.summary && rationale && (
          <p className={styles.rationale}>
            <span className={styles.rationaleMark} aria-hidden="true">›</span>
            {rationale}
          </p>
        )}
      </div>

      <div className={styles.divider} aria-hidden="true" />

      {/* ── Right: margin vs other models ─────────────────────────── */}
      <div className={styles.right}>
        <div className={styles.rightHeader}>Margin vs other models</div>

        <ul className={styles.deltaList}>
          {deltas.map((d) => (
            <li key={d.name} className={styles.deltaItem}>
              <div className={styles.deltaTop}>
                <span className={styles.deltaModel}>
                  #{d.rank} {d.name}
                </span>
                <TierBadge tier={d.tier} />
              </div>

              <div className={styles.deltaRow}>
                <span className={styles.deltaMetricValue}>
                  {primaryMetric} {formatMetric(d.value)}
                </span>
                <span className={styles.deltaArrow} aria-hidden="true">→</span>
                <span className={styles.deltaPct}>
                  {formatPctImprovement(d.pctBetter)}
                </span>
              </div>

              <div className={styles.deltaAbs}>
                Δ {formatMetric(d.absDelta)} {primaryMetric} absolute difference
              </div>
            </li>
          ))}
        </ul>

        {/* Multiplier summary for judges */}
        {deltas.length > 0 && (
          <div className={styles.multiplierRow}>
            {deltas.map((d) => {
              const mult = d.value > 0 ? (d.value / primaryValue).toFixed(1) : null;
              return mult ? (
                <span key={d.name} className={styles.multiplierChip}>
                  {mult}× better than {d.name}
                </span>
              ) : null;
            })}
          </div>
        )}
      </div>
    </div>
  );
}
