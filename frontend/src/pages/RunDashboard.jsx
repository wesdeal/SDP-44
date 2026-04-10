import styles from "./RunDashboard.module.css";
import RunOverviewHeader from "../components/overview/RunOverviewHeader.jsx";
import SummaryCards from "../components/overview/SummaryCards.jsx";
import RecommendationCard from "../components/recommendation/RecommendationCard.jsx";
import LeaderboardTable from "../components/leaderboard/LeaderboardTable.jsx";
import MetricComparisonSection from "../components/charts/MetricComparisonSection.jsx";
import ModelDetailSection from "../components/model-detail/ModelDetailSection.jsx";
import AdvancedDetailsSection from "../components/advanced/AdvancedDetailsSection.jsx";

/**
 * Section header with a label and optional right-side annotation.
 * All labels are monospace small-caps to preserve the terminal aesthetic.
 */
function SectionHeader({ label, annotation }) {
  return (
    <div className={styles.sectionHeader}>
      <span className={styles.sectionLabel}>{label}</span>
      {annotation && (
        <span className={styles.sectionAnnotation}>{annotation}</span>
      )}
    </div>
  );
}

export default function RunDashboard({ run }) {
  if (!run) return null;

  const winnerName =
    run.comparison_table?.ranking?.find((r) => r.is_best)?.model_name ?? null;

  return (
    <div className={styles.stack}>

      {/* ── Run overview ──────────────────────────────────────────── */}
      <section className={styles.block}>
        <SectionHeader label="Run Overview" />
        <RunOverviewHeader run={run} />
        <SummaryCards run={run} />
      </section>

      {/* ── Winner recommendation ─────────────────────────────────── */}
      <section className={styles.block}>
        <SectionHeader
          label="Recommendation"
          annotation={winnerName ? `Winner · ${winnerName}` : undefined}
        />
        <RecommendationCard run={run} />
      </section>

      {/* ── Model leaderboard ─────────────────────────────────────── */}
      <section className={styles.block}>
        <SectionHeader label="Leaderboard" annotation="ranked by MAE · lower is better" />
        <LeaderboardTable run={run} />
      </section>

      {/* ── Metric comparison charts ──────────────────────────────── */}
      <section className={styles.block}>
        <SectionHeader label="Metric Comparison" annotation="MAE · RMSE · MAPE · sMAPE" />
        <MetricComparisonSection run={run} />
      </section>

      {/* ── Per-model detail ──────────────────────────────────────── */}
      <section className={styles.block}>
        <SectionHeader
          label="Per-Model Detail"
          annotation={`${run.comparison_table?.ranking?.length ?? 0} models`}
        />
        <ModelDetailSection run={run} />
      </section>

      {/* ── Advanced technical details ────────────────────────────── */}
      <section className={`${styles.block} ${styles.lastBlock}`}>
        <SectionHeader label="Pipeline Methodology" annotation="expand panels to inspect" />
        <AdvancedDetailsSection run={run} />
      </section>

    </div>
  );
}
