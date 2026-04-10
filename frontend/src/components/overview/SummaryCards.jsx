import styles from "./SummaryCards.module.css";
import { humanizeTaskType } from "../../utils/format.js";
import { deriveSummaryStats } from "../../services/runService.js";

function SummaryCard({ label, value, sublabel, accent }) {
  return (
    <div className={`${styles.card} ${accent ? styles[accent] : ""}`}>
      <div className={styles.label}>{label}</div>
      <div className={styles.value}>{value}</div>
      {sublabel && <div className={styles.sublabel}>{sublabel}</div>}
    </div>
  );
}

export default function SummaryCards({ run }) {
  if (!run) return null;

  const stats = deriveSummaryStats(run);

  // Humanize the task type value in the first card
  const cards = stats.map((s) =>
    s.label === "Task type"
      ? { ...s, value: humanizeTaskType(s.value) }
      : s
  );

  return (
    <div className={styles.grid}>
      {cards.map((c) => (
        <SummaryCard key={c.label} {...c} />
      ))}
    </div>
  );
}
