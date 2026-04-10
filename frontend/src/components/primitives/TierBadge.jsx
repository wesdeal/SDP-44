import styles from "./TierBadge.module.css";

const TIER_LABEL = {
  baseline: "baseline",
  classical: "classical",
  specialized: "specialized",
};

export default function TierBadge({ tier }) {
  const key = (tier || "").toLowerCase();
  const label = TIER_LABEL[key] || tier || "—";
  return (
    <span className={`${styles.badge} ${styles[key] || ""}`}>{label}</span>
  );
}
