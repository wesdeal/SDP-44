import styles from "./RankBadge.module.css";

export default function RankBadge({ rank }) {
  const cls =
    rank === 1
      ? styles.rank1
      : rank === 2
      ? styles.rank2
      : rank === 3
      ? styles.rank3
      : styles.rankOther;
  return <span className={`${styles.badge} ${cls}`}>#{rank}</span>;
}
