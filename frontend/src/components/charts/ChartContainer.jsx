import styles from "./ChartContainer.module.css";

/**
 * Reusable framing for any chart panel: title, optional subtitle/eyebrow,
 * and a directionality pill ("lower is better" / "higher is better")
 * pinned to the top-right so judges never have to guess.
 */
export default function ChartContainer({
  title,
  subtitle,
  eyebrow,
  direction = "lower_is_better",
  size = "default", // "default" | "compact"
  children,
}) {
  const dirLabel =
    direction === "higher_is_better" ? "higher is better" : "lower is better";

  return (
    <div
      className={`${styles.panel} ${size === "compact" ? styles.compact : ""}`}
    >
      <header className={styles.header}>
        <div className={styles.headLeft}>
          {eyebrow && <div className={styles.eyebrow}>{eyebrow}</div>}
          {title && <h3 className={styles.title}>{title}</h3>}
          {subtitle && <div className={styles.subtitle}>{subtitle}</div>}
        </div>
        <div className={styles.headRight}>
          <span
            className={`${styles.dirPill} ${
              direction === "higher_is_better" ? styles.dirUp : styles.dirDown
            }`}
          >
            <span className={styles.dirArrow}>
              {direction === "higher_is_better" ? "↑" : "↓"}
            </span>
            {dirLabel}
          </span>
        </div>
      </header>
      <div className={styles.body}>{children}</div>
    </div>
  );
}
