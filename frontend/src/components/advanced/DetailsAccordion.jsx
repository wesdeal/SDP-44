import { useState } from "react";
import styles from "./DetailsAccordion.module.css";

/**
 * Reusable collapsible accordion panel.
 * Uses grid-template-rows: 0fr → 1fr for smooth CSS-only expand/collapse.
 *
 * Props:
 *   title        — string, required
 *   subtitle     — string, optional; shown beside the title at lower contrast
 *   badge        — string, optional; a right-aligned chip label
 *   badgeVariant — "default" | "cyan" | "amber" | "green" | "purple"
 *   defaultOpen  — bool, default false
 *   children     — panel body content
 */
export default function DetailsAccordion({
  title,
  subtitle,
  badge,
  badgeVariant = "default",
  defaultOpen = false,
  children,
}) {
  const [isOpen, setIsOpen] = useState(defaultOpen);

  return (
    <div className={`${styles.accordion} ${isOpen ? styles.open : ""}`}>
      <button
        className={styles.header}
        onClick={() => setIsOpen((o) => !o)}
        aria-expanded={isOpen}
        type="button"
      >
        <span className={styles.chevron} />
        <span className={styles.titleGroup}>
          <span className={styles.title}>{title}</span>
          {subtitle && <span className={styles.subtitle}>{subtitle}</span>}
        </span>
        {badge && (
          <span className={`${styles.badge} ${styles[`badge_${badgeVariant}`]}`}>
            {badge}
          </span>
        )}
      </button>

      <div className={styles.body}>
        <div className={styles.bodyInner}>{children}</div>
      </div>
    </div>
  );
}
