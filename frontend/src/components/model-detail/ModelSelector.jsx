import styles from "./ModelSelector.module.css";
import RankBadge from "../primitives/RankBadge.jsx";
import TierBadge from "../primitives/TierBadge.jsx";
import { colorForModel } from "../charts/chartTheme.js";

export default function ModelSelector({ models, selected, onSelect }) {
  return (
    <div className={styles.selector}>
      {models.map((m) => {
        const isActive = selected === m.name;
        return (
          <button
            key={m.name}
            className={`${styles.tab} ${isActive ? styles.active : ""}`}
            style={isActive ? { "--tab-accent": colorForModel(m.name) } : {}}
            onClick={() => onSelect(m.name)}
          >
            <RankBadge rank={m.rank} />
            <span className={styles.name}>{m.name}</span>
            <TierBadge tier={m.tier} />
          </button>
        );
      })}
    </div>
  );
}
