import { useState, useMemo } from "react";
import styles from "./ModelDetailSection.module.css";
import ModelSelector from "./ModelSelector.jsx";
import ModelDetailPanel from "./ModelDetailPanel.jsx";
import { deriveModelList } from "../../services/runService.js";

/**
 * Per-model detail section.
 * Data merging is delegated to deriveModelList() in runService.js.
 * Swap getRun() in api.js to get real backend data; this component
 * requires no changes as long as the artifact shapes match.
 */
export default function ModelDetailSection({ run }) {
  const models = useMemo(() => deriveModelList(run), [run]);
  const [selected, setSelected] = useState(models[0]?.name ?? null);
  const activeModel = models.find((m) => m.name === selected) ?? models[0];

  if (!models.length) return null;

  return (
    <div className={styles.root}>
      <ModelSelector models={models} selected={selected} onSelect={setSelected} />
      {activeModel && <ModelDetailPanel model={activeModel} />}
    </div>
  );
}
