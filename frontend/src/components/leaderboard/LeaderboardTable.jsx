import styles from "./LeaderboardTable.module.css";
import RankBadge from "../primitives/RankBadge.jsx";
import TierBadge from "../primitives/TierBadge.jsx";
import {
  formatMetric,
  formatDuration,
  formatMs,
  formatMetricByName,
} from "../../utils/format.js";
import { deriveLeaderboardRows } from "../../services/runService.js";

function StatusPill({ status }) {
  const ok = status === "evaluated" || status === "success";
  return (
    <span className={`${styles.status} ${ok ? styles.statusOk : styles.statusBad}`}>
      <span className={styles.statusDot} aria-hidden="true" />
      {status}
    </span>
  );
}

export default function LeaderboardTable({ run }) {
  if (!run) return null;

  const rows = deriveLeaderboardRows(run);
  const primary = run.eval_protocol?.primary_metric ?? "MAE";

  return (
    <div className={styles.outer}>
      <div className={styles.scrollWrapper}>
        <table className={styles.table}>
          <thead>
            <tr>
              <th className={styles.thRank}>Rank</th>
              <th className={styles.thLeft}>Model</th>
              <th className={styles.thLeft}>Tier</th>
              <th className={`${styles.thNum} ${primary === "MAE" ? styles.thPrimary : ""}`}>
                MAE{primary === "MAE" && <span className={styles.primaryMark}>★</span>}
              </th>
              <th className={`${styles.thNum} ${primary === "RMSE" ? styles.thPrimary : ""}`}>
                RMSE{primary === "RMSE" && <span className={styles.primaryMark}>★</span>}
              </th>
              <th className={styles.thNum}>MAPE</th>
              <th className={styles.thNum}>sMAPE</th>
              <th className={styles.thNum}>Train time</th>
              <th className={styles.thNum}>Inference</th>
              <th className={styles.thLeft}>Status</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr
                key={r.name}
                className={`${styles.row} ${r.is_best ? styles.rowBest : ""}`}
              >
                <td className={styles.tdRank}>
                  <RankBadge rank={r.rank} />
                </td>
                <td className={styles.tdModel}>
                  <div className={styles.modelName}>
                    {r.name}
                    {r.is_best && (
                      <span className={styles.bestTag}>BEST</span>
                    )}
                  </div>
                </td>
                <td className={styles.tdTier}>
                  <TierBadge tier={r.tier} />
                </td>
                <td className={`${styles.num} ${primary === "MAE" ? styles.numPrimary : ""} ${r.is_best && primary === "MAE" ? styles.numWinner : ""}`}>
                  {formatMetric(r.MAE)}
                </td>
                <td className={`${styles.num} ${primary === "RMSE" ? styles.numPrimary : ""}`}>
                  {formatMetric(r.RMSE)}
                </td>
                <td className={styles.num}>
                  {formatMetricByName(r.MAPE, "MAPE")}
                </td>
                <td className={styles.num}>
                  {formatMetricByName(r.sMAPE, "sMAPE")}
                </td>
                <td className={styles.num}>
                  {formatDuration(r.training_seconds)}
                </td>
                <td className={styles.num}>
                  {formatMs(r.inference_ms)}
                </td>
                <td className={styles.tdStatus}>
                  <StatusPill status={r.status} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
