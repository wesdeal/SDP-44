"""agents/artifact_assembly_agent.py — Artifact Assembly Agent (Phase 7.4)

Gathers all run artifacts and assembles a frontend-consumable dashboard/ bundle
under runs/{run_id}/dashboard/. Writes no new computed data — pure assembly.

Architecture authority: AGENT_ARCHITECTURE.md §4 Agent 9.
No legacy files imported. No other agent called directly.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from typing import Optional

from core.manifest import read_manifest, update_stage


class ArtifactAssemblyAgent:
    """Agent 9: Artifact Assembly Agent."""

    def run(self, manifest_path: str) -> None:
        """Assemble dashboard/ bundle from upstream artifacts.

        Sets stage status: running → completed | failed.
        This stage never hard-aborts on missing artifacts; it assembles
        whatever is available and records gaps in pipeline_log.json.

        Args:
            manifest_path: Path to the run's job_manifest.json.
        """
        update_stage(manifest_path, "artifact_assembly", "running")

        try:
            manifest = read_manifest(manifest_path)
            run_id = manifest["run_id"]

            run_dir = os.path.dirname(os.path.abspath(manifest_path))
            dashboard_dir = os.path.join(run_dir, "dashboard")
            dashboard_plots_dir = os.path.join(dashboard_dir, "plots")
            dashboard_cards_dir = os.path.join(dashboard_dir, "model_cards")
            dashboard_artifacts_dir = os.path.join(dashboard_dir, "artifacts")
            for d in (
                dashboard_dir,
                dashboard_plots_dir,
                dashboard_cards_dir,
                dashboard_artifacts_dir,
            ):
                os.makedirs(d, exist_ok=True)

            missing_log: list[dict] = []

            # ── Load upstream artifacts ──────────────────────────────────────
            eval_report = _load_json_safe(
                _artifact_path(manifest, "evaluation", "evaluation_report"),
                "evaluation_report",
                missing_log,
            )
            comparison_table = _load_json_safe(
                _artifact_path(manifest, "evaluation", "comparison_table"),
                "comparison_table",
                missing_log,
            )
            training_results = _load_json_safe(
                _artifact_path(manifest, "training", "training_results"),
                "training_results",
                missing_log,
            )
            selected_models_doc = _load_json_safe(
                _artifact_path(manifest, "model_selection", "selected_models"),
                "selected_models",
                missing_log,
            )
            eval_protocol = _load_json_safe(
                _artifact_path(manifest, "evaluation_protocol", "eval_protocol"),
                "eval_protocol",
                missing_log,
            )

            # Optional: ensemble report — not in manifest schema, check filesystem
            _ensemble_candidate = os.path.join(
                run_dir, "artifacts", "ensemble_report.json"
            )
            ensemble_report = _load_json_safe(
                _ensemble_candidate if os.path.exists(_ensemble_candidate) else None,
                "ensemble_report",
                missing_log,
                optional=True,
            )

            # ── run_summary.json ─────────────────────────────────────────────
            run_summary = {
                "run_id": run_id,
                "created_at": manifest.get("created_at"),
                "assembled_at": _now_iso(),
                "status": manifest.get("status", "unknown"),
                "input": manifest.get("input", {}),
                "config": manifest.get("config", {}),
                "stages": {
                    name: {
                        "status": stage.get("status"),
                        "started_at": stage.get("started_at"),
                        "completed_at": stage.get("completed_at"),
                        "error": stage.get("error"),
                    }
                    for name, stage in manifest.get("stages", {}).items()
                },
            }
            _write_json_atomic(
                os.path.join(dashboard_dir, "run_summary.json"), run_summary
            )

            # ── leaderboard.json ─────────────────────────────────────────────
            if (
                comparison_table is not None
                and eval_report is not None
                and eval_protocol is not None
            ):
                primary_metric = eval_report.get("primary_metric", "")
                higher_is_better = _metric_higher_is_better(
                    eval_protocol, primary_metric
                )
                leaderboard = _build_leaderboard(
                    run_id=run_id,
                    manifest=manifest,
                    comparison_table=comparison_table,
                    eval_protocol=eval_protocol,
                    primary_metric=primary_metric,
                    higher_is_better=higher_is_better,
                    ensemble_report=ensemble_report,
                )
                _write_json_atomic(
                    os.path.join(dashboard_dir, "leaderboard.json"), leaderboard
                )
            else:
                missing_log.append({
                    "artifact": "leaderboard.json",
                    "missing": True,
                    "reason": (
                        "upstream comparison_table, evaluation_report, or "
                        "eval_protocol missing"
                    ),
                })

            # ── model_cards/ ─────────────────────────────────────────────────
            if (
                eval_report is not None
                and training_results is not None
                and selected_models_doc is not None
                and comparison_table is not None
            ):
                _write_model_cards(
                    cards_dir=dashboard_cards_dir,
                    eval_report=eval_report,
                    training_results=training_results,
                    selected_models_doc=selected_models_doc,
                    comparison_table=comparison_table,
                    dashboard_dir=dashboard_dir,
                    missing_log=missing_log,
                )
            else:
                missing_log.append({
                    "artifact": "model_cards/",
                    "missing": True,
                    "reason": (
                        "upstream eval_report, training_results, selected_models, "
                        "or comparison_table missing"
                    ),
                })

            # ── Copy per-model eval plots ────────────────────────────────────
            if eval_report is not None:
                for model_record in eval_report.get("models", []):
                    plot_path = model_record.get("plot_path")
                    if plot_path and os.path.exists(plot_path):
                        shutil.copy2(
                            plot_path,
                            os.path.join(
                                dashboard_plots_dir, os.path.basename(plot_path)
                            ),
                        )
                    elif plot_path:
                        missing_log.append({
                            "artifact": f"plots/{os.path.basename(plot_path)}",
                            "missing": True,
                            "reason": f"plot file not found at {plot_path!r}",
                        })

            # ── comparison_chart.png ─────────────────────────────────────────
            chart_path = os.path.join(dashboard_plots_dir, "comparison_chart.png")
            if comparison_table is not None:
                chart_ok = _generate_comparison_chart(comparison_table, chart_path)
                if not chart_ok:
                    missing_log.append({
                        "artifact": "plots/comparison_chart.png",
                        "missing": True,
                        "reason": "chart generation failed",
                    })
            else:
                missing_log.append({
                    "artifact": "plots/comparison_chart.png",
                    "missing": True,
                    "reason": "comparison_table missing",
                })

            # ── Passthrough artifacts ────────────────────────────────────────
            _PASSTHROUGHS = [
                ("ingestion", "dataset_profile", "dataset_profile.json"),
                ("problem_classification", "task_spec", "task_spec.json"),
                ("preprocessing_planning", "preprocessing_plan", "preprocessing_plan.json"),
                ("evaluation_protocol", "eval_protocol", "eval_protocol.json"),
            ]
            for stage_key, artifact_key, dest_name in _PASSTHROUGHS:
                src = _artifact_path(manifest, stage_key, artifact_key)
                if src and os.path.exists(src):
                    shutil.copy2(
                        src, os.path.join(dashboard_artifacts_dir, dest_name)
                    )
                else:
                    missing_log.append({
                        "artifact": f"artifacts/{dest_name}",
                        "missing": True,
                        "reason": (
                            f"source not found: stage={stage_key!r} "
                            f"key={artifact_key!r} path={src!r}"
                        ),
                    })

            # ── README.txt ───────────────────────────────────────────────────
            _write_readme(os.path.join(dashboard_dir, "README.txt"), run_id)

            # ── pipeline_log.json ────────────────────────────────────────────
            pipeline_log = _build_pipeline_log(manifest, missing_log)
            _write_json_atomic(
                os.path.join(dashboard_dir, "pipeline_log.json"), pipeline_log
            )

            # ── Finalize ─────────────────────────────────────────────────────
            update_stage(
                manifest_path,
                "artifact_assembly",
                "completed",
                artifacts={"dashboard_bundle": dashboard_dir},
            )

        except Exception as exc:
            update_stage(
                manifest_path, "artifact_assembly", "failed", error=str(exc)
            )
            raise


# ---------------------------------------------------------------------------
# Leaderboard construction
# ---------------------------------------------------------------------------


def _build_leaderboard(
    run_id: str,
    manifest: dict,
    comparison_table: dict,
    eval_protocol: dict,
    primary_metric: str,
    higher_is_better: bool,
    ensemble_report: Optional[dict],
) -> dict:
    ranking = comparison_table.get("ranking", [])
    dataset_name = manifest.get("input", {}).get("original_filename", "unknown")
    task_type = eval_protocol.get("task_type", "")

    # Baseline reference: the last-ranked (worst) entry in the sorted ranking
    baseline_value: Optional[float] = None
    if ranking:
        baseline_value = ranking[-1].get("primary_metric_value")

    _RECOMMENDATION = {1: "Best Overall", 2: "Strong Performer"}

    models = []
    for entry in ranking:
        rank = entry["rank"]
        model_value = entry.get("primary_metric_value")

        delta, delta_pct = _compute_delta(
            model_value, baseline_value, higher_is_better
        )
        models.append({
            "rank": rank,
            "name": entry.get("model_name"),
            "tier": entry.get("tier"),
            "primary_metric_value": model_value,
            "delta_from_baseline": delta,
            "delta_pct": delta_pct,
            "recommendation": _RECOMMENDATION.get(rank, "Baseline"),
        })

    # Optionally append ensemble row (not re-ranked; appended as addendum)
    if ensemble_report is not None:
        ens_value = ensemble_report.get("metrics", {}).get(primary_metric)
        e_delta, e_delta_pct = _compute_delta(
            ens_value, baseline_value, higher_is_better
        )
        models.append({
            "rank": None,
            "name": "Ensemble",
            "tier": "ensemble",
            "primary_metric_value": ens_value,
            "delta_from_baseline": e_delta,
            "delta_pct": e_delta_pct,
            "recommendation": "Ensemble",
        })

    return {
        "run_id": run_id,
        "dataset_name": dataset_name,
        "task_type": task_type,
        "primary_metric": primary_metric,
        "higher_is_better": higher_is_better,
        "models": models,
        "generated_at": _now_iso(),
    }


def _compute_delta(
    model_value: Optional[float],
    baseline_value: Optional[float],
    higher_is_better: bool,
) -> tuple[float, float]:
    """Return (delta_from_baseline, delta_pct).

    Positive values always represent improvement over baseline.
    Returns (0.0, 0.0) if either value is None or baseline is zero.
    """
    if model_value is None or baseline_value is None:
        return 0.0, 0.0
    if higher_is_better:
        delta = model_value - baseline_value
    else:
        delta = baseline_value - model_value
    if baseline_value == 0:
        delta_pct = 0.0
    else:
        delta_pct = delta / abs(baseline_value) * 100.0
    return round(delta, 6), round(delta_pct, 4)


# ---------------------------------------------------------------------------
# Model card construction
# ---------------------------------------------------------------------------


def _write_model_cards(
    cards_dir: str,
    eval_report: dict,
    training_results: dict,
    selected_models_doc: dict,
    comparison_table: dict,
    dashboard_dir: str,
    missing_log: list,
) -> None:
    """Write one {ModelName}_card.json per evaluated model."""
    training_by_name = {
        m["name"]: m for m in training_results.get("models", [])
    }
    selection_by_name = {
        m["name"]: m for m in selected_models_doc.get("selected_models", [])
    }
    ranking_by_name = {
        r["model_name"]: r for r in comparison_table.get("ranking", [])
    }

    primary_metric = eval_report.get("primary_metric", "")
    task_type = selected_models_doc.get("task_type", "")

    for model_record in eval_report.get("models", []):
        model_name = model_record["name"]
        train_info = training_by_name.get(model_name, {})
        sel_info = selection_by_name.get(model_name, {})
        rank_info = ranking_by_name.get(model_name, {})

        rank = rank_info.get("rank")

        # Plot path: relative from dashboard/
        plot_abs = model_record.get("plot_path")
        if plot_abs:
            plot_rel = f"plots/{os.path.basename(plot_abs)}"
        else:
            plot_rel = None

        # Trained model path: relative from dashboard/
        model_path_abs = train_info.get("model_path")
        if model_path_abs:
            try:
                model_path_rel = os.path.relpath(model_path_abs, dashboard_dir)
            except ValueError:
                model_path_rel = model_path_abs
        else:
            model_path_rel = None

        card = {
            "model_name": model_name,
            "tier": model_record.get("tier", train_info.get("tier", "")),
            "task_type": task_type,
            "rationale_for_selection": sel_info.get("rationale", ""),
            "hyperparameters": train_info.get("hyperparameters", {}),
            "hyperparameter_source": train_info.get(
                "hyperparameter_source", "default"
            ),
            "training_duration_seconds": train_info.get(
                "training_duration_seconds"
            ),
            "metrics": {
                "primary_metric_name": primary_metric,
                "primary_metric_value": model_record.get("metrics", {}).get(
                    primary_metric
                ),
                "all_metrics": model_record.get("metrics", {}),
            },
            "rank": rank,
            "is_recommended": rank == 1,
            "plot_path": plot_rel,
            "trained_model_path": model_path_rel,
        }

        card_path = os.path.join(cards_dir, f"{model_name}_card.json")
        try:
            _write_json_atomic(card_path, card)
        except Exception as exc:  # noqa: BLE001
            missing_log.append({
                "artifact": f"model_cards/{model_name}_card.json",
                "missing": True,
                "reason": f"failed to write card: {exc}",
            })


# ---------------------------------------------------------------------------
# Comparison chart
# ---------------------------------------------------------------------------


def _generate_comparison_chart(comparison_table: dict, chart_path: str) -> bool:
    """Generate a bar chart of primary metric values. Returns True on success."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ranking = comparison_table.get("ranking", [])
        metric_name = comparison_table.get("ranked_by", "metric")

        valid = [
            r for r in ranking if r.get("primary_metric_value") is not None
        ]
        if not valid:
            return False

        names = [r["model_name"] for r in valid]
        values = [r["primary_metric_value"] for r in valid]

        fig, ax = plt.subplots(figsize=(max(4, len(names) * 2), 4))
        bars = ax.bar(range(len(names)), values)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=15, ha="right")
        ax.set_ylabel(metric_name)
        ax.set_title(f"Model Comparison — {metric_name}")

        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.4g}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        plt.tight_layout()
        fig.savefig(chart_path, dpi=80)
        plt.close(fig)
        return True
    except Exception:  # noqa: BLE001
        return False


# ---------------------------------------------------------------------------
# Pipeline log
# ---------------------------------------------------------------------------


def _build_pipeline_log(manifest: dict, missing_log: list) -> dict:
    """Build pipeline_log.json from manifest stage timing and missing artifact log."""
    stages = {}
    for name, stage in manifest.get("stages", {}).items():
        started_at = stage.get("started_at")
        completed_at = stage.get("completed_at")
        duration_s = None
        if started_at and completed_at:
            try:
                t0 = datetime.fromisoformat(started_at)
                t1 = datetime.fromisoformat(completed_at)
                duration_s = round((t1 - t0).total_seconds(), 3)
            except Exception:  # noqa: BLE001
                pass
        stages[name] = {
            "status": stage.get("status"),
            "started_at": started_at,
            "completed_at": completed_at,
            "duration_seconds": duration_s,
            "error": stage.get("error"),
        }

    return {
        "run_id": manifest.get("run_id"),
        "stages": stages,
        "missing_artifacts": missing_log,
        "assembled_at": _now_iso(),
    }


# ---------------------------------------------------------------------------
# README
# ---------------------------------------------------------------------------


def _write_readme(path: str, run_id: str) -> None:
    content = (
        "Pipeline Dashboard Bundle\n"
        "=========================\n"
        f"Run ID: {run_id}\n"
        "\n"
        "Contents:\n"
        "  run_summary.json        — Final pipeline run state and stage statuses\n"
        "  leaderboard.json        — Ranked model comparison by primary metric\n"
        "  pipeline_log.json       — Stage timing, status, and missing artifact log\n"
        "  model_cards/            — Per-model cards (metrics, hyperparameters, rank)\n"
        "  plots/                  — Per-model eval plots + comparison_chart.png\n"
        "  artifacts/              — Passthrough artifacts from upstream stages\n"
        "\n"
        "Generated by ArtifactAssemblyAgent.\n"
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _artifact_path(manifest: dict, stage: str, key: str) -> Optional[str]:
    """Safely retrieve an artifact path from the manifest stages dict."""
    try:
        return manifest["stages"][stage]["artifacts"][key]
    except (KeyError, TypeError):
        return None


def _load_json_safe(
    path: Optional[str],
    artifact_name: str,
    missing_log: list,
    optional: bool = False,
) -> Optional[dict]:
    """Load a JSON file; on failure append to missing_log and return None."""
    if path is None:
        if not optional:
            missing_log.append({
                "artifact": artifact_name,
                "missing": True,
                "reason": "path not found in manifest",
            })
        return None
    if not os.path.exists(path):
        if not optional:
            missing_log.append({
                "artifact": artifact_name,
                "missing": True,
                "reason": f"file not found at {path!r}",
            })
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:  # noqa: BLE001
        missing_log.append({
            "artifact": artifact_name,
            "missing": True,
            "reason": f"JSON parse error: {exc}",
        })
        return None


def _metric_higher_is_better(eval_protocol: dict, metric_name: str) -> bool:
    """Return higher_is_better for metric_name from eval_protocol.metrics list."""
    for m in eval_protocol.get("metrics", []):
        if m.get("name") == metric_name:
            return bool(m.get("higher_is_better", False))
    return False


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_atomic(path: str, data: dict) -> None:
    """Write *data* as JSON to *path* atomically (temp-file + rename)."""
    dir_name = os.path.dirname(path)
    os.makedirs(dir_name, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
