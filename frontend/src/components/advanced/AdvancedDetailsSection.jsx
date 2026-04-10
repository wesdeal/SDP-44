import styles from "./AdvancedDetailsSection.module.css";
import DetailsAccordion from "./DetailsAccordion.jsx";
import SplitStrategyPanel from "./SplitStrategyPanel.jsx";
import PreprocessingDetailsPanel from "./PreprocessingDetailsPanel.jsx";
import FeatureEngineeringPanel from "./FeatureEngineeringPanel.jsx";
import ModelStrategyPanel from "./ModelStrategyPanel.jsx";
import PipelineMetadataPanel from "./PipelineMetadataPanel.jsx";

const fe = (run) => run.feature_engineering;

export default function AdvancedDetailsSection({ run }) {
  const totalFeatures = fe(run).total_features;
  const { train, validation, test } = run.eval_protocol.split_counts;
  const total = train + validation + test;

  return (
    <div className={styles.section}>
      {/* Section intro */}
      <div className={styles.intro}>
        <p className={styles.introText}>
          Expand any panel below to inspect the methodology behind this run —
          split strategy, preprocessing steps, engineered features, and per-model
          strategy decisions. Panels are collapsed by default and do not affect the
          summary above.
        </p>
      </div>

      {/* Accordion panels */}
      <div className={styles.accordions}>
        <DetailsAccordion
          title="Split Strategy"
          subtitle="chronological · no shuffle"
          badge="MAE ranking"
          badgeVariant="amber"
        >
          <SplitStrategyPanel run={run} />
        </DetailsAccordion>

        <DetailsAccordion
          title="Preprocessing & Feature Engineering"
          subtitle={`${run.preprocessing_plan.steps.length} steps applied`}
          badge={`${totalFeatures} features total`}
          badgeVariant="cyan"
        >
          <PreprocessingDetailsPanel run={run} />
        </DetailsAccordion>

        <DetailsAccordion
          title="Feature Inventory"
          subtitle="grouped by type"
          badge={`${totalFeatures} columns`}
          badgeVariant="default"
        >
          <FeatureEngineeringPanel run={run} />
        </DetailsAccordion>

        <DetailsAccordion
          title="Model Strategy Metadata"
          subtitle="per-model configuration & rationale"
          badge={`${run.selected_models.selected_models.length} models`}
          badgeVariant="purple"
        >
          <ModelStrategyPanel run={run} />
        </DetailsAccordion>

        <DetailsAccordion
          title="Pipeline Configuration"
          subtitle="run-level metadata"
          badge={run.pipeline_metadata.using_mock_data ? "mock data" : "live"}
          badgeVariant={run.pipeline_metadata.using_mock_data ? "amber" : "green"}
        >
          <PipelineMetadataPanel run={run} />
        </DetailsAccordion>
      </div>
    </div>
  );
}
