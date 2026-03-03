"""
Enhanced Metadata Extraction with Preprocessing Pipeline Integration
Returns structured, actionable metadata that feeds directly into preprocessing
"""

from openai import OpenAI
import json
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
import pandas as pd
from datetime import datetime
from pathlib import Path
import numpy as np

# ============================================================================
# STRUCTURED DATA CLASSES
# ============================================================================

@dataclass
class ColumnSummary:
    name: str
    inferred_type: str
    example_values: List[Any]
    missing_fraction: float
    has_outliers: bool = False
    is_temporal: bool = False

@dataclass
class PreprocessingStep:
    """Structured preprocessing step with parameters"""
    method: str  # e.g., 'z_norm', 'imputation', 'detrend'
    parameters: Dict[str, Any]
    reason: str
    order: int  # Execution order

@dataclass
class DatasetMetadata:
    """Complete structured metadata"""
    dataset_name: str
    dataset_description: str
    data_source: str
    ingestion_date: str
    num_rows: int
    num_columns: int
    column_summaries: List[ColumnSummary]
    target_variable: str
    temporal_coverage: Dict[str, str]
    known_data_quality_issues: List[str]
    preprocessing_steps: List[PreprocessingStep]  # Actionable list!
    
    # Optional fields
    licenses_or_usage_rights: str = "Unknown"
    class_or_value_distribution_for_target: Dict = None
    related_documentation_links: List[str] = None
    security_or_privacy_notes: str = None
    
    # Derived insights
    has_missing_data: bool = False
    has_seasonality: bool = False
    has_trend: bool = False
    is_multivariate: bool = False
    recommended_models: List[str] = None
    model_reasoning: Dict[str, str] = None  # Justification for each recommended model

# ============================================================================
# ENHANCED METADATA EXTRACTOR
# ============================================================================

class MetadataExtractor:
    """Extract and structure metadata for automated pipeline"""
    
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
    
    def extract_from_file(self, file_path: str) -> DatasetMetadata:
        """Main extraction method with structured output"""

        # Store the input file path for later use
        self.input_file_path = file_path

        # Upload file to OpenAI
        file = self.client.files.create(
            file=open(file_path, "rb"),
            purpose="assistants"
        )

        # Create enhanced prompt
        prompt = self._create_enhanced_prompt()

        # Call OpenAI with code interpreter
        response = self.client.responses.create(
            model="gpt-5-mini",
            input=prompt,
            tools=[{
                "type": "code_interpreter",
                "container": {
                    "type": "auto",
                    "file_ids": [file.id]
                }
            }]
        )

        # Parse response
        output_text = response.output_text
        raw_metadata = json.loads(output_text)

        # Convert to structured format
        metadata = self._structure_metadata(raw_metadata)

        return metadata
    
    def _create_enhanced_prompt(self) -> str:
        """Create prompt that returns actionable preprocessing steps"""
        return """
You are a data analysis assistant specializing in time series data.

Analyze the uploaded dataset and extract metadata. Most importantly, provide ACTIONABLE preprocessing recommendations.

Required Analysis:
1. Basic Info: dataset_name, description, num_rows, num_columns
2. Column Analysis: For each column, provide name, type, examples, missing_fraction
3. Target Variable: Identify the target column for prediction
4. Temporal Coverage: earliest and latest timestamps (if applicable)
5. Data Quality: List specific issues found (missing values, outliers, inconsistent formats)

CRITICAL - Preprocessing Steps:
Analyze the data characteristics and recommend SPECIFIC preprocessing steps in EXECUTION ORDER.
For each step, provide:
- method: exact name from [imputation, detrend, z_norm, min_max, smoothing, differencing, log_transform, remove_outliers]
- parameters: specific values (e.g., {"strategy": "mean"} for imputation)
- reason: why this step is needed
- order: execution sequence (1, 2, 3, ...)

Decision Rules for Preprocessing:
- If missing_fraction > 0 → imputation (order: 1)
- If data has trend → detrend (order: 2)
- If different scales across columns → z_norm or min_max (order: 3)
- If noisy data → smoothing (order: 4)
- If non-stationary time series → differencing
- If right-skewed distribution → log_transform
- If outliers detected → remove_outliers (early, order: 1-2)

Additional Insights:
- has_missing_data: boolean
- has_seasonality: boolean (look for repeating patterns)
- has_trend: boolean (look for increasing/decreasing pattern)
- is_multivariate: boolean (more than one feature)
- recommended_models: Recommend 3-5 most suitable models for this dataset based on data characteristics.
  You may recommend ANY models you deem appropriate, including but not limited to:
  * Tree-based: RandomForest, XGBoost, LightGBM, CatBoost, GradientBoosting
  * Deep Learning: LSTM, GRU, Transformer, TCN (Temporal Convolutional Network)
  * Time Series Specific: ARIMA, SARIMA, Prophet, Chronos, TimesFM, N-BEATS, N-HiTS
  * Statistical: VAR, VECM, Exponential Smoothing
  * Hybrid: Any combination or ensemble approaches

  IMPORTANT:
  - Recommend 3-5 models (no more than 5)
  - Prioritize models best suited for the detected patterns (seasonality, trend, etc.)
  - Include at least one modern deep learning time series model if applicable (e.g., Chronos for zero-shot forecasting)
  - Consider dataset size: small datasets (<1000) favor simpler models, large datasets (>10000) can use complex models

Return ONLY valid JSON in this EXACT structure:
{
  "dataset_name": "string",
  "dataset_description": "string",
  "data_source": "string",
  "ingestion_date": "YYYY-MM-DD",
  "num_rows": int,
  "num_columns": int,
  "column_summaries": [
    {
      "name": "string",
      "inferred_type": "numeric|categorical|datetime",
      "example_values": [value1, value2, value3],
      "missing_fraction": float,
      "has_outliers": boolean,
      "is_temporal": boolean
    }
  ],
  "target_variable": "string",
  "temporal_coverage": {
    "earliest_timestamp": "string",
    "latest_timestamp": "string"
  },
  "known_data_quality_issues": ["issue1", "issue2"],
  "preprocessing_steps": [
    {
      "method": "imputation",
      "parameters": {"strategy": "mean"},
      "reason": "20% missing values detected",
      "order": 1
    },
    {
      "method": "detrend",
      "parameters": {"type": "linear"},
      "reason": "Strong upward trend detected",
      "order": 2
    },
    {
      "method": "z_norm",
      "parameters": {},
      "reason": "Features have different scales",
      "order": 3
    }
  ],
  "has_missing_data": boolean,
  "has_seasonality": boolean,
  "has_trend": boolean,
  "is_multivariate": boolean,
  "recommended_models": ["model1", "model2", "model3", "model4"],
  "model_reasoning": {
    "model1": "Brief justification for recommending this model",
    "model2": "Brief justification for recommending this model"
  },
  "licenses_or_usage_rights": "string",
  "related_documentation_links": [],
  "security_or_privacy_notes": "string"
}

Be specific and actionable. The preprocessing_steps list will be executed automatically.
"""
    
    def _structure_metadata(self, raw_data: Dict) -> DatasetMetadata:
        """Convert raw JSON to structured dataclass"""
        
        # Convert column summaries
        column_summaries = [
            ColumnSummary(**col) for col in raw_data.get('column_summaries', [])
        ]
        
        # Convert preprocessing steps
        preprocessing_steps = [
            PreprocessingStep(**step) 
            for step in sorted(
                raw_data.get('preprocessing_steps', []),
                key=lambda x: x.get('order', 999)
            )
        ]
        
        # Build metadata object
        metadata = DatasetMetadata(
            dataset_name=raw_data.get('dataset_name', 'Unknown'),
            dataset_description=raw_data.get('dataset_description', ''),
            data_source=raw_data.get('data_source', 'Unknown'),
            ingestion_date=raw_data.get('ingestion_date', datetime.now().strftime('%Y-%m-%d')),
            num_rows=raw_data.get('num_rows', 0),
            num_columns=raw_data.get('num_columns', 0),
            column_summaries=column_summaries,
            target_variable=raw_data.get('target_variable', ''),
            temporal_coverage=raw_data.get('temporal_coverage', {}),
            known_data_quality_issues=raw_data.get('known_data_quality_issues', []),
            preprocessing_steps=preprocessing_steps,
            has_missing_data=raw_data.get('has_missing_data', False),
            has_seasonality=raw_data.get('has_seasonality', False),
            has_trend=raw_data.get('has_trend', False),
            is_multivariate=raw_data.get('is_multivariate', False),
            recommended_models=raw_data.get('recommended_models', []),
            model_reasoning=raw_data.get('model_reasoning', {}),
            licenses_or_usage_rights=raw_data.get('licenses_or_usage_rights', 'Unknown'),
            related_documentation_links=raw_data.get('related_documentation_links', []),
            security_or_privacy_notes=raw_data.get('security_or_privacy_notes', '')
        )
        
        return metadata
    
    def get_preprocessing_pipeline(self, metadata: DatasetMetadata) -> List[str]:
        """Extract just the preprocessing method names in order"""
        return [step.method for step in metadata.preprocessing_steps]
    
    def get_preprocessing_config(self, metadata: DatasetMetadata) -> List[Dict]:
        """Get full preprocessing configuration with parameters"""
        return [
            {
                'method': step.method,
                'params': step.parameters,
                'reason': step.reason
            }
            for step in metadata.preprocessing_steps
        ]
    
    def print_summary(self, metadata: DatasetMetadata):
        """Print human-readable summary"""
        print("=" * 60)
        print(f"DATASET: {metadata.dataset_name}")
        print("=" * 60)
        print(f"Description: {metadata.dataset_description}")
        print(f"Rows: {metadata.num_rows:,} | Columns: {metadata.num_columns}")
        print(f"Target Variable: {metadata.target_variable}")
        print(f"\nData Characteristics:")
        print(f"  - Missing Data: {metadata.has_missing_data}")
        print(f"  - Seasonality: {metadata.has_seasonality}")
        print(f"  - Trend: {metadata.has_trend}")
        print(f"  - Multivariate: {metadata.is_multivariate}")
        
        print(f"\nRecommended Preprocessing Pipeline:")
        for step in metadata.preprocessing_steps:
            params_str = json.dumps(step.parameters) if step.parameters else "{}"
            print(f"  {step.order}. {step.method}{params_str}")
            print(f"     → {step.reason}")
        
        print(f"\nRecommended Models: {', '.join(metadata.recommended_models)}")
        
        if metadata.known_data_quality_issues:
            print(f"\nKnown Issues:")
            for issue in metadata.known_data_quality_issues:
                print(f"  - {issue}")
        
        print("=" * 60)

# ============================================================================
# INTEGRATION WITH PREPROCESSING PIPELINE
# ============================================================================

class PreprocessingPipeline:
    """Execute preprocessing steps from metadata"""
    
    def __init__(self):
        self.methods = {
            'imputation': self.impute_missing,
            'detrend': self.detrend,
            'z_norm': self.z_normalization,
            'min_max': self.min_max_normalization,
            'smoothing': self.moving_average_smoothing,
            'differencing': self.differencing,
            'log_transform': self.log_transform,
            'remove_outliers': self.remove_outliers,
        }
    
    def apply_from_metadata(self, df: pd.DataFrame, metadata: DatasetMetadata) -> pd.DataFrame:
        """Apply preprocessing steps from metadata in correct order"""
        print(f"\nApplying {len(metadata.preprocessing_steps)} preprocessing steps...")
        
        for step in metadata.preprocessing_steps:
            if step.method in self.methods:
                print(f"  {step.order}. Applying {step.method}... ", end='')
                df = self.methods[step.method](df, **step.parameters)
                print("✓")
            else:
                print(f"  Warning: Unknown method '{step.method}', skipping")
        
        return df
    
    # Preprocessing methods (implement these)
    def impute_missing(self, df: pd.DataFrame, strategy='mean', **kwargs) -> pd.DataFrame:
        print(f"Imputing missing values using {strategy} strategy...")
        # Separate numeric and non-numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns

        df_copy = df.copy()

        if strategy == 'mean':
            df_copy[numeric_cols] = df_copy[numeric_cols].fillna(df_copy[numeric_cols].mean())
        elif strategy == 'median':
            df_copy[numeric_cols] = df_copy[numeric_cols].fillna(df_copy[numeric_cols].median())
        elif strategy == 'forward':
            df_copy[numeric_cols] = df_copy[numeric_cols].fillna(method='ffill')
        else:
            df_copy[numeric_cols] = df_copy[numeric_cols].interpolate()

        # For non-numeric, use forward fill or most frequent
        if len(non_numeric_cols) > 0:
            df_copy[non_numeric_cols] = df_copy[non_numeric_cols].fillna(method='ffill')

        return df_copy

    def detrend(self, df: pd.DataFrame, type='linear', **kwargs) -> pd.DataFrame:
        from scipy import signal
        print("Detrending data...")
        df_copy = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            df_copy[col] = signal.detrend(df_copy[col])

        return df_copy

    def z_normalization(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        print("Applying Z-Normalization...")
        df_copy = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        df_copy[numeric_cols] = (df_copy[numeric_cols] - df_copy[numeric_cols].mean()) / df_copy[numeric_cols].std()

        return df_copy

    def min_max_normalization(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        print("Applying Min-Max Normalization...")
        df_copy = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        df_copy[numeric_cols] = (df_copy[numeric_cols] - df_copy[numeric_cols].min()) / (df_copy[numeric_cols].max() - df_copy[numeric_cols].min())

        return df_copy

    def moving_average_smoothing(self, df: pd.DataFrame, window=3, **kwargs) -> pd.DataFrame:
        print(f"Applying Moving Average Smoothing with window={window}...")
        df_copy = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        df_copy[numeric_cols] = df_copy[numeric_cols].rolling(window=window, center=True).mean().fillna(method='bfill').fillna(method='ffill')

        return df_copy

    def differencing(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        print("Applying Differencing...")
        df_copy = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        df_copy[numeric_cols] = df_copy[numeric_cols].diff().fillna(0)

        return df_copy

    def log_transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        print("Applying Log Transform...")
        df_copy = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        # Apply log transform only to positive values
        for col in numeric_cols:
            df_copy[col] = np.log1p(df_copy[col].clip(lower=0))

        return df_copy
    
    def remove_outliers(self, df: pd.DataFrame, method='iqr', threshold=1.5, **kwargs) -> pd.DataFrame:
        if method == 'iqr':
            print("Removing outliers using IQR method...")
            numeric_df = df.select_dtypes(include=[np.number])
            if numeric_df.empty:
                return df
            Q1 = numeric_df.quantile(0.25)
            Q3 = numeric_df.quantile(0.75)
            IQR = Q3 - Q1
            mask = ~((numeric_df < (Q1 - threshold * IQR)) | (numeric_df > (Q3 + threshold * IQR))).any(axis=1)
            return df.loc[mask]
        return df

# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def main(input_file_path: str = "./inputs/ETTh1.csv"):
    # Initialize extractor
    api_key = ""  # Replace with your key
    extractor = MetadataExtractor(api_key)

    # Extract metadata
    print("Extracting metadata...")
    metadata = extractor.extract_from_file(input_file_path)

    # Print summary
    extractor.print_summary(metadata)

    # Get preprocessing pipeline
    preprocessing_methods = extractor.get_preprocessing_pipeline(metadata)
    print(f"\nPreprocessing methods to apply: {preprocessing_methods}")

    # Load actual data
    df = pd.read_csv(input_file_path)
    print(f"\nOriginal data shape: {df.shape}")

    # Apply preprocessing automatically
    pipeline = PreprocessingPipeline()
    df_processed = pipeline.apply_from_metadata(df, metadata)
    print(f"Processed data shape: {df_processed.shape}")

    # Save results to Pipeline/outputs directory
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    outputs_dir = script_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # Generate output filenames based on input file
    input_filename = Path(input_file_path).stem  # Get filename without extension
    metadata_output_path = outputs_dir / f"{input_filename}_metadata.json"
    processed_data_path = outputs_dir / f"{input_filename}_processed.csv"

    output = {
        'metadata': asdict(metadata),
        'preprocessing_applied': preprocessing_methods,
        'original_shape': df.shape,
        'processed_shape': df_processed.shape,
        'input_file': input_file_path
    }

    with open(metadata_output_path, 'w') as f:
        f.write(json.dumps(output, indent=2, default=str))

    with open(processed_data_path, 'w') as f:
        f.write(df_processed.to_csv(index=False))

    print("\n✓ Metadata extraction and preprocessing complete!")
    print(f"✓ Metadata + pipeline summary saved to {metadata_output_path}")
    print(f"✓ Cleaned/preprocessed data saved to {processed_data_path}")

    return metadata, df_processed

if __name__ == "__main__":
    main()
