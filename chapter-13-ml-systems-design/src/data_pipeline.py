# data_pipeline.py - Production-ready ML data pipeline patterns
import pandas as pd
import numpy as np
from typing import Optional, Callable, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import json
import hashlib

logger = logging.getLogger(__name__)

# ── Data Quality Framework ────────────────────────────────────────
@dataclass
class DataQualityRule:
    name: str
    check: Callable[[pd.DataFrame], bool]
    severity: str = 'error'   # 'error' = halt pipeline; 'warning' = continue
    description: str = ''


@dataclass
class DataQualityReport:
    passed: List[str] = field(default_factory=list)
    failed: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    n_rows_input: int = 0
    n_rows_output: int = 0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    @property
    def success(self) -> bool:
        return len(self.failed) == 0


class DataQualityValidator:
    '''Validate DataFrames against a set of quality rules.'''

    def __init__(self, rules: List[DataQualityRule]):
        self.rules = rules

    def validate(self, df: pd.DataFrame) -> DataQualityReport:
        report = DataQualityReport(n_rows_input=len(df))

        for rule in self.rules:
            try:
                passed = rule.check(df)
            except Exception as e:
                passed = False
                logger.error(f'Rule {rule.name} raised exception: {e}')

            if passed:
                report.passed.append(rule.name)
                logger.info(f'✓ {rule.name}')
            else:
                if rule.severity == 'error':
                    report.failed.append(rule.name)
                    logger.error(f'✗ {rule.name}: {rule.description}')
                else:
                    report.warnings.append(rule.name)
                    logger.warning(f'⚠ {rule.name}: {rule.description}')

        report.n_rows_output = len(df)
        return report


# ── Define quality rules ──────────────────────────────────────────
def build_quality_rules(feature_cols: List[str],
                          target_col: str,
                          max_missing_pct: float = 0.1) -> List[DataQualityRule]:
    return [
        DataQualityRule(
            name='no_empty_dataframe',
            check=lambda df: len(df) > 0,
            severity='error',
            description='DataFrame must not be empty'
        ),
        DataQualityRule(
            name='required_columns_present',
            check=lambda df: all(c in df.columns for c in feature_cols + [target_col]),
            severity='error',
            description=f'Missing required columns'
        ),
        DataQualityRule(
            name='acceptable_missing_rate',
            check=lambda df: df[feature_cols].isnull().mean().max() <= max_missing_pct,
            severity='error',
            description=f'Missing rate exceeds {max_missing_pct:.0%}'
        ),
        DataQualityRule(
            name='no_duplicate_rows',
            check=lambda df: df.duplicated().sum() / len(df) < 0.05,
            severity='warning',
            description='More than 5% duplicate rows'
        ),
        DataQualityRule(
            name='target_not_all_one_class',
            check=lambda df: df[target_col].nunique() > 1,
            severity='error',
            description='Target column has only one class - cannot train'
        ),
        DataQualityRule(
            name='no_future_timestamps',
            check=lambda df: True,  # Implement: df['timestamp'] <= datetime.now()
            severity='warning',
            description='Future timestamps detected'
        ),
    ]


# ── ML Data Pipeline ─────────────────────────────────────────────
class MLDataPipeline:
    '''
    Reproducible, validated ML data pipeline.
    '''

    def __init__(self, name: str, version: str = '1.0'):
        self.name    = name
        self.version = version
        self.steps   = []
        self.reports = []

    def add_step(self, name: str, func: Callable) -> 'MLDataPipeline':
        self.steps.append((name, func))
        return self

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f'Pipeline {self.name} v{self.version} starting: {len(df)} rows')
        current_df = df.copy()

        for step_name, step_func in self.steps:
            t0 = datetime.utcnow()
            try:
                current_df = step_func(current_df)
                elapsed = (datetime.utcnow()-t0).total_seconds()
                logger.info(f'  Step [{step_name}]: {len(current_df)} rows, {elapsed:.2f}s')
            except Exception as e:
                logger.error(f'  Step [{step_name}] FAILED: {e}')
                raise

        logger.info(f'Pipeline complete: {len(df)} → {len(current_df)} rows')
        return current_df

    def get_data_hash(self, df: pd.DataFrame) -> str:
        '''Compute hash of DataFrame for lineage tracking.'''
        content = pd.util.hash_pandas_object(df).sum()
        return hashlib.md5(str(content).encode()).hexdigest()[:8]


# ── Example pipeline ─────────────────────────────────────────────
import numpy as np
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
df_raw = pd.DataFrame(data.data, columns=data.feature_names)
df_raw['target'] = data.target

# Inject some quality issues
df_raw.iloc[:20, 0] = np.nan   # Missing values
df_raw = pd.concat([df_raw, df_raw.iloc[:5]])  # Duplicates

# Build pipeline
pipeline = MLDataPipeline('cancer_prep', version='2.1')
pipeline.add_step('remove_duplicates',  lambda df: df.drop_duplicates())
pipeline.add_step('impute_missing',     lambda df: df.fillna(df.mean(numeric_only=True)))
pipeline.add_step('clip_outliers',      lambda df: df.copy().assign(**{
    c: df[c].clip(df[c].quantile(0.01), df[c].quantile(0.99))
    for c in df.select_dtypes(include=np.number).columns if c != 'target'
}))
pipeline.add_step('add_derived_features', lambda df: df.assign(
    area_per_perimeter = df['mean area'] / (df['mean perimeter'] + 1e-6),
    compactness_ratio  = df['mean compactness'] / (df['mean smoothness'] + 1e-6)
))

# Validate
rules  = build_quality_rules(list(data.feature_names), 'target')
validator = DataQualityValidator(rules)
report    = validator.validate(df_raw)

print('=== Data Quality Report ===')
print(f'Passed: {len(report.passed)}  Failed: {len(report.failed)}  Warnings: {len(report.warnings)}')
print(f'Pipeline can proceed: {report.success}')

# Run pipeline
if report.success:
    df_processed = pipeline.run(df_raw)
    print(f'\nPipeline output: {df_processed.shape}')
    print(f'Data hash: {pipeline.get_data_hash(df_processed)}')
