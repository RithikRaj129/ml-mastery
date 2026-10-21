# model_card.py - Generate standardised model documentation
import json
import numpy as np
import pandas as pd
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
from sklearn.datasets import load_breast_cancer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                               precision_score, recall_score, confusion_matrix)

@dataclass
class ModelCard:
    # Model overview
    model_name:          str
    model_version:       str
    model_type:          str
    created_date:        str
    created_by:          str
    description:         str

    # Intended use
    primary_use_cases:   List[str]
    out_of_scope_uses:   List[str]
    intended_users:      List[str]

    # Training data
    training_data:       str
    training_size:       int
    feature_names:       List[str]
    target_description:  str
    data_collection_date:str
    data_preprocessing:  List[str]

    # Evaluation
    evaluation_dataset:  str
    eval_metrics:        Dict[str, float]
    eval_by_subgroup:    Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Limitations and risks
    known_limitations:   List[str] = field(default_factory=list)
    known_biases:        List[str] = field(default_factory=list)
    out_of_distribution: List[str] = field(default_factory=list)

    # Ethical considerations
    fairness_analysis:   str = ''
    privacy_measures:    List[str] = field(default_factory=list)

    # Deployment
    deployment_context:  str = ''
    monitoring_plan:     str = ''
    retraining_schedule: str = ''

    # Approval
    approved_by:         str = ''
    approval_date:       str = ''
    review_date:         str = ''

    def to_markdown(self) -> str:
        sections = [
            f'# Model Card: {self.model_name} v{self.model_version}',
            f'**Created:** {self.created_date}  |  **Author:** {self.created_by}',
            f'**Approved by:** {self.approved_by}  |  **Review date:** {self.review_date}',
            '',
            '## Model Overview',
            f'**Type:** {self.model_type}',
            f'**Description:** {self.description}',
            '',
            '## Intended Use',
            '**Primary use cases:**',
            *[f'- {u}' for u in self.primary_use_cases],
            '**Out-of-scope uses:**',
            *[f'- {u}' for u in self.out_of_scope_uses],
            '',
            '## Training Data',
            f'**Source:** {self.training_data}',
            f'**Size:** {self.training_size:,} samples',
            f'**Target:** {self.target_description}',
            '**Preprocessing steps:**',
            *[f'- {s}' for s in self.data_preprocessing],
            '',
            '## Performance',
            '**Evaluation dataset:** ' + self.evaluation_dataset,
            *[f'- **{k}:** {v:.4f}' for k, v in self.eval_metrics.items()],
            '',
            '## Limitations and Risks',
            *[f'-   {l}' for l in self.known_limitations],
            '',
            '## Ethical Considerations',
            self.fairness_analysis,
            '**Privacy measures:**',
            *[f'- {p}' for p in self.privacy_measures],
            '',
            '## Deployment and Monitoring',
            f'**Context:** {self.deployment_context}',
            f'**Monitoring:** {self.monitoring_plan}',
            f'**Retraining:** {self.retraining_schedule}',
        ]
        return '\n'.join(sections)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


# ── Build model and create model card ────────────────────────────
data = load_breast_cancer()
X, y = data.data, data.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

model = Pipeline([('sc',StandardScaler()),
                   ('gb',GradientBoostingClassifier(n_estimators=200,random_state=42))])
model.fit(X_tr, y_tr)

y_pred = model.predict(X_te)
y_prob = model.predict_proba(X_te)[:,1]

card = ModelCard(
    model_name          = 'Breast Cancer Malignancy Classifier',
    model_version       = '3.2.1',
    model_type          = 'Gradient Boosting Classifier (sklearn Pipeline)',
    created_date        = datetime.now().strftime('%Y-%m-%d'),
    created_by          = 'ML Engineering Team',
    description         = ('Binary classifier predicting breast tumour malignancy from '
                           'fine needle aspirate (FNA) cell nucleus measurements.'),

    primary_use_cases   = [
        'Supporting radiologist review of FNA biopsy results',
        'Prioritising follow-up for high-risk samples',
        'Research and quality assurance workflows',
    ],
    out_of_scope_uses   = [
        'Autonomous diagnostic decision without clinician review',
        'Tumour type classification beyond malignant/benign',
        'Prognosis or treatment recommendation',
        'Populations with different FNA measurement protocols',
    ],
    intended_users      = ['Radiologists','Pathologists','Clinical research teams'],

    training_data       = 'UCI Breast Cancer Wisconsin (Diagnostic) Dataset',
    training_size       = len(X_tr),
    feature_names       = list(data.feature_names),
    target_description  = '0=Malignant, 1=Benign  (class balance: 37% malignant)',
    data_collection_date= '1992-2019',
    data_preprocessing  = [
        'StandardScaler: zero mean, unit variance per feature',
        'No missing values in source data',
        'Stratified train/test split (80/20)',
    ],

    evaluation_dataset  = 'Hold-out test set (20%, n=114, stratified)',
    eval_metrics        = {
        'accuracy':  float(accuracy_score(y_te, y_pred)),
        'roc_auc':   float(roc_auc_score(y_te, y_prob)),
        'f1_score':  float(f1_score(y_te, y_pred)),
        'precision': float(precision_score(y_te, y_pred)),
        'recall':    float(recall_score(y_te, y_pred)),
    },

    known_limitations   = [
        'Trained on data from a single institution - may not generalise to all FNA protocols',
        'No temporal validation - performance may degrade as equipment/techniques evolve',
        'Model cannot process missing FNA measurements',
        'Calibrated for class balance of 37% malignant - adjust threshold for other settings',
    ],
    known_biases        = [
        'Training data demographics not disclosed - potential selection bias',
        'No analysis of performance by patient age, race, or hospital type available',
    ],

    fairness_analysis   = ('Fairness analysis could not be completed: training data lacks '
                           'demographic labels. Recommend prospective collection of subgroup data.'),
    privacy_measures    = [
        'Model trained on de-identified data only',
        'Input features are morphological measurements, not identifiers',
        'Predictions logged without patient identifiers',
        'GDPR Article 22 compliance: all predictions reviewed by clinician',
    ],

    deployment_context  = 'Hospital radiology information system (RIS) integration',
    monitoring_plan     = 'Weekly PSI drift check; monthly AUC evaluation on labelled subset; immediate alert if AUC < 0.92',
    retraining_schedule = 'Quarterly or on drift alert (PSI > 0.25 on any feature)',

    approved_by         = 'Dr. Sarah Chen, Chief Medical Officer',
    approval_date       = datetime.now().strftime('%Y-%m-%d'),
    review_date         = '2025-06-01',
)

print('=== Model Card Generated ===')
print(f'Model: {card.model_name} v{card.model_version}')
print(f'AUC:   {card.eval_metrics["roc_auc"]:.4f}')
print(f'F1:    {card.eval_metrics["f1_score"]:.4f}')
print(f'\nModel card sections: overview, intended use, training data,')
print(f'performance, limitations, ethics, deployment')

# Save as JSON
with open('model_card_v321.json', 'w') as f:
    f.write(card.to_json())
print(f'Saved: model_card_v321.json')

# Print Markdown preview
print('\nMarkdown preview (first 500 chars):')
print(card.to_markdown()[:500] + '...')
