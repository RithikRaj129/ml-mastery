# mlops_lifecycle.py - MLOps maturity assessment tool
from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class MLOpsCapability:
    name: str
    current_score: int   # 0-3: 0=none, 1=partial, 2=mostly, 3=fully
    target_score:  int
    description:   str
    gap_actions:   List[str] = field(default_factory=list)


def assess_mlops_maturity(capabilities: List[MLOpsCapability]) -> Dict:
    total_current = sum(c.current_score for c in capabilities)
    total_max     = len(capabilities) * 3
    maturity_pct  = total_current / total_max * 100

    gaps = [c for c in capabilities if c.current_score < c.target_score]
    gaps.sort(key=lambda c: c.target_score - c.current_score, reverse=True)

    level = 0
    if maturity_pct >= 25:  level = 1
    if maturity_pct >= 50:  level = 2
    if maturity_pct >= 75:  level = 3

    return {
        'maturity_pct':   round(maturity_pct, 1),
        'level':          level,
        'level_name':     ['Manual ML','ML Pipeline Automation','CI/CD for ML','Continuous Training'][level],
        'top_gaps':       gaps[:5],
        'total_current':  total_current,
        'total_max':      total_max,
    }


# Assess a hypothetical team
capabilities = [
    MLOpsCapability('Data versioning',      1, 3, 'DVC or Delta Lake for dataset versioning',
                    ['Set up DVC', 'Version training datasets', 'Track data lineage']),
    MLOpsCapability('Experiment tracking',  2, 3, 'MLflow or W&B for all experiments',
                    ['Log all experiments to MLflow', 'Define standard metrics']),
    MLOpsCapability('Automated training',   1, 3, 'Pipeline runs without manual steps',
                    ['Build Airflow DAG for training', 'Automate feature extraction']),
    MLOpsCapability('Model registry',       2, 3, 'Centralised versioned model store',
                    ['Set up MLflow Registry', 'Enforce staging workflow']),
    MLOpsCapability('Automated deployment', 1, 3, 'CI/CD pipeline deploys approved models',
                    ['Write GitHub Actions pipeline', 'Add quality gate checks']),
    MLOpsCapability('Data drift monitoring',0, 3, 'Automated drift detection and alerting',
                    ['Implement PSI checks', 'Set up Evidently reports', 'Configure alerts']),
    MLOpsCapability('Model monitoring',     0, 3, 'Track prediction distribution and accuracy',
                    ['Log all predictions', 'Compute performance on labelled subset']),
    MLOpsCapability('Automated retraining', 0, 3, 'Models retrain on drift trigger',
                    ['Define retraining triggers', 'Build champion-challenger']),
    MLOpsCapability('Model governance',     1, 3, 'Model cards, audit trails, approvals',
                    ['Create model card template', 'Log approval decisions']),
    MLOpsCapability('Alerting',             1, 2, 'Automated alerts for model and system issues',
                    ['Set up Slack alerts', 'Define SLAs and thresholds']),
]

result = assess_mlops_maturity(capabilities)

print('=== MLOps Maturity Assessment ===')
print(f'Score:   {result["total_current"]}/{result["total_max"]}  ({result["maturity_pct"]}%)')
print(f'Level:   {result["level"]} - {result["level_name"]}')
print(f'\nTop 5 Gaps to Address:')
for i, cap in enumerate(result['top_gaps'], 1):
    gap = cap.target_score - cap.current_score
    print(f'\n  {i}. {cap.name} (current={cap.current_score}/3, target={cap.target_score}/3, gap={gap})')
    for action in cap.gap_actions[:2]:
        print(f'     → {action}')
