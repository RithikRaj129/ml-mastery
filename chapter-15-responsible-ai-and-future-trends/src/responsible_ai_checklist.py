# responsible_ai_checklist.py - Practical RAI assessment tool
from dataclasses import dataclass, field
from typing import List, Dict
from enum import Enum

class RiskLevel(Enum):
    LOW = 'LOW'
    MEDIUM = 'MEDIUM'
    HIGH = 'HIGH'
    CRITICAL = 'CRITICAL'

@dataclass
class RAIChecklistItem:
    category:    str
    question:    str
    answered:    bool = False
    answer:      str  = ''
    risk_if_no:  RiskLevel = RiskLevel.MEDIUM


RESPONSIBLE_AI_CHECKLIST = [
    # Fairness
    RAIChecklistItem('Fairness', 'Have you defined the protected attributes relevant to your use case?',
risk_if_no=RiskLevel.HIGH),
    RAIChecklistItem('Fairness', 'Have you measured model performance across demographic subgroups?',
risk_if_no=RiskLevel.HIGH),
    RAIChecklistItem('Fairness', 'Is the disparity in false positive rate across groups acceptable?',
risk_if_no=RiskLevel.CRITICAL),
    RAIChecklistItem('Fairness', 'Is training data representative of all groups the model will serve?',
risk_if_no=RiskLevel.HIGH),

    # Transparency
    RAIChecklistItem('Transparency', 'Does a Model Card exist and is it publicly accessible?',
risk_if_no=RiskLevel.MEDIUM),
    RAIChecklistItem('Transparency', 'Are all stakeholders informed that AI is being used?', risk_if_no=RiskLevel.HIGH),
    RAIChecklistItem('Transparency', 'Is there a public disclosure of what data was used for training?',
risk_if_no=RiskLevel.MEDIUM),

    # Explainability
    RAIChecklistItem('Explainability', 'Can individual predictions be explained to affected parties?',
risk_if_no=RiskLevel.HIGH),
    RAIChecklistItem('Explainability', 'Is there a right-to-explanation process (GDPR Art. 22)?',
risk_if_no=RiskLevel.CRITICAL),
    RAIChecklistItem('Explainability', 'Can the model explanation be understood by a non-technical stakeholder?',
risk_if_no=RiskLevel.MEDIUM),

    # Privacy
    RAIChecklistItem('Privacy', 'Is data collection minimised to what is necessary?', risk_if_no=RiskLevel.HIGH),
    RAIChecklistItem('Privacy', 'Is there explicit consent for automated decision-making?',
risk_if_no=RiskLevel.CRITICAL),
    RAIChecklistItem('Privacy', 'Has the model been tested for membership inference attacks?',
risk_if_no=RiskLevel.MEDIUM),
    RAIChecklistItem('Privacy', 'Are PII fields excluded from model inputs or anonymised?', risk_if_no=RiskLevel.HIGH),

    # Accountability
    RAIChecklistItem('Accountability', 'Is there a named responsible party for this model in production?',
risk_if_no=RiskLevel.HIGH),
    RAIChecklistItem('Accountability', 'Is there a human review process for high-stakes decisions?',
risk_if_no=RiskLevel.CRITICAL),
    RAIChecklistItem('Accountability', 'Is there an incident response plan for model failures?', risk_if_no=RiskLevel.HIGH),
    RAIChecklistItem('Accountability', 'Is there a defined process for affected parties to contest decisions?',
risk_if_no=RiskLevel.HIGH),

    # Safety
    RAIChecklistItem('Safety', 'Has the model been red-teamed for adversarial inputs?', risk_if_no=RiskLevel.HIGH),
    RAIChecklistItem('Safety', 'Is there a kill switch / fallback if the model behaves unexpectedly?',
risk_if_no=RiskLevel.CRITICAL),
    RAIChecklistItem('Safety', 'Has the model been tested on out-of-distribution inputs?', risk_if_no=RiskLevel.MEDIUM),
]


def run_rai_assessment(checklist: list, use_case: str, answers: Dict[str, bool]) -> Dict:
    '''Score a use case against the RAI checklist.'''
    results_by_category = {}
    critical_gaps = []
    high_gaps     = []

    for item in checklist:
        answered = answers.get(item.question, False)
        if item.category not in results_by_category:
            results_by_category[item.category] = {'passed':0,'total':0,'gaps':[]}
        results_by_category[item.category]['total'] += 1
        if answered:
            results_by_category[item.category]['passed'] += 1
        else:
            results_by_category[item.category]['gaps'].append(item.question)
            if item.risk_if_no == RiskLevel.CRITICAL: critical_gaps.append(item.question)
            elif item.risk_if_no == RiskLevel.HIGH:   high_gaps.append(item.question)

    total_passed = sum(v['passed'] for v in results_by_category.values())
    total_items  = len(checklist)

    return {
        'use_case':        use_case,
        'score_pct':       round(total_passed/total_items*100, 1),
        'passed':          total_passed,
        'total':           total_items,
        'critical_gaps':   critical_gaps,
        'high_gaps':       high_gaps,
        'by_category':     results_by_category,
        'ready_for_prod':  len(critical_gaps) == 0,
    }


# Assess a loan approval model
answers_loan_model = {
    # Fairness - partially addressed
    'Have you defined the protected attributes relevant to your use case?': True,
    'Have you measured model performance across demographic subgroups?': True,
    'Is the disparity in false positive rate across groups acceptable?': False,  # CRITICAL gap
    'Is training data representative of all groups the model will serve?': False,

    # Transparency
    'Does a Model Card exist and is it publicly accessible?': True,
    'Are all stakeholders informed that AI is being used?': True,
    'Is there a public disclosure of what data was used for training?': False,

    # Explainability
    'Can individual predictions be explained to affected parties?': True,
    'Is there a right-to-explanation process (GDPR Art. 22)?': False,  # CRITICAL gap
    'Can the model explanation be understood by a non-technical stakeholder?': True,

    # Privacy
    'Is data collection minimised to what is necessary?': True,
    'Is there explicit consent for automated decision-making?': False,  # CRITICAL gap
    'Has the model been tested for membership inference attacks?': False,
    'Are PII fields excluded from model inputs or anonymised?': True,

    # Accountability
    'Is there a named responsible party for this model in production?': True,
    'Is there a human review process for high-stakes decisions?': True,
    'Is there an incident response plan for model failures?': False,
    'Is there a defined process for affected parties to contest decisions?': False,

    # Safety
    'Has the model been red-teamed for adversarial inputs?': False,
    'Is there a kill switch / fallback if the model behaves unexpectedly?': True,
    'Has the model been tested on out-of-distribution inputs?': False,
}

result = run_rai_assessment(RESPONSIBLE_AI_CHECKLIST, 'Consumer Loan Approval Model', answers_loan_model)

print('=== Responsible AI Assessment: Consumer Loan Approval Model ===')
print(f'Overall Score: {result["passed"]}/{result["total"]}  ({result["score_pct"]}%)')
print(f'Ready for Production: {result["ready_for_prod"]} ({len(result["critical_gaps"])} critical gaps)')

print(f'\nScore by Category:')
for cat, stats in result['by_category'].items():
    pct = stats['passed']/stats['total']*100
    bar = '█'*int(pct//10) + '░'*(10-int(pct//10))
    print(f'  {cat:<15} {bar} {stats["passed"]}/{stats["total"]} ({pct:.0f}%)')

print(f'\n  Critical Gaps (MUST fix before deployment):')
for gap in result['critical_gaps']:
    print(f'  • {gap}')

print(f'\n  High Priority Gaps:')
for gap in result['high_gaps'][:3]:
    print(f'  • {gap}')
