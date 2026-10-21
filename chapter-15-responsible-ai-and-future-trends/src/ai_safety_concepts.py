# ai_safety_concepts.py - Practical near-term safety concepts
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# ── 1. Adversarial Robustness ─────────────────────────────────────
data = load_breast_cancer()
X_sc = StandardScaler().fit_transform(data.data)
X_tr, X_te, y_tr, y_te = train_test_split(X_sc, data.target, test_size=0.2, random_state=42)

model = GradientBoostingClassifier(n_estimators=100, random_state=42).fit(X_tr, y_tr)
print(f'=== AI Safety: Robustness Testing ===')
print(f'Clean AUC: {roc_auc_score(y_te, model.predict_proba(X_te)[:,1]):.4f}')

# FGSM-style perturbation (gradient sign attack for tabular data)
def tabular_attack(model, X, y, epsilon=0.1):
    '''Add noise in direction that increases loss.'''
    X_adv = X.copy()
    probs  = model.predict_proba(X)[:,1]
    errors = probs - y  # Gradient of log-loss

    for j in range(X.shape[1]):
        # Estimate gradient via finite differences
        X_plus = X.copy(); X_plus[:,j] += 1e-5
        grad_j = (model.predict_proba(X_plus)[:,1] - probs) / 1e-5
        # Perturb in direction of gradient
        X_adv[:,j] += epsilon * np.sign(grad_j * errors)

    return X_adv

for eps in [0.01, 0.05, 0.10, 0.20, 0.50]:
    X_adv = tabular_attack(model, X_te, y_te, epsilon=eps)
    adv_auc = roc_auc_score(y_te, model.predict_proba(X_adv)[:,1])
    drop = roc_auc_score(y_te, model.predict_proba(X_te)[:,1]) - adv_auc
    print(f'  ε={eps:.2f}: adversarial AUC={adv_auc:.4f}  drop={drop:.4f}')

# ── 2. Uncertainty Quantification ────────────────────────────────
print('\n=== Uncertainty Quantification ===')
print('Models should know what they do not know.')

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_tr, y_tr)

# Prediction variance across trees as proxy for uncertainty
tree_preds = np.array([t.predict_proba(X_te)[:,1] for t in rf.estimators_])
pred_mean  = tree_preds.mean(axis=0)
pred_std   = tree_preds.std(axis=0)   # Epistemic uncertainty

# High uncertainty = model is confused
high_uncertainty_idx = np.argsort(pred_std)[::-1][:10]
low_uncertainty_idx  = np.argsort(pred_std)[:10]

print(f'High uncertainty samples (avg std={pred_std[high_uncertainty_idx].mean():.4f}):')
print(f'  Accuracy on these: {(rf.predict(X_te[high_uncertainty_idx])==y_te[high_uncertainty_idx]).mean():.2%}')
print(f'Low uncertainty samples (avg std={pred_std[low_uncertainty_idx].mean():.4f}):')
print(f'  Accuracy on these: {(rf.predict(X_te[low_uncertainty_idx])==y_te[low_uncertainty_idx]).mean():.2%}')
print('Insight: route high-uncertainty cases to human review!')

# ── 3. Near-term safety checklist ────────────────────────────────
print('\n=== Near-Term ML Safety Checklist ===')
safety_items = [
    ('Red-team adversarial inputs',       'Test model with edge cases, adversarial examples, and deliberate misuse scenarios'),
    ('OOD detection',                    'Flag inputs that are far from training distribution as unreliable'),
    ('Uncertainty quantification',        'Provide calibrated confidence; refuse to predict when uncertain'),
    ('Human-in-the-loop',                'High-stakes decisions require human review; never fully automate'),
    ('Output filtering',                 'Post-process model outputs to prevent harmful content (generation models)'),
    ('Rollback capability',              'Immediately revert to previous model if safety issues detected'),
    ('Incident response plan',           'What do you do if the model causes harm? Who is responsible?'),
    ('Access controls',                  'Limit who can query the model; rate limiting; API key management'),
]
for name, desc in safety_items:
    print(f'  ✓ {name:<35}: {desc}')
