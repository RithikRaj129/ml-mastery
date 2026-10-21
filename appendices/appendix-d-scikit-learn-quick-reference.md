# Appendix D — scikit-learn Quick Reference

*Machine Learning Mastery: Mathematics to MLOps* · quick-reference code from the book.

These are cheat sheets, not runnable scripts — several lines show alternatives side by side.

```python
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
    mean_squared_error, r2_score, confusion_matrix, classification_report)

# ── Data Splitting ────────────────────────────────────────────────
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# ── Preprocessing ─────────────────────────────────────────────────
scaler = StandardScaler().fit(X_tr)   # ALWAYS fit on train only
X_tr_sc = scaler.transform(X_tr)
X_te_sc = scaler.transform(X_te)      # Apply same transform to test

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median').fit(X_tr)

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)

# ── Pipelines ─────────────────────────────────────────────────────
pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler()),
    ('model',   GradientBoostingClassifier(n_estimators=100))
])
pipe.fit(X_tr, y_tr)      # ALL steps fitted together
pipe.predict(X_te)         # Transforms X_te automatically
pipe.score(X_te, y_te)

# ── Classification Algorithms ──────────────────────────────────────
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier,
    GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# ── Regression Algorithms ─────────────────────────────────────────
from sklearn.linear_model import (LinearRegression, Ridge, Lasso,
    ElasticNet, SGDRegressor)
from sklearn.ensemble import (RandomForestRegressor,
    GradientBoostingRegressor)
from sklearn.svm import SVR

# ── Unsupervised ──────────────────────────────────────────────────
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, TruncatedSVD, NMF
from sklearn.manifold import TSNE

# ── Cross-Validation ──────────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
print(f'{scores.mean():.4f} ± {scores.std():.4f}')

from sklearn.model_selection import cross_validate
results = cross_validate(model, X, y, cv=cv,
    scoring=['accuracy','roc_auc','f1'], return_train_score=True)

# ── Hyperparameter Tuning ─────────────────────────────────────────
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
gs = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
gs.fit(X_tr, y_tr)
print(gs.best_params_, gs.best_score_)

# ── Classification Metrics ────────────────────────────────────────
print(classification_report(y_te, y_pred, target_names=['neg','pos']))
print(confusion_matrix(y_te, y_pred))
print(f'AUC: {roc_auc_score(y_te, y_prob):.4f}')

from sklearn.metrics import roc_curve, precision_recall_curve
fpr, tpr, thresholds = roc_curve(y_te, y_prob)

# ── Regression Metrics ────────────────────────────────────────────
from sklearn.metrics import mean_absolute_error, mean_squared_error
mae  = mean_absolute_error(y_te, y_pred)
mse  = mean_squared_error(y_te, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_te, y_pred)

# ── Feature Importance ────────────────────────────────────────────
importances = model.feature_importances_   # Tree-based
coefs = model.coef_                        # Linear models
from sklearn.inspection import permutation_importance
result = permutation_importance(model, X_te, y_te, n_repeats=30)
```
