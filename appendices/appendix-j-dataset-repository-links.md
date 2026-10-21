# Appendix J — Dataset Repository Links

*Machine Learning Mastery: Mathematics to MLOps* · quick-reference code from the book.

These are cheat sheets, not runnable scripts — several lines show alternatives side by side.

```python
# scikit-learn built-in datasets
from sklearn.datasets import (
    load_iris,           # 150 samples, 4 features, 3 classes
    load_breast_cancer,  # 569 samples, 30 features, binary
    load_wine,           # 178 samples, 13 features, 3 classes
    load_digits,         # 1797 samples, 64 features, 10 classes
    load_boston,         # Housing regression (deprecated)
    fetch_california_housing,  # 20640 samples, 8 features, regression
    make_classification, make_regression, make_blobs,
    make_moons, make_circles   # Synthetic datasets
)

# Seaborn datasets (CSV download from GitHub)
import seaborn as sns
tips     = sns.load_dataset('tips')      # Restaurant tips
titanic  = sns.load_dataset('titanic')   # Titanic survivors
flights  = sns.load_dataset('flights')   # Monthly airline passengers
penguins = sns.load_dataset('penguins')  # Palmer penguins
mpg      = sns.load_dataset('mpg')       # Car fuel efficiency

# Statsmodels datasets
import statsmodels.api as sm
air_pass = sm.datasets.get_rdataset('AirPassengers').data
sunspot  = sm.datasets.sunspots.load_pandas().data
```
