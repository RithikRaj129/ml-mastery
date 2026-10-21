# Machine Learning Mastery: Mathematics to MLOps

Official companion code repository for the book **_Machine Learning Mastery: Mathematics to MLOps_** by **Rithik Raj Vaishya**.

<!-- Add the book cover image here once ready: -->
<!-- <img src="assets/cover.png" alt="Book cover" width="250"> -->

Every chapter of the book contains working, runnable Python code with actual output shown. This repository contains all of that code, organised chapter by chapter, so you can run, modify and experiment with every example as you read.

---

## 📖 About the Book

*Machine Learning Mastery* is a complete, end-to-end guide to machine learning for practitioners. It is a structured journey from the mathematics that makes the algorithms work, through the Python tools and the algorithms themselves, all the way to the deployment and monitoring infrastructure that makes models useful in the real world.

The book has **15 chapters in 6 parts**, plus 12 appendices:

| Part | Theme |
|------|-------|
| **I — Foundations of Machine Learning** | What ML is, the mathematics behind it, Python tools, data preprocessing & EDA |
| **II — Supervised Learning** | Regression and classification — algorithms, metrics and practical decisions |
| **III — Unsupervised Learning** | Clustering and dimensionality reduction |
| **IV — Advanced Machine Learning** | Ensembles, model optimization and time series forecasting |
| **V — ML Engineering & Deployment** | Serving, containers, CI/CD, pipelines, feature stores, MLflow, drift monitoring |
| **VI — Real-World AI Systems** | Responsible AI, fairness, privacy, emerging paradigms and a career roadmap |

**Who it's for:** developers and analysts moving into ML, working data scientists filling gaps, and learners preparing for industry. You only need basic Python — no prior ML, statistics or advanced mathematics is assumed.

**Get the book:**
- 📕 Paperback: _link coming soon_
- 📱 Kindle eBook: _link coming soon_

ISBN (Paperback): _add your ISBN here_

---

## 📂 Chapter Guide

### Part I — Foundations of Machine Learning

| Ch | Title | What's covered | Code |
|---:|-------|----------------|------|
| 1 | Introduction to Machine Learning | ML paradigms, the ML workflow, tools & libraries, environment setup | [chapter-01](./chapter-01-introduction-to-ml) |
| 2 | Mathematics for Machine Learning | Linear algebra, eigenvalues, calculus, gradients, probability, Bayes, statistics, hypothesis testing, cost functions, gradient descent | [chapter-02](./chapter-02-mathematics-for-ml) |
| 3 | Python for Machine Learning | NumPy, Pandas, Matplotlib, Seaborn | [chapter-03](./chapter-03-python-for-ml) |
| 4 | Data Preprocessing & EDA | Missing values, outliers, encoding, scaling, feature engineering, EDA, data splitting, preventing leakage | [chapter-04](./chapter-04-preprocessing-and-eda) |

### Part II — Supervised Learning

| Ch | Title | What's covered | Code |
|---:|-------|----------------|------|
| 5 | Linear Regression | OLS, polynomial, Ridge (L2), Lasso (L1), assumptions, MAE / RMSE / R² | [chapter-05](./chapter-05-linear-regression) |
| 6 | Classification Algorithms | Logistic regression, decision trees, KNN, Naive Bayes, SVM, confusion matrix, F1, AUC-ROC, class imbalance | [chapter-06](./chapter-06-classification) |

### Part III — Unsupervised Learning

| Ch | Title | What's covered | Code |
|---:|-------|----------------|------|
| 7 | Clustering Techniques | K-Means, DBSCAN, hierarchical clustering, GMMs, silhouette score, elbow method | [chapter-07](./chapter-07-clustering) |
| 8 | Dimensionality Reduction | PCA, LDA, t-SNE, UMAP | [chapter-08](./chapter-08-dimensionality-reduction) |

### Part IV — Advanced Machine Learning

| Ch | Title | What's covered | Code |
|---:|-------|----------------|------|
| 9 | Ensemble Learning | Bagging & Random Forests, AdaBoost, XGBoost, LightGBM, CatBoost, stacking, voting | [chapter-09](./chapter-09-ensemble-learning) |
| 10 | Model Optimization | Bias-variance, cross-validation, regularization, Grid / Random / Bayesian (Optuna) tuning, learning curves, SHAP & LIME, early stopping | [chapter-10](./chapter-10-model-optimization) |
| 11 | Time Series Machine Learning | Stationarity, ACF / PACF, ARIMA & SARIMA, Holt-Winters, Prophet, time series features, walk-forward validation | [chapter-11](./chapter-11-time-series) |

### Part V — ML Engineering & Deployment

| Ch | Title | What's covered | Code |
|---:|-------|----------------|------|
| 12 | Model Deployment | Serialisation, versioning, FastAPI & Flask APIs, Streamlit, Docker, GitHub Actions CI/CD, SageMaker / Vertex AI / Azure ML, ONNX | [chapter-12](./chapter-12-model-deployment) |
| 13 | ML Systems Design | Data pipelines, feature stores, real-time vs batch inference, Kafka, Airflow, Spark, distributed training, GPU vs TPU, vector databases, RAG | [chapter-13](./chapter-13-ml-systems-design) |
| 14 | MLOps & Monitoring | MLflow tracking, model registry, data drift (KS test, PSI), concept drift, champion-challenger, automated retraining, alerting, model cards | [chapter-14](./chapter-14-mlops-and-monitoring) |

### Part VI — Real-World AI Systems

| Ch | Title | What's covered | Code |
|---:|-------|----------------|------|
| 15 | Responsible AI & Future Trends | XAI, fairness & bias, differential privacy, federated learning, AutoML, RL basics, self-supervised learning, AI safety, agentic AI, career roadmap | [chapter-15](./chapter-15-responsible-ai) |

### Appendices

| Appendix | Title |
|:--------:|-------|
| A | Python Quick Reference |
| B | NumPy Quick Reference |
| C | Pandas Quick Reference |
| D | scikit-learn Quick Reference |
| E | PyTorch Essentials |
| F | TensorFlow / Keras Essentials |
| G | Essential ML Formulas |
| H | 100 ML Interview Questions |
| I | 50 ML Project Ideas |
| J | Dataset Repository Links |
| K | Essential Papers & Books in ML |
| L | Glossary of Key ML Terms |

Code for the PyTorch and TensorFlow / Keras appendices is in [`appendices/`](./appendices).

---

## 🗂️ Repository Structure

```
ml-mastery/
├── README.md
├── requirements.txt
├── chapter-01-introduction-to-ml/
├── chapter-02-mathematics-for-ml/
├── chapter-03-python-for-ml/
├── chapter-04-preprocessing-and-eda/
├── chapter-05-linear-regression/
├── chapter-06-classification/
├── chapter-07-clustering/
├── chapter-08-dimensionality-reduction/
├── chapter-09-ensemble-learning/
├── chapter-10-model-optimization/
├── chapter-11-time-series/
├── chapter-12-model-deployment/
├── chapter-13-ml-systems-design/
├── chapter-14-mlops-and-monitoring/
├── chapter-15-responsible-ai/
├── appendices/
└── datasets/
```

Inside each chapter folder, notebooks are numbered by section to match the book. For example, `chapter-05-linear-regression/5.3_ridge_regression.ipynb` is the code for Section 5.3.

---

## ⚙️ Getting Started

You can run every example either in **Google Colab** (no setup) or **locally**. Full instructions are in **Chapter 1, Section 1.9** of the book.

### Option 1: Google Colab (easiest)

Open any notebook in this repo on GitHub and change `github.com` to `githubtocolab.com` in the URL, or upload it at [colab.research.google.com](https://colab.research.google.com). Install anything Colab is missing in a cell:

```python
!pip install lightgbm catboost optuna shap
```

### Option 2: Run locally

**1. Clone the repository**
```bash
git clone https://github.com/RithikRaj129/ml-mastery.git
cd ml-mastery
```

**2. Create a virtual environment**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Launch Jupyter**
```bash
jupyter lab
```

**Requirements:** Python **3.11 or later**. All code is tested against the library versions listed in Chapter 1 and pinned in [`requirements.txt`](./requirements.txt).

### Core libraries used

| Library | Version | Used for |
|---------|---------|----------|
| NumPy | 1.26.x | Numerical computing, linear algebra |
| Pandas | 2.2.x | Data manipulation |
| Matplotlib / Seaborn | 3.8.x / 0.13.x | Visualisation |
| scikit-learn | 1.4.x | Classical ML |
| XGBoost / LightGBM / CatBoost | 2.0.x / 4.x / 1.2.x | Gradient boosting |
| TensorFlow / PyTorch | 2.16.x / 2.3.x | Deep learning |
| Hugging Face Transformers | 4.x | Pre-trained NLP models |
| MLflow | 2.x | Experiment tracking & model registry |
| Optuna | 3.x | Hyperparameter tuning |
| SHAP | 0.45.x | Model explainability |

Some chapters need extra tools (for example Docker for Chapter 12, Kafka / Airflow / Spark for Chapter 13). Each of those chapter folders has its own `README.md` with setup steps.

---

## 🐞 Found an Error?

If you find a bug in the code or an error in the book, please [open an issue](https://github.com/RithikRaj129/ml-mastery/issues) and include the **chapter, section and page number**. Confirmed corrections are listed below.

### Errata
_No corrections yet._

---

## 📜 License

The **code** in this repository is released under the [MIT License](LICENSE). You are free to use it in your own projects.

The **book text, figures and content** are © 2026 Rithik Raj Vaishya. All rights reserved.

---

## 👤 About the Author

**Rithik Raj Vaishya** is a data science practitioner, corporate trainer and founder of **RV TechLearn LLP**, an IT training and EdTech company. He is a doctoral candidate (DBA) in Leadership & AI Strategy at ESGCI, Paris.

- 💼 LinkedIn: _https://www.linkedin.com/in/rithikrajvaishya/_

---

⭐ If this repository helps you, please give it a star!
