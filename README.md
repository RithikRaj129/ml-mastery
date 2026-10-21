# Machine Learning Mastery: Mathematics to MLOps

Official companion code repository for the book **_Machine Learning Mastery: Mathematics to MLOps_** by **Rithik Raj Vaishya**.

<!-- Add the book cover once ready: <img src="assets/cover.png" alt="Book cover" width="250"> -->

Every chapter of the book contains working, runnable Python code. This repository has **all of it — 230 examples across 145 notebooks** — organised by chapter and section, so you can run, modify and experiment with every example as you read.

**Get the book:**
- 📕 Paperback: _link coming soon_
- 📱 Kindle eBook: _link coming soon_

ISBN (Paperback): _add your ISBN here_

---

## 📖 About the Book

*Machine Learning Mastery* is a complete, end-to-end guide to machine learning for practitioners — from the mathematics that makes the algorithms work, through the Python tools and algorithms themselves, to the deployment and monitoring infrastructure that makes models useful in the real world.

**Who it's for:** developers and analysts moving into ML, working data scientists filling gaps, and learners preparing for industry. You only need basic Python — no prior ML, statistics or advanced mathematics is assumed.

---

## 📂 Chapters

Click a chapter to see every section notebook, or open any notebook straight in Google Colab.

### Part I — Foundations of Machine Learning

| Ch | Chapter | Notebooks | Topics covered |
|---:|---|:-:|---|
| 1 | [Introduction to Machine Learning](./chapter-01-introduction-to-machine-learning/) | 2 | Tools & Libraries, Setting Up Development Environment |
| 2 | [Mathematics for Machine Learning](./chapter-02-mathematics-for-machine-learning/) | 14 | Linear Algebra Fundamentals, Matrix Operations, Vector Spaces, Eigenvalues & Eigenvectors, Calculus for ML, Gradients & Optimization, Probability Basics, Conditional Probability, Bayes Theorem, Statistics for Data Science, … |
| 3 | [Python for Machine Learning](./chapter-03-python-for-machine-learning/) | 11 | Python Basics Refresher, Data Types & Variables, Operators, Control Flow Statement, Functions & Lambda Functions, Object-Oriented Programming, File Handling, Exception Handling, NumPy Fundamentals, Pandas for Data Analysis, … |
| 4 | [Data Preprocessing & Exploratory Data Analysis](./chapter-04-data-preprocessing-and-exploratory-data-analysis/) | 14 | Understanding Datasets, Data Collection Methods, Data Cleaning Techniques, Handling Missing Values, Outlier Detection and Treatment, Feature Scaling, Encoding Categorical Variables, Exploratory Data Analysis, … |

### Part II — Supervised Learning

| Ch | Chapter | Notebooks | Topics covered |
|---:|---|:-:|---|
| 5 | [Linear Regression](./chapter-05-linear-regression/) | 10 | Simple Linear Regression, Linear Regression from Scratch, Multiple Linear Regression, Polynomial Regression, Assumptions of Regression, Gradient Descent Optimization, Regression Metrics, Bias-Variance Tradeoff, Regularization, … |
| 6 | [Classification Algorithms](./chapter-06-classification-algorithms/) | 16 | Logistic Regression, The Sigmoid Function, K-Nearest Neighbours (KNN), Naive Bayes, Support Vector Machines (SVM), Decision Trees, Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost, Evaluation Metrics, … |

### Part III — Unsupervised Learning

| Ch | Chapter | Notebooks | Topics covered |
|---:|---|:-:|---|
| 7 | [Clustering Techniques](./chapter-07-clustering-techniques/) | 8 | Introduction to Clustering, K-Means Clustering, K-Means from Scratch, Hierarchical Clustering, DBSCAN, Gaussian Mixture Models (GMM), Cluster Evaluation Metrics, Business Applications of Clustering |
| 8 | [Dimensionality Reduction](./chapter-08-dimensionality-reduction/) | 6 | The Curse of Dimensionality, Principal Component Analysis (PCA), Linear Discriminant Analysis (LDA), UMAP (Uniform Manifold Approximation and Projection), Feature Selection vs Feature Extraction, … |

### Part IV — Advanced Machine Learning

| Ch | Chapter | Notebooks | Topics covered |
|---:|---|:-:|---|
| 9 | [Ensemble Learning](./chapter-09-ensemble-learning/) | 10 | Introduction to Ensemble Learning, Bagging (Bootstrap Aggregating), Boosting, Stacking (Stacked Generalisation), Voting Classifiers, Random Forest Deep Dive, AdaBoost (Adaptive Boosting), XGBoost - Deep Dive, CatBoost, … |
| 10 | [Model Optimization](./chapter-10-model-optimization/) | 9 | Cross-Validation, Grid Search, Random Search, Bayesian Optimization, Feature Engineering, Overfitting vs Underfitting, Regularization, Model Explainability, SHAP and LIME |
| 11 | [Time Series Machine Learning](./chapter-11-time-series-machine-learning/) | 10 | Components of Time Series, Understanding Time-Based Data, Time Series Visualization and Analysis, Time Series Preprocessing, Feature Engineering for Time Series, Statistical Foundation of Time Series, … |

### Part V — ML Engineering & Deployment

| Ch | Chapter | Notebooks | Topics covered |
|---:|---|:-:|---|
| 12 | [Model Deployment](./chapter-12-model-deployment/) | 7 | Saving ML Models, Flask Deployment, FastAPI Deployment, Streamlit Dashboards, Cloud Deployment, API Integration, CI/CD for ML |
| 13 | [ML Systems Design](./chapter-13-ml-systems-design/) | 11 | ML System Architecture, Data Pipelines, Feature Stores, Batch vs Real-Time Inference, Apache Kafka for ML, Apache Airflow, Apache Spark for ML, Distributed Training, GPU vs TPU, Vector Databases, Scalable AI Systems |
| 14 | [MLOps & Monitoring](./chapter-14-mlops-and-monitoring/) | 9 | MLOps Lifecycle, Data Drift, Model Drift, Retraining Pipelines, MLflow, Monitoring Dashboards, Logging and Alerting, Governance and Compliance, Production ML Pipeline Case Study - Customer Churn Prediction |

### Part VI — Real-World AI Systems

| Ch | Chapter | Notebooks | Topics covered |
|---:|---|:-:|---|
| 15 | [Responsible AI & Future Trends](./chapter-15-responsible-ai-and-future-trends/) | 8 | Responsible AI, Fairness and Bias, Privacy in Machine Learning, Federated Learning, AutoML - Automated Machine Learning, Reinforcement Learning Basics, AI Safety, Career Roadmap in AI and ML |

### Appendices

Quick-reference code for Python, NumPy, Pandas, scikit-learn, PyTorch and TensorFlow / Keras is in [`appendices/`](./appendices/).

---

## 🗂️ Repository Structure

```
ml-mastery/
├── README.md
├── requirements.txt              # Core libraries (Chapters 1–12, 14–15)
├── requirements-advanced.txt     # Big-data, streaming & AutoML extras (Chapters 13, 15)
├── LICENSE
├── chapter-01-introduction-to-machine-learning/
├── chapter-02-mathematics-for-machine-learning/
├── chapter-03-python-for-machine-learning/
├── chapter-04-data-preprocessing-and-exploratory-data-analysis/
├── chapter-05-linear-regression/
├── chapter-06-classification-algorithms/
├── chapter-07-clustering-techniques/
├── chapter-08-dimensionality-reduction/
├── chapter-09-ensemble-learning/
├── chapter-10-model-optimization/
├── chapter-11-time-series-machine-learning/
├── chapter-12-model-deployment/
├── chapter-13-ml-systems-design/
├── chapter-14-mlops-and-monitoring/
├── chapter-15-responsible-ai-and-future-trends/
└── appendices/
```

Each chapter folder contains one notebook per book section, named by section number. For example, `chapter-05-linear-regression/05.03_linear-regression-from-scratch.ipynb` is the code for **Section 5.3**. Chapters 12–15 also have a `src/` folder with the standalone apps, scripts and config files shown in the book.

**No dataset downloads are needed.** Every example uses scikit-learn's built-in datasets or generates its own synthetic data.

---

## ⚙️ Getting Started

### Option 1: Google Colab (no setup)

Click the **Open in Colab** link next to any notebook in a chapter's README. If Colab is missing a library, install it in a cell:

```python
!pip install lightgbm catboost optuna shap umap-learn prophet
```

### Option 2: Run locally

**1. Clone the repository**
```bash
git clone https://github.com/RithikRaj129/ml-mastery.git
cd ml-mastery
```

**2. Create a virtual environment (Python 3.11 or later)**
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
# Optional - only for Chapter 13 big-data examples and Chapter 15 AutoML:
pip install -r requirements-advanced.txt
```

**4. Launch Jupyter**
```bash
jupyter lab
```

Full setup instructions, including Anaconda, VS Code and Docker, are in **Chapter 1, Section 1.9** of the book.

### Core libraries

| Library | Version | Used for |
|---------|---------|----------|
| NumPy / SciPy | 1.26.x / 1.11+ | Numerical computing, linear algebra, statistics |
| Pandas | 2.2.x | Data manipulation |
| Matplotlib / Seaborn | 3.8.x / 0.13.x | Visualisation |
| scikit-learn | 1.4.x | Classical ML |
| XGBoost / LightGBM / CatBoost | 2.0.x / 4.x / 1.2.x | Gradient boosting |
| PyTorch | 2.3.x | Deep learning |
| statsmodels / Prophet | 0.14 / 1.1 | Time series |
| Optuna / SHAP / LIME | 3.x / 0.45.x / 0.2 | Tuning & explainability |
| MLflow | 2.x | Experiment tracking & model registry |
| FastAPI / Flask / Streamlit | — | Model serving & dashboards |

---

## 🐞 Found an Error?

If you find a bug in the code or an error in the book, please [open an issue](https://github.com/RithikRaj129/ml-mastery/issues) with the **chapter and section number**. Confirmed corrections are listed below.

### Errata
_No corrections yet._

---

## 📜 License

The **code** in this repository is released under the [MIT License](LICENSE) — you are free to use it in your own projects.

The **book text, figures and content** are © 2026 Rithik Raj Vaishya. All rights reserved.

---

## 👤 About the Author

**Rithik Raj Vaishya** is a Data Scientist and Corporate Trainer with over nine years of experience at the intersection of machine learning, data engineering and applied AI. He has built and deployed ML systems across industries and designed and delivered machine learning and data science programmes for working professionals. He is the founder of **RV TechLearn LLP**.

- 💼 LinkedIn: [linkedin.com/in/rithikrajvaishya](https://linkedin.com/in/rithikrajvaishya)
- 📧 Email: rithikrajvaishy2@gmail.com

---

⭐ If this repository helps you, please give it a star!
