# career_roadmap.py - Structured skill progression guide

CAREER_ROADMAP = {
    'Phase 1 - Foundations (0-6 months)': {
        'goal': 'Build the mathematical and programming foundation everything else rests on',
        'skills': [
            'Python fluency: numpy, pandas, matplotlib, seaborn',
            'Statistics: distributions, hypothesis testing, confidence intervals',
            'Linear algebra: vectors, matrices, eigendecomposition',
            'Calculus: gradients, chain rule, optimisation',
            'SQL: joins, aggregations, window functions',
            'Version control: git, GitHub, pull requests',
            'Data manipulation: cleaning, EDA, feature engineering',
        ],
        'resources': [
            'This book (Chapters 1-4)',
            'StatQuest YouTube (statistics + ML intuition)',
            '3Blue1Brown Essence of Linear Algebra / Calculus',
            'LeetCode Easy/Medium (Python coding practice)',
        ],
        'projects': [
            'EDA and visualisation of a public dataset (Kaggle)',
            'Reproduce a simple ML paper result from scratch',
        ],
    },

    'Phase 2 - Core ML (6-18 months)': {
        'goal': 'Master the full supervised/unsupervised ML toolkit',
        'skills': [
            'Supervised learning: linear/logistic regression, decision trees, SVM',
            'Ensemble methods: Random Forest, XGBoost, LightGBM, CatBoost',
            'Unsupervised: clustering, PCA, dimensionality reduction',
            'Model evaluation: cross-validation, metrics, learning curves',
            'Feature engineering and selection',
            'Hyperparameter tuning: Optuna, RandomSearch',
            'ML pipelines with sklearn',
        ],
        'resources': [
            'This book (Chapters 5-11)',
            'Hands-On Machine Learning (Aurélien Géron)',
            'Kaggle competitions (start with Titanic, House Prices)',
        ],
        'projects': [
            'End-to-end ML project: business problem → deployed model',
            'Top 25% in a Kaggle competition',
            'Write a technical blog post explaining your approach',
        ],
    },

    'Phase 3 - Specialisation (18-36 months)': {
        'goal': 'Develop deep expertise in one or more ML subfields',
        'tracks': {
            'ML Engineering': [
                'Docker, Kubernetes, cloud (AWS/GCP/Azure)',
                'FastAPI, Flask for model serving',
                'MLflow, Airflow, Spark',
                'Kafka for streaming ML',
                'MLOps: monitoring, drift detection, retraining',
            ],
            'Deep Learning': [
                'PyTorch or TensorFlow fundamentals',
                'CNNs, RNNs, Transformers, attention',
                'Pre-trained models and fine-tuning',
                'LLMs: GPT, BERT, LLaMA, Mistral',
                'Diffusion models, multimodal AI',
            ],
            'Research': [
                'Read and implement papers (arxiv.org daily)',
                'Contribute to open-source ML projects',
                'Present at local ML meetups',
                'Write original experiments and blog about results',
                'Consider a Masters or PhD',
            ],
            'AI Engineering (LLMs)': [
                'Prompt engineering and chain-of-thought',
                'RAG (Retrieval-Augmented Generation)',
                'LLM fine-tuning: LoRA, QLoRA, PEFT',
                'Agentic frameworks: LangChain, LlamaIndex, AutoGen',
                'Evaluation of LLM-based systems',
            ],
        },
    },

    'Phase 4 - Senior Practitioner (3+ years)': {
        'goal': 'Lead projects, mentor others, and operate at system level',
        'skills': [
            'Technical leadership: project planning, architecture decisions',
            'Communication: present complex ML to non-technical stakeholders',
            'Mentorship: guide junior team members in best practices',
            'Business acumen: translate ML capabilities into business value',
            'Responsible AI: fairness, privacy, governance, compliance',
            'Research literacy: evaluate and adopt new techniques critically',
        ],
        'projects': [
            'Lead an ML project from scoping to production',
            'Build and present an AI product roadmap',
            'Speak at a conference or write a technical paper',
            'Contribute to company-wide ML standards or guidelines',
        ],
    },
}

# ── Print roadmap ─────────────────────────────────────────────────
print('=== AI/ML Career Roadmap ===')
for phase, content in CAREER_ROADMAP.items():
    print(f'\n{phase}')
    print(f'  Goal: {content["goal"]}')
    if 'skills' in content:
        print(f'  Key Skills ({len(content["skills"])}):')
        for skill in content['skills'][:4]:
            print(f'    • {skill}')
        if len(content['skills']) > 4:
            print(f'    + {len(content["skills"])-4} more...')
    if 'tracks' in content:
        print(f'  Specialisation Tracks: {list(content["tracks"].keys())}')
    if 'projects' in content:
        print(f'  Milestone Projects:')
        for p in content['projects']:
            print(f'    ✓ {p}')

# ── Essential habits ──────────────────────────────────────────────
print('\n=== Essential Habits for Career Longevity ===')
habits = {
    'Read papers weekly':       'arxiv.org/cs.LG - 30 min/week keeps you current',
    'Build in public':          'GitHub portfolio, blog posts, Kaggle notebooks',
    'Teach to learn':           'Write tutorials, give talks, mentor - deepens understanding',
    'Network deliberately':      'ML conferences (NeurIPS, ICML, ICLR), meetups, Twitter/X',
    'Prioritise fundamentals':   'Frameworks change; linear algebra and probability do not',
    'Work on real problems':     'Personal projects on problems you care about build genuine skills',
    'Stay curious':              'The best ML practitioners are genuinely excited by new ideas',
    'Practice responsible AI':   'Build ethical habits now; they become non-negotiable later',
}
for habit, advice in habits.items():
    print(f'  {habit:<30}: {advice}')
