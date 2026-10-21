# distributed_training.py - Data-parallel training with Ray
# Install: pip install ray[train]

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# ── Data Parallelism with Ray Train ──────────────────────────────
try:
    import ray
    from ray import train
    from ray.train.sklearn import SklearnTrainer
    from ray.train import RunConfig, ScalingConfig

    # Generate dataset
    X, y = make_classification(n_samples=100000, n_features=50, n_informative=20, random_state=42)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    import pandas as pd
    train_df = pd.DataFrame(X_tr, columns=[f'f{i}' for i in range(50)])
    train_df['label'] = y_tr

    from sklearn.ensemble import GradientBoostingClassifier

    trainer = SklearnTrainer(
        estimator=GradientBoostingClassifier(n_estimators=200, random_state=42),
        label_column='label',
        scaling_config=ScalingConfig(trainer_resources={'CPU': 4}),
        run_config=RunConfig(name='gb_distributed'),
    )

    result = trainer.fit(ray.data.from_pandas(train_df))
    print(f'Ray Train completed. Checkpoint: {result.checkpoint}')

except ImportError:
    print('Ray not installed. Showing distributed training patterns.')


# ── Horovod-style data parallelism (conceptual) ───────────────────
print('\n=== Distributed Training Patterns ===')
patterns = {
    'Horovod': {
        'use': 'pip install horovod; works with TensorFlow, PyTorch, MXNet',
        'launch': 'horovodrun -np 8 -H server1:4,server2:4 python train.py',
        'code': 'hvd.init(); model → DistributedOptimizer; hvd.allreduce(gradients)',
    },
    'PyTorch DDP': {
        'use': 'Built into PyTorch; single-machine multi-GPU or multi-node',
        'launch': 'torchrun --nproc_per_node=4 train.py',
        'code': 'model = DistributedDataParallel(model); sampler = DistributedSampler(dataset)',
    },
    'Ray Train': {
        'use': 'pip install ray[train]; works with sklearn, PyTorch, TF, XGBoost',
        'launch': 'Python script; Ray handles cluster management',
        'code': 'ScalingConfig(num_workers=8, use_gpu=True); TorchTrainer / XGBoostTrainer',
    },
    'Dask-ML': {
        'use': 'pip install dask dask-ml; sklearn-compatible API',
        'launch': 'dask.distributed.LocalCluster or cloud cluster',
        'code': 'from dask_ml.ensemble import BlockwiseVotingClassifier',
    },
}
for name, info in patterns.items():
    print(f'\n{name}:')
    for key, val in info.items():
        print(f'  {key:8s}: {val}')


# ── Gradient synchronisation explained ───────────────────────────
print('\n=== Data Parallelism: How Gradient Sync Works ===')
steps = [
    ('1. Replicate', 'Copy full model to all N workers (GPUs/machines)'),
    ('2. Partition',  'Split each mini-batch into N micro-batches'),
    ('3. Forward',    'Each worker computes forward pass on its micro-batch'),
    ('4. Backward',   'Each worker computes local gradients'),
    ('5. All-Reduce', 'Average gradients across all workers (Ring All-Reduce: O(N) → O(1))'),
    ('6. Update',     'All workers apply identical gradient update to identical model'),
    ('7. Repeat',     'All workers now have identical updated model; proceed to next batch'),
]
for step, desc in steps:
    print(f'  {step:12s}: {desc}')

print('\nEffective batch size = per_worker_batch × N_workers')
print('Throughput scales ~linearly with N_workers')
print('Communication bottleneck: gradient size / network bandwidth')
