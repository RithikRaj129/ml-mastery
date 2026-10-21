# federated_learning_sim.py - Simulate Federated Averaging
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import copy
import warnings; warnings.filterwarnings('ignore')

# ── Simple FedAvg simulation ──────────────────────────────────────
class FederatedClient:
    def __init__(self, client_id: int, X: np.ndarray, y: np.ndarray):
        self.client_id = client_id
        self.X = X
        self.y = y
        self.n = len(y)
        self.weights = None
        self.bias    = None

    def receive_global_weights(self, weights: np.ndarray, bias: float):
        self.weights = weights.copy()
        self.bias    = bias

    def local_train(self, lr: float = 0.1, n_epochs: int = 5) -> tuple:
        '''Train locally and return updated weights.'''
        w = self.weights.copy()
        b = self.bias
        n = len(self.X)

        for _ in range(n_epochs):
            p   = 1/(1+np.exp(-np.clip(self.X @ w + b, -500, 500)))
            err = p - self.y
            w  -= lr * (self.X.T @ err) / n
            b  -= lr * err.mean()

        return w, b  # Return updated weights (gradient = w - global_w)


class FederatedServer:
    def __init__(self, n_features: int):
        self.weights = np.zeros(n_features)
        self.bias    = 0.0
        self.history = []

    def distribute_weights(self, clients: list) -> None:
        for client in clients:
            client.receive_global_weights(self.weights, self.bias)

    def aggregate(self, clients: list, updated_weights: list) -> None:
        '''FedAvg: weighted average by dataset size.'''
        total_n = sum(c.n for c in clients)
        new_w   = np.zeros_like(self.weights)
        new_b   = 0.0
        for client, (w, b) in zip(clients, updated_weights):
            weight = client.n / total_n
            new_w += weight * w
            new_b += weight * b
        self.weights = new_w
        self.bias    = new_b

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = 1/(1+np.exp(-np.clip(X @ self.weights + self.bias, -500, 500)))
        return np.column_stack([1-p, p])


# ── Data - split across 5 clients (non-IID to make it realistic) ─
data = load_breast_cancer()
X_sc = StandardScaler().fit_transform(data.data)
X_tr, X_te, y_tr, y_te = train_test_split(X_sc, data.target, test_size=0.2, random_state=42)

N_CLIENTS = 5
# Non-IID split: each client gets more of one class
np.random.seed(42)
client_datasets = []
indices = np.random.permutation(len(X_tr))
chunks  = np.array_split(indices, N_CLIENTS)
for i, chunk in enumerate(chunks):
    client_datasets.append(FederatedClient(i, X_tr[chunk], y_tr[chunk]))

print('=== Federated Learning Simulation ===')
print(f'Clients: {N_CLIENTS}')
for c in client_datasets:
    print(f'  Client {c.client_id}: {c.n} samples, {c.y.mean():.0%} positive')

# ── Run federated training for 20 rounds ─────────────────────────
server = FederatedServer(n_features=X_tr.shape[1])

print(f'\nRound  AUC (Federated)  AUC (Central baseline)')
print('='*52)

# Central baseline
central_lr = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
central_auc = roc_auc_score(y_te, central_lr.predict_proba(X_te)[:,1])

for round_num in range(1, 21):
    # Distribute global weights
    server.distribute_weights(client_datasets)

    # Each client trains locally
    updated = [c.local_train(lr=0.1, n_epochs=3) for c in client_datasets]

    # Aggregate
    server.aggregate(client_datasets, updated)

    # Evaluate global model
    fed_auc = roc_auc_score(y_te, server.predict_proba(X_te)[:,1])

    if round_num in [1, 3, 5, 10, 15, 20]:
        gap = central_auc - fed_auc
        print(f'{round_num:>5}  {fed_auc:>15.4f}  {central_auc:>22.4f}  (gap: {gap:.4f})')

print(f'\nFinal federated AUC: {fed_auc:.4f}')
print(f'Central (non-private): {central_auc:.4f}')
print(f'Privacy cost: {central_auc-fed_auc:.4f} AUC')
print(f'\nKey advantage: NO raw data ever left any client!')
