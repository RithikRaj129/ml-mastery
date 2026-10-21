# Appendix E — PyTorch Essentials

*Machine Learning Mastery: Mathematics to MLOps* · quick-reference code from the book.

These are cheat sheets, not runnable scripts — several lines show alternatives side by side.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

# ── Device Setup ──────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Tensors ───────────────────────────────────────────────────────
x = torch.tensor([1.0, 2.0, 3.0])          # From list
x = torch.from_numpy(np_array)             # From NumPy
x = x.to(device)                           # Move to GPU
x.numpy()                                  # Back to NumPy
torch.zeros(3,4)  torch.ones(3,4)  torch.randn(3,4)
torch.arange(0,10,0.5)  torch.linspace(0,1,100)

# ── Neural Network Definition ─────────────────────────────────────
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.3):
        super().__init__()
        layers = []
        dims   = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims)-2:
                layers.append(nn.ReLU())
                layers.append(nn.BatchNorm1d(dims[i+1]))
                layers.append(nn.Dropout(dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

model = MLPClassifier(30, [128, 64], 1).to(device)

# ── Training Loop ─────────────────────────────────────────────────
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch).squeeze()
        loss   = criterion(logits, y_batch.float())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    all_probs, all_labels = [], []
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        probs   = torch.sigmoid(model(X_batch).squeeze()).cpu()
        all_probs.extend(probs.numpy())
        all_labels.extend(y_batch.numpy())
    return np.array(all_probs), np.array(all_labels)

# ── DataLoader ────────────────────────────────────────────────────
X_t = torch.FloatTensor(X_train)
y_t = torch.FloatTensor(y_train)
dataset    = TensorDataset(X_t, y_t)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

# ── Saving and Loading ────────────────────────────────────────────
torch.save(model.state_dict(), 'model.pt')
model.load_state_dict(torch.load('model.pt', map_location=device))
torch.save({'epoch':epoch,'model':model.state_dict(),'opt':optimizer.state_dict()}, 'checkpoint.pt')

# ── Common Loss Functions ─────────────────────────────────────────
nn.BCEWithLogitsLoss()      # Binary classification
nn.CrossEntropyLoss()       # Multi-class (logits input)
nn.MSELoss()                # Regression
nn.L1Loss()                 # Regression - MAE
nn.HuberLoss(delta=1.0)    # Regression - robust

# ── Activation Functions ──────────────────────────────────────────
nn.ReLU()   nn.LeakyReLU(0.01)   nn.GELU()   nn.Sigmoid()   nn.Tanh()
nn.Softmax(dim=1)   nn.LogSoftmax(dim=1)

# ── Regularisation ───────────────────────────────────────────────
nn.Dropout(p=0.5)           # Zero out 50% of activations during training
nn.BatchNorm1d(hidden_dim)  # Normalise activations
nn.LayerNorm(hidden_dim)    # Preferred for Transformers
weight_decay=1e-4           # L2 via optimizer (AdamW)
```
