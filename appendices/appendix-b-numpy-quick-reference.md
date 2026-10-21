# Appendix B — NumPy Quick Reference

*Machine Learning Mastery: Mathematics to MLOps* · quick-reference code from the book.

These are cheat sheets, not runnable scripts — several lines show alternatives side by side.

```python
import numpy as np

# ── Array Creation ───────────────────────────────────────────────
np.array([1,2,3])                  # From list
np.zeros((3,4))  np.ones((3,4))   # Zero/one arrays
np.eye(3)                          # Identity matrix
np.arange(0, 10, 0.5)             # Like range() with floats
np.linspace(0, 1, 100)            # 100 evenly spaced points
np.random.randn(3, 4)             # Standard normal (3×4)
np.random.rand(3, 4)              # Uniform [0,1]
np.random.randint(0, 10, (3,4))   # Random integers
np.random.seed(42)                 # Reproducibility

# ── Shape and Type ───────────────────────────────────────────────
a.shape    a.ndim    a.size    a.dtype
a.reshape(2, 6)       a.flatten()     a.ravel()
a.T                   a.squeeze()     np.expand_dims(a, 0)
a.astype(np.float32)  a.copy()

# ── Indexing and Slicing ─────────────────────────────────────────
a[0]     a[-1]     a[1:4]     a[::2]     a[1,2]
a[[0,2,4]]              # Fancy indexing
a[a > 0]                # Boolean indexing
a[np.where(a > 0)]      # Where indexing
np.nonzero(a)           # Indices of non-zero elements

# ── Math Operations ──────────────────────────────────────────────
a + b    a - b    a * b    a / b    a ** 2    a // b    a % b
np.sqrt(a)   np.log(a)   np.log1p(a)   np.exp(a)
np.abs(a)    np.clip(a, 0, 1)    np.round(a, 3)
np.sin(a)    np.cos(a)   np.tanh(a)

# ── Aggregation ──────────────────────────────────────────────────
a.sum()      a.sum(axis=0)     a.sum(axis=1)
a.mean()     a.std()           a.var()
a.min()      a.max()           a.argmin()    a.argmax()
a.cumsum()   a.cumprod()
np.median(a)    np.percentile(a, [25,50,75])
np.unique(a)    np.unique(a, return_counts=True)

# ── Linear Algebra ────────────────────────────────────────────────
np.dot(a, b)              # Dot product / matrix multiply
a @ b                     # Matrix multiply (Python 3.5+)
np.linalg.norm(a)         # L2 norm
np.linalg.norm(a, axis=1) # Row-wise norms
np.linalg.inv(A)          # Matrix inverse
np.linalg.det(A)          # Determinant
eigvals, eigvecs = np.linalg.eig(A)
U, S, Vt = np.linalg.svd(A)
np.linalg.solve(A, b)     # Solve Ax = b

# ── Stacking and Splitting ────────────────────────────────────────
np.concatenate([a, b], axis=0)   np.vstack([a,b])   np.hstack([a,b])
np.stack([a, b], axis=0)         # New axis
np.split(a, 3)                   np.array_split(a, 5)
np.column_stack([v1, v2, v3])    # Vectors as columns

# ── Sorting ──────────────────────────────────────────────────────
np.sort(a)               # Returns sorted copy
np.argsort(a)            # Indices that would sort
np.argsort(a)[::-1]      # Descending order indices

# ── Broadcasting rules ───────────────────────────────────────────
# Two dimensions are compatible when: equal, or one is 1
# (3,4) + (4,) → (3,4)   |   (3,1) + (1,4) → (3,4)
a = np.random.randn(100, 10)
a_normalised = (a - a.mean(axis=0)) / (a.std(axis=0) + 1e-8)

# ── Masking ──────────────────────────────────────────────────────
mask = a > 0
a[mask] = 0                      # Zero out positives
np.where(a > 0, a, 0)            # Conditional replace
np.ma.masked_where(a < 0, a)     # Masked array
```
