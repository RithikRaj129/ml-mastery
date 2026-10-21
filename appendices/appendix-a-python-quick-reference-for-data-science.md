# Appendix A — Python Quick Reference for Data Science

*Machine Learning Mastery: Mathematics to MLOps* · quick-reference code from the book.

These are cheat sheets, not runnable scripts — several lines show alternatives side by side.

```python
# ── Data Types ───────────────────────────────────────────────────
x = 42           # int        float: 3.14     bool: True/False
s = 'hello'      # str        list: [1,2,3]   tuple: (1,2,3)
d = {'k': 'v'}   # dict       set: {1,2,3}    None: None

# ── String Methods ───────────────────────────────────────────────
s.upper()  s.lower()  s.strip()  s.split(',')  ','.join(['a','b'])
f'Hello {name}'  # f-string formatting
f'{value:.4f}'   # 4 decimal places

# ── List / Dict Comprehensions ───────────────────────────────────
squares   = [x**2 for x in range(10) if x % 2 == 0]
mapping   = {k: v for k, v in zip(keys, values)}
filtered  = {k: v for k, v in d.items() if v > 0}

# ── Functions ────────────────────────────────────────────────────
def greet(name: str, greeting: str = 'Hello') -> str:
    return f'{greeting}, {name}!'

add = lambda x, y: x + y           # Lambda
from functools import partial
double = partial(lambda x,n: x*n, n=2)

# ── *args and **kwargs ───────────────────────────────────────────
def func(*args, **kwargs):
    print(args, kwargs)  # (1,2,3) {'key': 'val'}

# ── Classes ──────────────────────────────────────────────────────
class Model:
    def __init__(self, lr: float = 0.01):
        self.lr = lr
    def fit(self, X, y): ...
    def predict(self, X): ...
    @property
    def params(self): return {'lr': self.lr}
    def __repr__(self): return f'Model(lr={self.lr})'

# ── Error Handling ───────────────────────────────────────────────
try:
    result = risky_operation()
except ValueError as e:
    print(f'ValueError: {e}')
except (TypeError, KeyError) as e:
    print(f'Multiple: {e}')
except Exception as e:
    raise RuntimeError('wrapped') from e
finally:
    cleanup()

# ── Context Managers ─────────────────────────────────────────────
with open('file.csv', 'r') as f:
    data = f.read()

# ── File I/O ─────────────────────────────────────────────────────
import json, pickle, csv
json.dump(obj, open('f.json','w'), indent=2)
obj = json.load(open('f.json'))
pickle.dump(model, open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))

# ── Itertools & Useful Stdlib ─────────────────────────────────────
from itertools import product, combinations, chain
from collections import defaultdict, Counter, OrderedDict
from pathlib import Path
from datetime import datetime, timedelta
import os, sys, time, copy, random, math

# ── Type Hints ───────────────────────────────────────────────────
from typing import List, Dict, Optional, Tuple, Union, Any, Callable
def process(data: List[Dict[str, Any]]) -> Optional[float]: ...

# ── Dataclasses ──────────────────────────────────────────────────
from dataclasses import dataclass, field
@dataclass
class Config:
    learning_rate: float = 0.01
    n_epochs:      int   = 100
    features:      List[str] = field(default_factory=list)

# ── Generators ───────────────────────────────────────────────────
def batch_generator(X, y, batch_size=32):
    for i in range(0, len(X), batch_size):
        yield X[i:i+batch_size], y[i:i+batch_size]

# ── Useful one-liners ─────────────────────────────────────────────
max(lst, key=lambda x: x['score'])   # Max by field
sorted(lst, key=lambda x: -x['val']) # Sort descending
any(x > 0 for x in lst)              # Any positive
all(x > 0 for x in lst)              # All positive
dict(zip(keys, values))              # Two lists → dict
flat = [x for sub in nested for x in sub]  # Flatten
```
