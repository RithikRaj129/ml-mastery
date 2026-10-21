# Appendix C — Pandas Quick Reference

*Machine Learning Mastery: Mathematics to MLOps* · quick-reference code from the book.

These are cheat sheets, not runnable scripts — several lines show alternatives side by side.

```python
import pandas as pd
import numpy as np

# ── Creating DataFrames ───────────────────────────────────────────
df = pd.DataFrame({'A': [1,2,3], 'B': [4,5,6]})
df = pd.read_csv('file.csv', parse_dates=['date'])
df = pd.read_excel('file.xlsx', sheet_name='Sheet1')
df = pd.read_parquet('file.parquet')
df = pd.read_json('file.json')
df = pd.DataFrame(np.random.randn(100, 4), columns=list('ABCD'))

# ── Inspection ───────────────────────────────────────────────────
df.head(5)   df.tail(5)   df.shape   df.dtypes   df.info()
df.describe()            # Statistical summary
df.describe(include='all') # Including categoricals
df.nunique()             # Unique values per column
df.isnull().sum()        # Missing count per column
df.isnull().mean()*100   # Missing percentage
df.duplicated().sum()    # Number of duplicate rows

# ── Selection ─────────────────────────────────────────────────────
df['col']                         # Series
df[['col1','col2']]               # DataFrame
df.loc[0, 'col']                  # Label-based
df.iloc[0, 1]                     # Position-based
df.loc[df['A'] > 0, ['B','C']]   # Boolean + column select
df.query('A > 0 and B < 5')       # Query syntax

# ── Filtering ─────────────────────────────────────────────────────
df[df['A'] > 0]
df[(df['A'] > 0) & (df['B'] < 5)]  # AND
df[(df['A'] > 0) | (df['B'] < 5)]  # OR
df[df['A'].between(1, 5)]
df[df['A'].isin([1, 2, 3])]
df[~df['A'].isin([1, 2, 3])]       # NOT isin
df[df['name'].str.contains('John', na=False)]

# ── Adding / Modifying Columns ────────────────────────────────────
df['new'] = df['A'] * 2
df['new'] = df['A'].apply(lambda x: x**2)
df['cat'] = pd.cut(df['A'], bins=3, labels=['low','med','high'])
df['qcat']= pd.qcut(df['A'], q=4, labels=False)
df = df.assign(B_log=np.log1p(df['B']), ratio=df['A']/df['B'])

# ── Missing Values ─────────────────────────────────────────────────
df.dropna()                            # Drop rows with any NaN
df.dropna(subset=['col1','col2'])      # Drop if NaN in these cols
df.fillna(0)                           # Fill with constant
df.fillna(df.mean())                   # Fill with column mean
df['col'].fillna(method='ffill')       # Forward fill
df['col'].interpolate('linear')        # Interpolate

# ── Group By ──────────────────────────────────────────────────────
df.groupby('A').agg({'B':'mean','C':'sum'})
df.groupby(['A','B'])['C'].agg(['mean','std','count'])
df.groupby('A').transform('mean')      # Returns same shape
df.pivot_table(values='C', index='A', columns='B', aggfunc='mean')

# ── Merging ───────────────────────────────────────────────────────
pd.merge(df1, df2, on='key', how='left')   # left/right/inner/outer
pd.merge(df1, df2, left_on='k1', right_on='k2')
pd.concat([df1, df2], ignore_index=True)   # Stack rows
pd.concat([df1, df2], axis=1)              # Stack columns

# ── Sorting ───────────────────────────────────────────────────────
df.sort_values('A', ascending=False)
df.sort_values(['A','B'], ascending=[True, False])
df.nlargest(10, 'score')    df.nsmallest(10, 'score')

# ── DateTime ──────────────────────────────────────────────────────
df['date'] = pd.to_datetime(df['date_str'])
df['year']  = df['date'].dt.year
df['month'] = df['date'].dt.month
df['dow']   = df['date'].dt.dayofweek  # 0=Monday
df.resample('D', on='date').sum()       # Daily resample
df.set_index('date').rolling('7D').mean()

# ── Apply ─────────────────────────────────────────────────────────
df.apply(np.sqrt)                         # Element-wise
df.apply(lambda row: row.sum(), axis=1)   # Row-wise
df['A'].map({'cat':'C','dog':'D'})         # Map values
df['A'].str.upper()                        # String methods
```
