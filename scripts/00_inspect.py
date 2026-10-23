import sys
import pandas as pd

d = sys.argv[1].rstrip('/') + '/'
Y = pd.read_csv(d + 'geuvadis_peer_normalized.csv', index_col=0)
E = pd.read_csv(d + 'enformer_sample_preds.csv', index_col=0)
P = pd.read_csv(d + 'predixcan_197KB_sample_preds.csv', index_col=0)
X = pd.read_csv(d + 'xpresso_sample_preds.csv', index_col=0)
M = pd.read_csv(d + 'meta_sample_preds.csv', index_col=0)

print('shapes  Y', Y.shape, 'E', E.shape, 'P', P.shape, 'X', X.shape, 'M', M.shape)
print('Y col mean (5):', Y.mean().head().values.round(4))
print('Y col std  (5):', Y.std().head().values.round(4))
print('Y idx', list(Y.index[:3]))
print('E idx', list(E.index[:3]))
print('P idx', list(P.index[:3]))
print('X idx', list(X.index[:3]))
print('M idx', list(M.index[:3]))
print('index identical Y==E', list(Y.index) == list(E.index),
      '| Y==P', list(Y.index) == list(P.index),
      '| Y==X', list(Y.index) == list(X.index),
      '| Y==M', list(Y.index) == list(M.index))
for nm, df in [('E', E), ('P', P), ('X', X), ('M', M)]:
    print(nm, 'gene overlap with Y:', len(set(Y.columns) & set(df.columns)),
          '| nan frac %.4f' % df.isna().mean().mean(),
          '| constant-col frac %.4f' % (df.std() == 0).mean())
print('Y nan frac %.4f' % Y.isna().mean().mean())
