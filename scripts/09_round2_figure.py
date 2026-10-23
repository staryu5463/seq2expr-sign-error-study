"""Round-2 figure: the AUC bar was passed and the method still failed."""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = sys.argv[1].rstrip('/') + '/'
pol = pd.read_csv(OUT + 'correction_policies.csv')
clf = pd.read_csv(OUT + 'flip_classifier.csv')
tri = pd.read_csv(OUT + 'triage_Enformer.csv')

# quintile table recomputed in 07; hard-coded here from its output for the panel
quint = pd.DataFrame({
    'q': [1, 2, 3, 4, 5],
    'flip_rate': [0.260, 0.318, 0.385, 0.379, 0.419],
    'mean_abs_r': [0.2538, 0.2272, 0.2090, 0.2019, 0.1884],
})

fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.1))

# A: AUC vs the bar
h = clf[(clf.features == 'honest')]
x = np.arange(len(h))
ax[0].bar(x, h.loc_auc if 'loc_auc' in h else h.loco_auc, 0.55, color='#4c72b0')
ax[0].errorbar(x, h.null_auc, yerr=h.null_sd, fmt='_', color='k', capsize=4,
               label='shuffled-label null')
ax[0].axhline(0.55, ls='--', c='#c44e52', label='pre-registered bar')
ax[0].set_xticks(x)
ax[0].set_xticklabels([f'{t}\n{c}' for t, c in zip(h.target, h.clf)], fontsize=7.5)
ax[0].set_ylim(0.45, 0.62)
ax[0].set_ylabel('leave-one-chromosome-out AUC')
ax[0].set_title('A  the classifier clears the bar')
ax[0].legend(fontsize=7.5, frameon=False)

# B: but flipping gains nothing
e = pol[pol.model == 'Enformer']
order = ['baseline', 'flip if p>0.5', 'flip if p>0.6', 'flip if p>0.7',
         'keep lowest 75% p', 'keep lowest 50% p', 'sign oracle (bound)']
e = e.set_index('policy').loc[order].reset_index()
cols = ['#999999', '#c44e52', '#c44e52', '#c44e52', '#55a868', '#55a868', '#cccccc']
ax[1].barh(np.arange(len(e))[::-1], e.mean_r, color=cols)
ax[1].set_yticks(np.arange(len(e))[::-1])
ax[1].set_yticklabels(e.policy, fontsize=8)
ax[1].axvline(e.mean_r.iloc[0], ls=':', c='k', lw=1)
ax[1].set_xlabel('mean across-individual $r$  (Enformer)')
ax[1].set_title('B  flipping gains nothing;\nfiltering gains 42%')

# C: why -- flip rate never crosses 0.5
a2 = ax[2]
a2.plot(quint.q, quint.flip_rate, 'o-', color='#c44e52', label='flip rate')
a2.axhline(0.5, ls='--', c='k', lw=1)
a2.set_ylim(0.2, 0.56)
a2.set_xlabel('quintile of predicted flip probability')
a2.set_ylabel('flip rate', color='#c44e52')
a2.set_xticks(quint.q)
a3 = a2.twinx()
a3.plot(quint.q, quint.mean_abs_r, 's-', color='#4c72b0', label='mean |r|')
a3.set_ylabel('mean $|r|$', color='#4c72b0')
a3.spines[['top']].set_visible(False)
a2.set_title('C  it finds unreliable genes,\nnot inverted ones')

for a in ax:
    a.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig(OUT + 'figure_round2.png', dpi=160)
print('wrote', OUT + 'figure_round2.png')
print('\ntop triage features for Enformer:')
print(tri.head(5).round(4).to_string(index=False))
