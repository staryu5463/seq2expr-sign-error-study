"""Final summary numbers and figure."""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = sys.argv[1].rstrip('/') + '/'
R = pd.read_csv(OUT + 'per_gene_r.csv', index_col=0)
sw = pd.read_csv(OUT + 'locus_swap.csv')
fd = pd.read_csv(OUT + 'flip_vs_distance.csv')
ag = pd.read_csv(OUT + 'flip_agreement.csv')

n = 421
r_crit = 1.959964 / np.sqrt(n - 1)
SEQ = ['Enformer', 'Basenji2', 'ExPecto', 'Xpresso']
s = sw.set_index('model')

print('=== headline ===')
px_signed = s.loc['PrediXcan', 'mean_r']
for m in SEQ:
    signed, unsigned = s.loc[m, 'mean_r'], s.loc[m, 'mean_abs_r']
    gap_closed = (unsigned - signed) / (px_signed - signed)
    print('%-9s signed %.4f | unsigned %.4f | swap-null %.4f | sign accuracy %.1f%% '
          '| sign fix closes %.0f%% of the gap to PrediXcan'
          % (m, signed, unsigned, s.loc[m, 'locus_swap_abs'],
             100 * (1 - s.loc[m, 'frac_neg_among_fdr']), 100 * gap_closed))
print('%-9s signed %.4f | unsigned %.4f | sign accuracy %.1f%%'
      % ('PrediXcan', px_signed, s.loc['PrediXcan', 'mean_abs_r'],
         100 * (1 - s.loc['PrediXcan', 'frac_neg_among_fdr'])))
print('\nfraction of the available magnitude that survives signing'
      ' (mean r / mean |r|, FDR-significant genes only):')
for m in SEQ + ['PrediXcan']:
    print('  %-9s %.3f' % (m, s.loc[m, 'mean_r_among_fdr'] / s.loc[m, 'mean_abs_r_among_fdr']))

fig, ax = plt.subplots(1, 4, figsize=(17, 4.0))

# panel 1: signed vs unsigned vs nulls
x = np.arange(len(SEQ) + 1)
mods = SEQ + ['PrediXcan']
w = 0.26
ax[0].bar(x - w, [s.loc[m, 'mean_r'] for m in mods], w, label='signed  mean $r$', color='#c44e52')
ax[0].bar(x, [s.loc[m, 'mean_abs_r'] for m in mods], w, label='unsigned  mean $|r|$', color='#4c72b0')
ax[0].bar(x + w, [s.loc[m, 'locus_swap_abs'] for m in mods], w,
          label='locus-swap null', color='#bbbbbb')
ax[0].axhline(s['label_perm_abs'].mean(), ls=':', c='k', lw=1,
              label='label-permutation null')
ax[0].set_xticks(x); ax[0].set_xticklabels(mods, rotation=20, ha='right')
ax[0].set_ylabel('across-individual Spearman')
ax[0].set_title('A  the signal is there; the sign is not')
ax[0].legend(fontsize=7.5, frameon=False)

# panel 2: distribution of per-gene r for Enformer
ax[1].hist(R['Enformer'].dropna(), bins=60, color='#4c72b0', alpha=.85)
ax[1].axvline(0, c='k', lw=1)
ax[1].axvline(r_crit, c='#c44e52', ls='--', lw=1)
ax[1].axvline(-r_crit, c='#c44e52', ls='--', lw=1)
ax[1].set_xlabel('per-gene across-individual $r$  (Enformer)')
ax[1].set_ylabel('genes')
neg = (R['Enformer'] < -r_crit).sum()
pos = (R['Enformer'] > r_crit).sum()
ax[1].set_title('B  %d genes significant positive,\n%d significant negative' % (pos, neg))

# panel 3: flip rate vs lead-eQTL distance
for m, c in zip(SEQ, ['#4c72b0', '#55a868', '#8172b2', '#ccb974']):
    sub = fd[(fd.model == m) & fd.n.gt(0)]
    xs = np.arange(len(sub))
    ax[2].errorbar(xs, sub.flip_rate,
                   yerr=[sub.flip_rate - sub.ci_lo, sub.ci_hi - sub.flip_rate],
                   marker='o', capsize=3, label=m, color=c)
    ax[2].set_xticks(xs); ax[2].set_xticklabels(sub['bin'], rotation=15)
ax[2].axhline(0.5, ls=':', c='k', lw=1)
ax[2].set_ylabel('fraction with the wrong sign')
ax[2].set_xlabel('lead cis-eQTL distance to TSS')
ax[2].set_title('C  sign decays with distance\n(annotation caps at 20 kb)')
ax[2].legend(fontsize=7.5, frameon=False)

# panel 4: flip agreement across architectures
xs = ag['n_models_flipped']
ax[3].bar(xs - 0.2, ag['observed'], 0.4, label='observed', color='#4c72b0')
ax[3].bar(xs + 0.2, ag['expected_if_independent'], 0.4,
          label='if models were independent', color='#bbbbbb')
ax[3].set_xlabel('how many of the 4 models flip the sign')
ax[3].set_ylabel('genes')
ax[3].set_title('D  the same genes flip in every\narchitecture (all-4: %.1fx)'
                % (ag.loc[4, 'observed'] / ag.loc[4, 'expected_if_independent']))
ax[3].legend(fontsize=7.5, frameon=False)

for a in ax:
    a.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig(OUT + 'figure.png', dpi=160)
print('\nwrote', OUT + 'figure.png')
