"""
Where do the sign errors come from?

Given that the unsigned across-individual signal is ~3.8x its permutation null
while the signed signal is ~1.2x, the sign is the thing that is broken. Three
candidate mechanisms are tested here, plus one more confound control.

  A. distance  -- flips get more likely as the lead cis-eQTL moves away from the
                  TSS. This is what Karollus et al. 2023 would predict: the model
                  under-weights distal elements, so its sign there is a coin flip.
  B. direction -- flips track the *sign* of the true eQTL effect, i.e. the model
                  carries a fixed polarity prior and is right only when the locus
                  happens to agree with it.
  C. shared    -- the same genes flip in every architecture, which would make the
                  error a property of the training paradigm, not of the model.
  D. control   -- residualize the top-k individual-level PCs of the prediction
                  matrix out of both sides, to rule out a global per-individual
                  technical factor (e.g. variant count) manufacturing per-gene
                  correlations with arbitrary signs.
"""
import sys
import numpy as np
import pandas as pd

PRED_DIR = sys.argv[1].rstrip('/') + '/'
EQTL = sys.argv[2]
OUT_DIR = sys.argv[3].rstrip('/') + '/'
SEED = 0

MODELS = {
    'Enformer': 'enformer_sample_preds.csv',
    'Basenji2': 'basenji2_sample_preds.csv',
    'ExPecto': 'expecto_sample_preds.csv',
    'Xpresso': 'xpresso_sample_preds.csv',
    'PrediXcan': 'predixcan_197KB_sample_preds.csv',
}
SEQ_MODELS = ['Enformer', 'Basenji2', 'ExPecto', 'Xpresso']


def colwise_spearman(A, B):
    ra = pd.DataFrame(A).rank(axis=0).to_numpy()
    rb = pd.DataFrame(B).rank(axis=0).to_numpy()
    ra = ra - ra.mean(0)
    rb = rb - rb.mean(0)
    num = (ra * rb).sum(0)
    den = np.sqrt((ra ** 2).sum(0) * (rb ** 2).sum(0))
    with np.errstate(invalid='ignore', divide='ignore'):
        return num / den


def wilson(k, n):
    """Wilson 95% CI for a proportion."""
    if n == 0:
        return (np.nan, np.nan)
    z, p = 1.959964, k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (c - h, c + h)


def main():
    Y = pd.read_csv(PRED_DIR + 'geuvadis_peer_normalized.csv', index_col=0)
    n = len(Y)
    r_crit = 1.959964 / np.sqrt(n - 1)
    R = pd.read_csv(OUT_DIR + 'per_gene_r.csv', index_col=0)

    eq = pd.read_csv(EQTL)
    eq = eq.dropna(subset=['name', 'distance'])
    eq['name'] = eq['name'].str.lower()
    eq = eq.drop_duplicates('name').set_index('name')
    print('lead-eQTL annotation available for %d of %d genes' % (
        len(set(R.index) & set(eq.index)), len(R)))
    print('distance quantiles (bp):',
          eq['distance'].abs().quantile([0, .25, .5, .75, 1]).astype(int).to_dict(), '\n')

    # ---------- A. flip rate vs lead-eQTL distance to TSS ----------
    print('=== A. does the sign break down with distance from the TSS? ===')
    print('(genes with |r| > %.4f only; flip = model r points the wrong way)\n' % r_crit)
    edges = [0, 1e3, 1e4, 5e4, 1e5, np.inf]
    labels = ['<1kb', '1-10kb', '10-50kb', '50-100kb', '>100kb']
    recA = []
    for m in SEQ_MODELS + ['PrediXcan']:
        r = R[m].dropna()
        common = r.index.intersection(eq.index)
        r = r.loc[common]
        dist = eq.loc[common, 'distance'].abs()
        sig = r.abs() > r_crit
        b = pd.cut(dist[sig], bins=edges, labels=labels)
        for lab in labels:
            sel = (b == lab)
            k = int((r[sig][sel] < 0).sum())
            tot = int(sel.sum())
            lo, hi = wilson(k, tot)
            recA.append(dict(model=m, bin=lab, n=tot, flip_rate=k / tot if tot else np.nan,
                             ci_lo=lo, ci_hi=hi,
                             mean_abs_r=r[sig][sel].abs().mean()))
    A = pd.DataFrame(recA)
    for m in SEQ_MODELS + ['PrediXcan']:
        sub = A[A.model == m]
        print(m)
        print(sub[['bin', 'n', 'flip_rate', 'ci_lo', 'ci_hi', 'mean_abs_r']]
              .round(4).to_string(index=False))
        print()
    A.to_csv(OUT_DIR + 'flip_vs_distance.csv', index=False)

    # ---------- B. flip rate vs true eQTL effect direction ----------
    print('=== B. does the sign error track the true eQTL effect direction? ===')
    recB = []
    for m in SEQ_MODELS + ['PrediXcan']:
        r = R[m].dropna()
        common = r.index.intersection(eq.index)
        r = r.loc[common]
        rv = eq.loc[common, 'rvalue'].astype(float)
        sig = (r.abs() > r_crit) & rv.notna()
        for s, lab in [(1, 'eQTL r>0'), (-1, 'eQTL r<0')]:
            sel = sig & (np.sign(rv) == s)
            k = int((r[sel] < 0).sum())
            tot = int(sel.sum())
            lo, hi = wilson(k, tot)
            recB.append(dict(model=m, group=lab, n=tot,
                             flip_rate=k / tot if tot else np.nan, ci_lo=lo, ci_hi=hi))
    B = pd.DataFrame(recB)
    print(B.round(4).to_string(index=False))
    B.to_csv(OUT_DIR + 'flip_vs_eqtl_direction.csv', index=False)

    # ---------- C. is the same gene flipped by every architecture? ----------
    print('\n=== C. cross-model structure of the per-gene signed correlation ===')
    print('Pearson correlation between models of the per-gene signed r:')
    print(R[SEQ_MODELS + ['PrediXcan']].corr().round(3).to_string())
    print('\nsign-concordance on genes significant in BOTH models'
          ' (chance = 0.5; off-diagonal):')
    cm = pd.DataFrame(index=SEQ_MODELS + ['PrediXcan'], columns=SEQ_MODELS + ['PrediXcan'],
                      dtype=float)
    cnt = cm.copy()
    for a in cm.index:
        for b in cm.columns:
            ra, rb = R[a], R[b]
            sel = (ra.abs() > r_crit) & (rb.abs() > r_crit) & ra.notna() & rb.notna()
            cm.loc[a, b] = (np.sign(ra[sel]) == np.sign(rb[sel])).mean()
            cnt.loc[a, b] = int(sel.sum())
    print(cm.astype(float).round(3).to_string())
    cm.to_csv(OUT_DIR + 'sign_concordance.csv')

    # fraction of sequence models agreeing with the truth, per gene
    flips = pd.DataFrame({m: (R[m] < 0) for m in SEQ_MODELS})
    sigall = pd.DataFrame({m: (R[m].abs() > r_crit) for m in SEQ_MODELS})
    both = sigall.all(axis=1)
    k = flips[both].sum(axis=1)
    print('\namong the %d genes significant in all 4 sequence models,' % int(both.sum()))
    print('how many of the 4 flip the sign:')
    obs = k.value_counts().reindex(range(5), fill_value=0)
    # independence null using each model's own marginal flip rate
    p = [flips[both][m].mean() for m in SEQ_MODELS]
    from itertools import product
    exp = np.zeros(5)
    for bits in product([0, 1], repeat=4):
        pr = np.prod([p[i] if b else 1 - p[i] for i, b in enumerate(bits)])
        exp[sum(bits)] += pr
    tab = pd.DataFrame({'n_models_flipped': range(5), 'observed': obs.values,
                        'expected_if_independent': (exp * both.sum()).round(1)})
    print(tab.to_string(index=False))
    tab.to_csv(OUT_DIR + 'flip_agreement.csv', index=False)

    # ---------- D. global per-individual factor control ----------
    print('\n=== D. control: remove the top-k individual-level PCs of the predictions ===')
    rng = np.random.default_rng(SEED)
    recD = []
    for m in ['Enformer', 'PrediXcan']:
        P = pd.read_csv(PRED_DIR + MODELS[m], index_col=0)
        genes = [g for g in Y.columns if g in P.columns]
        yv = Y[genes].to_numpy(float)
        pv = P[genes].to_numpy(float)
        keep = ~np.isnan(pv).any(0) & (np.nanstd(pv, 0) > 0)
        yv, pv = yv[:, keep], pv[:, keep]
        Z = (pv - pv.mean(0)) / pv.std(0)
        U, S, _ = np.linalg.svd(Z, full_matrices=False)
        for k_pc in [0, 1, 5, 20]:
            if k_pc == 0:
                yr, pr_ = yv, pv
            else:
                D = np.c_[np.ones(n), U[:, :k_pc]]
                beta, *_ = np.linalg.lstsq(D, yv, rcond=None)
                yr = yv - D @ beta
                beta, *_ = np.linalg.lstsq(D, pv, rcond=None)
                pr_ = pv - D @ beta
            r = colwise_spearman(yr, pr_)
            na = np.array([np.nanmean(np.abs(colwise_spearman(yr, pr_[rng.permutation(n)])))
                           for _ in range(40)])
            sig = np.abs(r) > r_crit
            recD.append(dict(model=m, k_pcs_removed=k_pc,
                             mean_r=np.nanmean(r), mean_abs_r=np.nanmean(np.abs(r)),
                             null_abs_r=na.mean(), ratio_abs=np.nanmean(np.abs(r)) / na.mean(),
                             frac_sig=sig.mean(), frac_sig_neg=(r[sig] < 0).mean()))
    D_ = pd.DataFrame(recD)
    print(D_.round(4).to_string(index=False))
    print('\nvariance explained by the top PCs of the Enformer prediction matrix:')
    print((S ** 2 / (S ** 2).sum())[:5].round(4))
    D_.to_csv(OUT_DIR + 'pc_control.csv', index=False)


if __name__ == '__main__':
    main()
