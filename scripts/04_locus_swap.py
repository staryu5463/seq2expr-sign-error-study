"""
Decisive control: is the unsigned across-individual signal locus-specific?

The permutation null used so far shuffles individual labels, which destroys every
individual-level correspondence at once. A sharper null keeps each individual's
identity intact and instead breaks the pairing between a gene's expression and
the model's prediction *for that gene*: correlate Y[:, g] with the prediction for
a different gene g' on a different chromosome.

If mean |r| stays near 0.147 under this locus-swap null, then the signal is some
individual-level structure shared across genes and carries no locus information.
If it collapses toward the label-permutation null, the model really is responding
to sequence at the right locus and only the sign is wrong.

Also reports the eQTL distance distribution and a BH-FDR version of the gene
counts, so the headline does not rest on an uncorrected threshold.
"""
import sys
import numpy as np
import pandas as pd

PRED_DIR = sys.argv[1].rstrip('/') + '/'
GENE_LIST = sys.argv[2]
EQTL = sys.argv[3]
OUT_DIR = sys.argv[4].rstrip('/') + '/'
N_SWAP = int(sys.argv[5]) if len(sys.argv) > 5 else 30
SEED = 0

MODELS = {
    'Enformer': 'enformer_sample_preds.csv',
    'Basenji2': 'basenji2_sample_preds.csv',
    'ExPecto': 'expecto_sample_preds.csv',
    'Xpresso': 'xpresso_sample_preds.csv',
    'PrediXcan': 'predixcan_197KB_sample_preds.csv',
}


def colwise_spearman(A, B):
    ra = pd.DataFrame(A).rank(axis=0).to_numpy()
    rb = pd.DataFrame(B).rank(axis=0).to_numpy()
    ra = ra - ra.mean(0)
    rb = rb - rb.mean(0)
    num = (ra * rb).sum(0)
    den = np.sqrt((ra ** 2).sum(0) * (rb ** 2).sum(0))
    with np.errstate(invalid='ignore', divide='ignore'):
        return num / den


def bh_fdr(p):
    """Benjamini-Hochberg q-values. NaN p-values are carried through as NaN
    instead of poisoning the reverse cumulative minimum."""
    p = np.asarray(p, float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    pv = p[ok]
    m = pv.size
    if m == 0:
        return out
    o = np.argsort(pv)
    q = np.empty(m)
    q[o] = np.minimum.accumulate((pv[o] * m / np.arange(1, m + 1))[::-1])[::-1]
    out[ok] = np.minimum(q, 1.0)
    return out


def main():
    Y = pd.read_csv(PRED_DIR + 'geuvadis_peer_normalized.csv', index_col=0)
    n = len(Y)
    r_crit = 1.959964 / np.sqrt(n - 1)

    gl = pd.read_csv(GENE_LIST, header=None,
                     names=['ensg', 'chrom', 'tss', 'symbol', 'strand'])
    gl['symbol'] = gl['symbol'].str.lower()
    chrom = gl.drop_duplicates('symbol').set_index('symbol')['chrom'].astype(str)

    eq = pd.read_csv(EQTL).dropna(subset=['name', 'distance'])
    d = eq['distance'].abs()
    print('lead cis-eQTL distance to TSS, bp:')
    print(d.describe(percentiles=[.1, .25, .5, .75, .9, .99]).round(0).to_string())
    print('fraction beyond 50kb: %.4f | beyond 100kb: %.4f\n' % ((d > 5e4).mean(), (d > 1e5).mean()))

    rng = np.random.default_rng(SEED)
    rows = []
    for name, fn in MODELS.items():
        P = pd.read_csv(PRED_DIR + fn, index_col=0)
        genes = [g for g in Y.columns if g in P.columns]
        yv = Y[genes].to_numpy(float)
        pv = P[genes].to_numpy(float)
        keep = ~np.isnan(pv).any(0) & (np.nanstd(pv, 0) > 0)
        genes = [g for g, k in zip(genes, keep) if k]
        yv, pv = yv[:, keep], pv[:, keep]
        ch = chrom.reindex(genes).to_numpy()

        r = colwise_spearman(yv, pv)
        # two-sided p from Fisher z on the Spearman r
        z = np.arctanh(np.clip(r, -0.999999, 0.999999)) * np.sqrt(n - 3)
        from math import erfc
        p = np.array([erfc(abs(v) / np.sqrt(2)) for v in z])
        q = bh_fdr(p)

        # locus-swap null: pair gene g's expression with the prediction for a
        # gene on a different chromosome
        swap_abs, swap_frac = [], []
        for _ in range(N_SWAP):
            perm = np.arange(len(genes))
            for attempt in range(60):
                perm = rng.permutation(len(genes))
                bad = (ch[perm] == ch)
                if bad.mean() < 0.02:
                    break
            ok = ch[perm] != ch
            rs = colwise_spearman(yv[:, ok], pv[:, perm][:, ok])
            swap_abs.append(np.nanmean(np.abs(rs)))
            swap_frac.append(np.nanmean(np.abs(rs) > r_crit))
        swap_abs = np.array(swap_abs)
        swap_frac = np.array(swap_frac)

        # label-permutation null for reference
        lab_abs = np.array([np.nanmean(np.abs(colwise_spearman(yv, pv[rng.permutation(n)])))
                            for _ in range(N_SWAP)])

        sig = np.abs(r) > r_crit
        sigq = q < 0.05
        rows.append(dict(
            model=name,
            mean_r=np.nanmean(r), mean_abs_r=np.nanmean(np.abs(r)),
            locus_swap_abs=swap_abs.mean(), locus_swap_sd=swap_abs.std(),
            label_perm_abs=lab_abs.mean(),
            ratio_vs_swap=np.nanmean(np.abs(r)) / swap_abs.mean(),
            frac_sig_p05=sig.mean(), locus_swap_frac_sig=swap_frac.mean(),
            frac_sig_fdr05=sigq.mean(),
            frac_neg_among_fdr=(r[sigq] < 0).mean() if sigq.any() else np.nan,
            mean_abs_r_among_fdr=np.nanmean(np.abs(r[sigq])) if sigq.any() else np.nan,
            mean_r_among_fdr=np.nanmean(r[sigq]) if sigq.any() else np.nan,
        ))
        print('%-10s done' % name)

    res = pd.DataFrame(rows)
    res.to_csv(OUT_DIR + 'locus_swap.csv', index=False)
    pd.set_option('display.width', 240)
    print('\n=== is the unsigned signal locus-specific? ===')
    print(res[['model', 'mean_r', 'mean_abs_r', 'locus_swap_abs', 'label_perm_abs',
               'ratio_vs_swap']].round(4).to_string(index=False))
    print('\n=== gene counts: uncorrected vs BH-FDR 5% ===')
    print(res[['model', 'frac_sig_p05', 'locus_swap_frac_sig', 'frac_sig_fdr05',
               'frac_neg_among_fdr', 'mean_r_among_fdr', 'mean_abs_r_among_fdr']]
          .round(4).to_string(index=False))


if __name__ == '__main__':
    main()
