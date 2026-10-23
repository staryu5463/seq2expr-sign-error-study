"""
The flip classifier does not correct signs, but ranking genes by its score does
separate reliable genes from unreliable ones. What carries that information?

Each Y-free feature is used on its own as a triage score: keep the half of genes
it ranks as most reliable, and measure the mean across-individual Spearman on the
kept half. A random 50% split is the null. Direction of each score is chosen on
training chromosomes only, so the ranking stays honest under leave-one-chromosome-out.
"""
import sys
import numpy as np
import pandas as pd

PRED_DIR = sys.argv[1].rstrip('/') + '/'
GENE_LIST = sys.argv[2]
OUT = sys.argv[3].rstrip('/') + '/'
SEED = 0

MODELS = {
    'Enformer': 'enformer_sample_preds.csv',
    'Basenji2': 'basenji2_sample_preds.csv',
    'ExPecto': 'expecto_sample_preds.csv',
    'Xpresso': 'xpresso_sample_preds.csv',
}


def colwise_spearman(A, B):
    ra = pd.DataFrame(A).rank(axis=0).to_numpy()
    rb = pd.DataFrame(B).rank(axis=0).to_numpy()
    ra = ra - ra.mean(0); rb = rb - rb.mean(0)
    num = (ra * rb).sum(0)
    den = np.sqrt((ra ** 2).sum(0) * (rb ** 2).sum(0))
    with np.errstate(invalid='ignore', divide='ignore'):
        return num / den


def main():
    rng = np.random.default_rng(SEED)
    Y = pd.read_csv(PRED_DIR + 'geuvadis_peer_normalized.csv', index_col=0)
    n = len(Y)
    r_crit = 1.959964 / np.sqrt(n - 1)
    gl = pd.read_csv(GENE_LIST, header=None,
                     names=['ensg', 'chrom', 'tss', 'symbol', 'strand'])
    gl['symbol'] = gl['symbol'].str.lower()
    gl = gl.drop_duplicates('symbol').set_index('symbol')

    P = {m: pd.read_csv(PRED_DIR + f, index_col=0) for m, f in MODELS.items()}
    genes = sorted(set(Y.columns).intersection(*[set(p.columns) for p in P.values()])
                   & set(gl.index))
    yv = Y[genes].to_numpy(float)
    pv = {m: P[m][genes].to_numpy(float) for m in MODELS}

    F = pd.DataFrame(index=genes)
    for m in MODELS:
        a = pv[m]
        F[f'{m}_std'] = np.log1p(a.std(0))
        F[f'{m}_iqr'] = np.log1p(np.percentile(a, 95, axis=0) - np.percentile(a, 5, axis=0))
        F[f'{m}_nuniq'] = np.log1p([len(np.unique(a[:, i])) for i in range(a.shape[1])])
    mods = list(MODELS)
    for i in range(len(mods)):
        for j in range(i + 1, len(mods)):
            F[f'agree_{mods[i]}_{mods[j]}'] = np.abs(colwise_spearman(pv[mods[i]], pv[mods[j]]))
    chrom = gl['chrom'].reindex(genes).astype(str).to_numpy()

    for target in ['Enformer', 'Basenji2']:
        r = colwise_spearman(yv, pv[target])
        sel = (np.abs(r) > r_crit) & np.isfinite(r)
        rs = r[sel]
        g = chrom[sel]
        base = rs.mean()

        null = []
        for _ in range(500):
            k = rng.permutation(len(rs))[:len(rs) // 2]
            null.append(rs[k].mean())
        null = np.array(null)

        print('\n=== %s | %d genes | baseline mean r = %.4f | random-50%% null %.4f +- %.4f ==='
              % (target, sel.sum(), base, null.mean(), null.std()))
        print('%-26s %8s %8s %7s' % ('triage score (top half)', 'mean_r', 'delta', 'z'))
        out = []
        for col in F.columns:
            v = F.loc[:, col].to_numpy(float)[sel]
            v = np.nan_to_num(v, nan=np.nanmedian(v))
            # pick the direction on training chromosomes, apply to the held-out one
            keep = np.zeros(len(rs), bool)
            for c in np.unique(g):
                te, tr = g == c, g != c
                if te.sum() < 5 or tr.sum() < 50:
                    continue
                hi = rs[tr][v[tr] > np.median(v[tr])].mean()
                lo = rs[tr][v[tr] <= np.median(v[tr])].mean()
                thr = np.median(v[tr])
                keep[te] = (v[te] > thr) if hi >= lo else (v[te] <= thr)
            if keep.sum() < 50:
                continue
            mr = rs[keep].mean()
            out.append((col, mr, mr - base, (mr - null.mean()) / null.std(), int(keep.sum())))
        out.sort(key=lambda t: -t[1])
        for col, mr, d, z, k in out:
            print('%-26s %8.4f %+8.4f %+7.2f  (n=%d)' % (col, mr, d, z, k))
        pd.DataFrame(out, columns=['feature', 'mean_r', 'delta', 'z', 'n_kept']) \
            .to_csv(OUT + f'triage_{target}.csv', index=False)


if __name__ == '__main__':
    main()
