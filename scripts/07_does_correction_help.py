"""
End-to-end test of the flip classifier: does acting on it actually help?

An AUC above chance is not enough. A classifier can score above chance by finding
genes where the model is *unreliable* (flip probability near 0.5) rather than genes
where it is *reliably inverted* (flip probability near 1). Only the second is a sign
correction. The first just finds noise, and flipping those genes gains nothing.

Three policies are applied to the out-of-fold flip probabilities and scored by the
metric the field actually reports -- mean across-individual Spearman:

  flip-all      negate the prediction wherever p_flip > 0.5
  flip-confident negate only where p_flip > a high threshold
  filter        drop high-p_flip genes instead of flipping them (reliability triage)

and compared against the uncorrected baseline and the sign-oracle upper bound.
"""
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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
    Y = pd.read_csv(PRED_DIR + 'geuvadis_peer_normalized.csv', index_col=0)
    n = len(Y)
    r_crit = 1.959964 / np.sqrt(n - 1)
    gl = pd.read_csv(GENE_LIST, header=None,
                     names=['ensg', 'chrom', 'tss', 'symbol', 'strand'])
    gl['symbol'] = gl['symbol'].str.lower()
    gl = gl.drop_duplicates('symbol').set_index('symbol')
    ref = pd.read_csv(PRED_DIR + 'all_models_ref_preds.csv')
    ref['gene'] = ref['gene'].str.lower()
    ref = ref.drop_duplicates('gene').set_index('gene')

    P = {m: pd.read_csv(PRED_DIR + f, index_col=0) for m, f in MODELS.items()}
    genes = sorted(set(Y.columns).intersection(*[set(p.columns) for p in P.values()])
                   & set(gl.index))
    yv = Y[genes].to_numpy(float)
    pv = {m: P[m][genes].to_numpy(float) for m in MODELS}
    r_all = {m: pd.Series(colwise_spearman(yv, pv[m]), index=genes) for m in MODELS}

    F = pd.DataFrame(index=genes)
    for m in MODELS:
        a = pv[m]
        F[f'{m}_std'] = np.log1p(a.std(0))
        F[f'{m}_iqr'] = np.log1p(np.percentile(a, 95, axis=0) - np.percentile(a, 5, axis=0))
        F[f'{m}_mean'] = a.mean(0)
        F[f'{m}_nuniq'] = np.log1p([len(np.unique(a[:, i])) for i in range(a.shape[1])])
        if m in ref.columns:
            F[f'{m}_ref'] = ref[m].reindex(genes).to_numpy()
    mods = list(MODELS)
    for i in range(len(mods)):
        for j in range(i + 1, len(mods)):
            F[f'agree_{mods[i]}_{mods[j]}'] = np.abs(colwise_spearman(pv[mods[i]], pv[mods[j]]))
    F['tss'] = gl['tss'].reindex(genes).to_numpy()
    chrom = gl['chrom'].reindex(genes).astype(str).to_numpy()
    Xall = np.nan_to_num(F.to_numpy(float), nan=0., posinf=0., neginf=0.)

    print('%-10s %-22s %8s %8s %8s' % ('model', 'policy', 'mean_r', 'n_genes', 'delta'))
    rows = []
    for target in ['Enformer', 'Basenji2']:
        rr = r_all[target]
        sel = (rr.abs() > r_crit).to_numpy() & np.isfinite(rr.to_numpy())
        y = (rr.to_numpy()[sel] < 0).astype(int)
        X, g, rsel = Xall[sel], chrom[sel], rr.to_numpy()[sel]

        oof = np.full(len(y), np.nan)
        for c in np.unique(g):
            te = g == c; tr = ~te
            if y[tr].sum() < 10 or (1 - y[tr]).sum() < 10 or te.sum() < 10:
                continue
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=2000, C=0.5, random_state=SEED))
            clf.fit(X[tr], y[tr])
            oof[te] = clf.predict_proba(X[te])[:, 1]
        ok = np.isfinite(oof)
        rs, pf, ys = rsel[ok], oof[ok], y[ok]

        base = rs.mean()
        oracle = np.abs(rs).mean()
        print('%-10s %-22s %8.4f %8d %8s' % (target, 'baseline', base, len(rs), '-'))
        rows.append(dict(model=target, policy='baseline', mean_r=base, n=len(rs), delta=0.0))

        for thr in [0.5, 0.6, 0.7]:
            corr = np.where(pf > thr, -rs, rs)
            rows.append(dict(model=target, policy='flip if p>%.1f' % thr,
                             mean_r=corr.mean(), n=len(rs), delta=corr.mean() - base))
            print('%-10s %-22s %8.4f %8d %+8.4f'
                  % (target, 'flip if p>%.1f' % thr, corr.mean(), len(rs), corr.mean() - base))

        for q in [0.75, 0.5]:
            t = np.quantile(pf, q)
            keep = pf <= t
            rows.append(dict(model=target, policy='keep lowest %d%% p' % int(q * 100),
                             mean_r=rs[keep].mean(), n=int(keep.sum()),
                             delta=rs[keep].mean() - base))
            print('%-10s %-22s %8.4f %8d %+8.4f'
                  % (target, 'keep lowest %d%% p' % int(q * 100), rs[keep].mean(),
                     keep.sum(), rs[keep].mean() - base))

        rows.append(dict(model=target, policy='sign oracle (bound)', mean_r=oracle,
                         n=len(rs), delta=oracle - base))
        print('%-10s %-22s %8.4f %8d %+8.4f'
              % (target, 'sign oracle (bound)', oracle, len(rs), oracle - base))

        # is the classifier finding inverted genes, or merely unreliable ones?
        bins = pd.qcut(pf, 5, labels=False, duplicates='drop')
        print('\n  what the classifier is actually tracking (%s):' % target)
        print('  %-8s %8s %10s %10s' % ('p_flip Q', 'flip rate', 'mean |r|', 'mean r'))
        for b in range(bins.max() + 1):
            m = bins == b
            print('  %-8d %8.3f %10.4f %10.4f'
                  % (b + 1, ys[m].mean(), np.abs(rs[m]).mean(), rs[m].mean()))
        print()

    pd.DataFrame(rows).to_csv(OUT + 'correction_policies.csv', index=False)


if __name__ == '__main__':
    main()
