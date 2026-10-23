"""
Pre-registered falsification test: is the sign error predictable at test time?

If a classifier can tell, for a held-out chromosome, which genes Enformer will
flip, then the sign can be corrected post hoc with no retraining and Enformer's
across-individual r moves from 0.045 toward 0.147. The README commits to
AUC > 0.55 on held-out chromosomes as the bar; below that the line is dead.

Two feature sets:
  honest -- computable without ever seeing the test gene's expression:
            dispersion of the model's own predictions across individuals, the
            reference prediction, and how much the four architectures agree with
            each other on this gene (agreement is sign-symmetric, so it cannot by
            itself reveal the sign -- it is in as a difficulty proxy).
  leaky  -- adds lead cis-eQTL distance / MAF / strength. Those were derived from
            this same expression matrix, so this variant is an optimistic bound,
            not a usable method. Reported separately and labelled as such.

Validation is leave-one-chromosome-out. The null is the same pipeline with the
labels shuffled within each fold.
"""
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

PRED_DIR = sys.argv[1].rstrip('/') + '/'
GENE_LIST = sys.argv[2]
EQTL = sys.argv[3]
OUT = sys.argv[4].rstrip('/') + '/'
SEED = 0

MODELS = {
    'Enformer': 'enformer_sample_preds.csv',
    'Basenji2': 'basenji2_sample_preds.csv',
    'ExPecto': 'expecto_sample_preds.csv',
    'Xpresso': 'xpresso_sample_preds.csv',
}
TARGETS = ['Enformer', 'Basenji2']


def colwise_spearman(A, B):
    ra = pd.DataFrame(A).rank(axis=0).to_numpy()
    rb = pd.DataFrame(B).rank(axis=0).to_numpy()
    ra = ra - ra.mean(0); rb = rb - rb.mean(0)
    num = (ra * rb).sum(0)
    den = np.sqrt((ra ** 2).sum(0) * (rb ** 2).sum(0))
    with np.errstate(invalid='ignore', divide='ignore'):
        return num / den


def loco_auc(X, y, groups, clf_factory, rng, n_null=20):
    """Leave-one-chromosome-out AUC, pooled over folds, plus a shuffled-label null."""
    oof = np.full(len(y), np.nan)
    for g in np.unique(groups):
        te = groups == g
        tr = ~te
        if y[tr].sum() < 10 or (1 - y[tr]).sum() < 10 or te.sum() < 10:
            continue
        clf = clf_factory()
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    ok = ~np.isnan(oof)
    auc = roc_auc_score(y[ok], oof[ok]) if len(np.unique(y[ok])) > 1 else np.nan

    nulls = []
    for _ in range(n_null):
        ys = y.copy()
        for g in np.unique(groups):
            m = groups == g
            ys[m] = rng.permutation(ys[m])
        o = np.full(len(y), np.nan)
        for g in np.unique(groups):
            te = groups == g; tr = ~te
            if ys[tr].sum() < 10 or (1 - ys[tr]).sum() < 10 or te.sum() < 10:
                continue
            c = clf_factory(); c.fit(X[tr], ys[tr])
            o[te] = c.predict_proba(X[te])[:, 1]
        k = ~np.isnan(o)
        if len(np.unique(ys[k])) > 1:
            nulls.append(roc_auc_score(ys[k], o[k]))
    return auc, np.mean(nulls), np.std(nulls), int(ok.sum())


def main():
    rng = np.random.default_rng(SEED)
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

    eq = pd.read_csv(EQTL).dropna(subset=['name', 'distance'])
    eq['name'] = eq['name'].str.lower()
    eq = eq.drop_duplicates('name').set_index('name')

    P = {m: pd.read_csv(PRED_DIR + f, index_col=0) for m, f in MODELS.items()}
    genes = sorted(set(Y.columns).intersection(*[set(p.columns) for p in P.values()])
                   & set(gl.index))
    print('genes usable: %d' % len(genes))

    yv = Y[genes].to_numpy(float)
    pv = {m: P[m][genes].to_numpy(float) for m in MODELS}
    r = {m: pd.Series(colwise_spearman(yv, pv[m]), index=genes) for m in MODELS}

    # ---- Y-free features ----
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
            F[f'agree_{mods[i]}_{mods[j]}'] = np.abs(
                colwise_spearman(pv[mods[i]], pv[mods[j]]))
    F['tss'] = gl['tss'].reindex(genes).to_numpy()
    honest_cols = list(F.columns)

    F['eqtl_dist'] = np.log1p(eq['distance'].abs().reindex(genes).to_numpy())
    F['eqtl_maf'] = eq['minor_allele_freq'].reindex(genes).to_numpy()
    F['eqtl_log10p'] = eq['log10pvalue'].reindex(genes).to_numpy()
    leaky_cols = honest_cols + ['eqtl_dist', 'eqtl_maf', 'eqtl_log10p']

    chrom = gl['chrom'].reindex(genes).astype(str).to_numpy()

    defs = {
        'HGB': lambda: HistGradientBoostingClassifier(
            max_iter=200, max_depth=4, learning_rate=0.06, random_state=SEED),
        'logreg': lambda: make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, C=0.5, random_state=SEED)),
    }

    rows = []
    for target in TARGETS:
        rr = r[target]
        sel = (rr.abs() > r_crit).to_numpy() & np.isfinite(rr.to_numpy())
        y = (rr.to_numpy()[sel] < 0).astype(int)
        print('\n--- target: %s | %d significant genes | flip rate %.3f ---'
              % (target, sel.sum(), y.mean()))
        for fs_name, cols in [('honest', honest_cols), ('leaky(+eQTL)', leaky_cols)]:
            X = F.loc[:, cols].to_numpy(float)[sel]
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            g = chrom[sel]
            for clf_name, fac in defs.items():
                auc, nm, nsd, cov = loco_auc(X, y, g, fac, rng)
                z = (auc - nm) / nsd if nsd > 0 else np.nan
                rows.append(dict(target=target, features=fs_name, clf=clf_name,
                                 loco_auc=auc, null_auc=nm, null_sd=nsd, z=z,
                                 n=cov, n_feat=len(cols)))
                print('  %-13s %-7s AUC %.4f   null %.4f +- %.4f   z %+.2f'
                      % (fs_name, clf_name, auc, nm, nsd, z))

    res = pd.DataFrame(rows)
    res.to_csv(OUT + 'flip_classifier.csv', index=False)

    best = res[res.features == 'honest']['loco_auc'].max()
    print('\n=== verdict ===')
    print('best honest leave-one-chromosome-out AUC: %.4f  (pre-registered bar: 0.55)' % best)
    print('ALIVE: post-hoc sign correction is learnable' if best > 0.55
          else 'DEAD: the sign is not predictable from test-time-available features')


if __name__ == '__main__':
    main()
