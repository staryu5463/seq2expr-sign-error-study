"""
Population-structure control for the signed/unsigned decomposition.

GEUVADIS spans 5 populations. Ancestry shifts both the personal sequence and the
measured expression, so a model could score |r| > 0 across individuals purely by
tracking ancestry, with a per-gene sign that is essentially arbitrary. If that is
what drives the unsigned signal, removing population structure should collapse
mean |r| back to its permutation null.

Controls run here:
  (1) raw             -- no correction (reference)
  (2) pop-residual    -- population dummies regressed out of Y and of predictions
  (3) EUR-only        -- YRI dropped (removes the dominant ancestry axis / batch)
  (4) EUR pop-resid   -- both of the above
Each is compared against a permutation null computed on the same matrices.
"""
import sys
import numpy as np
import pandas as pd

PRED_DIR = sys.argv[1].rstrip('/') + '/'
PED = sys.argv[2]
OUT_DIR = sys.argv[3].rstrip('/') + '/'
N_PERM = int(sys.argv[4]) if len(sys.argv) > 4 else 100
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


def residualize(M, D):
    """Regress design matrix D out of every column of M (least squares)."""
    beta, *_ = np.linalg.lstsq(D, M, rcond=None)
    return M - D @ beta


def main():
    Y = pd.read_csv(PRED_DIR + 'geuvadis_peer_normalized.csv', index_col=0)
    ped = pd.read_csv(PED, sep='\t')
    pop = ped.set_index('Individual ID')['Population']
    pops = pop.reindex(Y.index)
    print('population counts over the %d GEUVADIS individuals:' % len(Y))
    print(pops.value_counts().to_string())
    print('unmapped:', int(pops.isna().sum()), '\n')

    eur_mask = (pops != 'YRI').to_numpy()
    rng = np.random.default_rng(SEED)

    rows = []
    for name, fn in MODELS.items():
        P = pd.read_csv(PRED_DIR + fn, index_col=0)
        genes = [g for g in Y.columns if g in P.columns]
        yv_all = Y[genes].to_numpy(float)
        pv_all = P[genes].to_numpy(float)
        keep = ~np.isnan(pv_all).any(0) & (np.nanstd(pv_all, 0) > 0)
        yv_all, pv_all = yv_all[:, keep], pv_all[:, keep]

        for setting in ['raw', 'pop-residual', 'EUR-only', 'EUR pop-resid']:
            sub = eur_mask if setting.startswith('EUR') else np.ones(len(Y), bool)
            yv, pv, pp = yv_all[sub], pv_all[sub], pops[sub]
            if 'resid' in setting:
                D = pd.get_dummies(pp, drop_first=False).to_numpy(float)
                yv = residualize(yv, D)
                pv = residualize(pv, D)
            n = yv.shape[0]
            r_crit = 1.959964 / np.sqrt(n - 1)
            r = colwise_spearman(yv, pv)

            na, nf = np.empty(N_PERM), np.empty(N_PERM)
            for i in range(N_PERM):
                rn = colwise_spearman(yv, pv[rng.permutation(n)])
                na[i] = np.nanmean(np.abs(rn))
                nf[i] = np.nanmean(np.abs(rn) > r_crit)
            sig = np.abs(r) > r_crit
            rows.append(dict(
                model=name, setting=setting, n=n,
                mean_r=np.nanmean(r), mean_abs_r=np.nanmean(np.abs(r)),
                null_abs_r=na.mean(), ratio_abs=np.nanmean(np.abs(r)) / na.mean(),
                z_abs=(np.nanmean(np.abs(r)) - na.mean()) / na.std(),
                frac_sig=sig.mean(), null_frac_sig=nf.mean(),
                frac_sig_neg=(r[sig] < 0).mean() if sig.any() else np.nan,
            ))
        print('%-10s done' % name)

    res = pd.DataFrame(rows)
    res.to_csv(OUT_DIR + 'ancestry_control.csv', index=False)
    pd.set_option('display.width', 220)
    print('\n=== unsigned signal vs its own permutation null, under each control ===')
    print(res[['model', 'setting', 'n', 'mean_r', 'mean_abs_r', 'null_abs_r',
               'ratio_abs', 'z_abs']].round(4).to_string(index=False))
    print('\n=== fraction of genes with significant |r|, and how many point the wrong way ===')
    print(res[['model', 'setting', 'frac_sig', 'null_frac_sig', 'frac_sig_neg']]
          .round(4).to_string(index=False))


if __name__ == '__main__':
    main()
