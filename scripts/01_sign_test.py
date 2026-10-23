"""
Across-individual information content of sequence-to-expression models, separated
from the sign of the predicted effect.

Question: Huang et al. 2023 (Nat Genet) report mean across-individual Spearman
r = 0.045 for Enformer on GEUVADIS. Sasse et al. 2023 (Nat Genet) separately
report that Enformer gets the *direction* of variant effects wrong. If both are
true, then |r| should be far above its permutation null while r itself is near
zero -- i.e. the information is present but sign-scrambled.

This script measures that directly.
"""
import sys
import numpy as np
import pandas as pd

PRED_DIR = sys.argv[1].rstrip('/') + '/'
OUT_DIR = sys.argv[2].rstrip('/') + '/'
N_PERM = int(sys.argv[3]) if len(sys.argv) > 3 else 200
SEED = 0

MODELS = {
    'Enformer': 'enformer_sample_preds.csv',
    'Basenji2': 'basenji2_sample_preds.csv',
    'ExPecto': 'expecto_sample_preds.csv',
    'Xpresso': 'xpresso_sample_preds.csv',
    'PrediXcan': 'predixcan_197KB_sample_preds.csv',
}


def colwise_spearman(A, B):
    """Spearman r per column of two (n_indiv, n_gene) arrays."""
    ra = pd.DataFrame(A).rank(axis=0).to_numpy()
    rb = pd.DataFrame(B).rank(axis=0).to_numpy()
    ra = ra - ra.mean(0)
    rb = rb - rb.mean(0)
    num = (ra * rb).sum(0)
    den = np.sqrt((ra ** 2).sum(0) * (rb ** 2).sum(0))
    with np.errstate(invalid='ignore', divide='ignore'):
        return num / den


def main():
    Y = pd.read_csv(PRED_DIR + 'geuvadis_peer_normalized.csv', index_col=0)
    n = Y.shape[0]
    # critical |r| for a two-sided test at alpha=0.05, no multiple-testing correction
    r_crit = 1.959964 / np.sqrt(n - 1)
    print('n individuals = %d   |r| critical (p<0.05, uncorrected) = %.4f' % (n, r_crit))
    print('analytic E|r| under the null = %.4f\n' % (np.sqrt(2 / np.pi) / np.sqrt(n - 1)))

    rng = np.random.default_rng(SEED)
    perms = [rng.permutation(n) for _ in range(N_PERM)]

    rows = []
    per_gene = {}
    for name, fn in MODELS.items():
        P = pd.read_csv(PRED_DIR + fn, index_col=0)
        genes = [g for g in Y.columns if g in P.columns]
        yv = Y[genes].to_numpy(float)
        pv = P[genes].to_numpy(float)
        keep = ~np.isnan(pv).any(0) & (np.nanstd(pv, 0) > 0)
        genes = [g for g, k in zip(genes, keep) if k]
        yv, pv = yv[:, keep], pv[:, keep]

        r = colwise_spearman(yv, pv)
        per_gene[name] = pd.Series(r, index=genes)

        # permutation null: shuffle individual labels of the prediction matrix.
        # One shuffle is applied to all genes at once, so gene-gene correlation
        # structure in the predictions is preserved; only the individual-level
        # correspondence to expression is destroyed.
        null_absmean = np.empty(N_PERM)
        null_frac_sig = np.empty(N_PERM)
        for i, pm in enumerate(perms):
            rn = colwise_spearman(yv, pv[pm])
            null_absmean[i] = np.nanmean(np.abs(rn))
            null_frac_sig[i] = np.nanmean(np.abs(rn) > r_crit)

        sig = np.abs(r) > r_crit
        rows.append(dict(
            model=name, n_genes=len(genes),
            mean_r=np.nanmean(r), median_r=np.nanmedian(r),
            mean_abs_r=np.nanmean(np.abs(r)),
            null_mean_abs_r=null_absmean.mean(),
            null_mean_abs_r_sd=null_absmean.std(),
            z_abs=(np.nanmean(np.abs(r)) - null_absmean.mean()) / null_absmean.std(),
            frac_sig=sig.mean(),
            null_frac_sig=null_frac_sig.mean(),
            frac_sig_negative=(r[sig] < 0).mean(),
            mean_r_among_sig=np.nanmean(r[sig]),
            mean_abs_r_among_sig=np.nanmean(np.abs(r[sig])),
        ))
        print('%-10s done (%d genes)' % (name, len(genes)))

    res = pd.DataFrame(rows)
    pd.DataFrame(per_gene).to_csv(OUT_DIR + 'per_gene_r.csv')
    res.to_csv(OUT_DIR + 'sign_test_summary.csv', index=False)

    pd.set_option('display.width', 200)
    print('\n=== across-individual Spearman, signed vs unsigned ===')
    print(res[['model', 'n_genes', 'mean_r', 'median_r', 'mean_abs_r',
               'null_mean_abs_r', 'z_abs']].round(4).to_string(index=False))
    print('\n=== how many genes carry a significant signal, and which way does it point ===')
    print(res[['model', 'frac_sig', 'null_frac_sig', 'frac_sig_negative',
               'mean_r_among_sig', 'mean_abs_r_among_sig']].round(4).to_string(index=False))


if __name__ == '__main__':
    main()
