# Half of the personal-genome failure of sequence-to-expression models is a sign error

Sequence-to-expression models (Enformer, Basenji2, ExPecto, Xpresso) are known to do
badly at explaining expression *across individuals*: Huang et al. 2023 report a mean
across-individual Spearman of **0.045** for Enformer on GEUVADIS, versus 0.57 across
genes. The usual reading is that the models do not read personal sequence at all.

Separating the **magnitude** of the across-individual correlation from its **sign**
says something different.

![summary figure](results/figure.png)

## Result

On GEUVADIS (421 individuals x 3259 genes):

| model | signed mean *r* | unsigned mean \|*r*\| | locus-swap null | sign accuracy | signed / unsigned |
|---|---|---|---|---|---|
| Enformer | 0.0453 | **0.1471** | 0.0414 | **65.4 %** | 0.357 |
| Basenji2 | 0.0374 | 0.1468 | 0.0413 | 62.0 % | 0.291 |
| ExPecto | 0.0338 | 0.1437 | 0.0413 | 61.3 % | 0.264 |
| Xpresso | 0.0068 | 0.1362 | 0.0411 | 52.4 % (chance) | 0.061 |
| PrediXcan | 0.2485 | 0.2605 | 0.0404 | 96.4 % | 0.959 |

The signed mean reproduces the published 0.045. The unsigned mean is **3.55x its
locus-swap null**. At BH-FDR 5 %, **54.5 % of genes carry a real across-individual
signal — and 34.7 % of those point the wrong way.**

So the models are not blind to personal sequence. They localise the signal to the
right locus and then get the direction wrong about a third of the time.

Sign accuracy is monotone in model class: Enformer 65.4 % > Basenji2 62.0 % >
ExPecto 61.3 % > Xpresso 52.4 %. Xpresso only sees the promoter, and its sign sits
exactly at chance — an internal consistency check on the measure.

If the signs were corrected, Enformer would move 0.045 -> 0.147, closing about **half**
the gap to the PrediXcan elastic-net baseline. Sign is roughly half the problem;
effect-size calibration is the other half.

## Controls

| control | what it does | outcome |
|---|---|---|
| population structure | regress out 5 population dummies; separately drop YRI (EUR-only) | ratio 3.78 -> 3.80 / 3.76 / 3.80, unchanged |
| global individual factor | regress out top 1 / 5 / 20 individual-level PCs of the prediction matrix | 3.75 -> 3.81 -> 3.82 -> 3.76, unchanged |
| **locus specificity** | pair gene *g*'s expression with the prediction for a gene *g'* on a different chromosome | \|*r*\| collapses 0.147 -> 0.0414 (label-permutation null 0.0390) |

The third is the decisive one: it keeps every individual's identity and breaks only the
gene-to-locus pairing. The signal collapsing to the null means the models really are
responding to sequence at the correct locus.

## Where the sign error comes from

**Ruled out — a fixed polarity prior.** Flip rate does not depend on the true eQTL
effect direction (Enformer: 36.3 % when eQTL *r* > 0, 34.0 % when *r* < 0, overlapping
CIs). The easiest thing to fix is not the problem.

**Partly supported — distance.** Within 0–20 kb the flip rate rises with lead-eQTL
distance to the TSS (Basenji2 31 % -> 45 %, non-overlapping CIs; Enformer 32 % -> 40 %).
Xpresso is flat at ~47–49 % everywhere, as a promoter-only model should be.
Caveat: the eQTL annotation used here tops out at 19,977 bp, so the genuinely distal
regime that Karollus et al. 2023 describe is **not** tested.

**Strongly supported — the error is shared across architectures.** Among the 822 genes
significant in all four models:

| models flipping | observed | expected if independent |
|---|---|---|
| 0 | 192 | 119.8 |
| 1 | 252 | 298.5 |
| 2 | 201 | 275.4 |
| 3 | 111 | 111.6 |
| **4** | **66** | **16.8** |

All four flipping together is **3.9x** the independence expectation; pairwise sign
concordance Enformer-Basenji2 is **0.710** (chance 0.5). A transformer at 196 kb, a
dilated CNN at 131 kb and a 40 kb CNN fail on the *same genes*. The sign error is a
property of the training paradigm, not of the architecture — which is a quantitative
version of the qualitative observation in the literature that fine-tuned Borzoi and
fine-tuned Enformer land in the same place.

## Limitations

1. PrediXcan was fit on this cohort, so the "closes half the gap" figure inherits that
   optimism. The main result does not depend on PrediXcan.
2. The eQTL annotation caps at 20 kb; the distal regime is untested.
3. One cohort, one cell type (LCL), and the 3259 genes were preselected for having a
   significant cis-eQTL, so absolute fractions do not extrapolate genome-wide.
4. A "sign oracle" is not a method. \|*r*\| is an upper bound, not achievable performance.
5. Only the 2021–2023 model generation is covered. Whether Borzoi or AlphaGenome still
   has this is untested here.

## Next

Is the sign **predictable** from features available at test time (lead-eQTL distance,
MAF, local sequence context, the model's own attributions)? If a flip classifier beats
AUC 0.55 on held-out chromosomes, it is a post-hoc sign correction that needs no
retraining. If it does not, that line is dead. Either way it is a small experiment.

## Reproducing

Everything runs on CPU in a few minutes. No GPU, no genome download.

```bash
git clone https://github.com/ni-lab/personalized-expression-benchmark.git upstream
curl -o g1k.ped https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/technical/working/20130606_sample_info/20130606_g1k.ped

P=upstream/predictions; D=upstream/data; R=results
python -I scripts/01_sign_test.py        $P $R 200
python -I scripts/02_ancestry_control.py $P g1k.ped $R 100
python -I scripts/03_mechanism.py        $P $D/eqtl_features.csv $R
python -I scripts/04_locus_swap.py       $P $D/gene_list.csv $D/eqtl_features.csv $R 30
python -I scripts/05_summary_figure.py   $R
```

Predictions, ground truth and the eQTL annotation all ship inside the upstream repo.

## Sources

- Huang et al., *Personal transcriptome variation is poorly explained by current genomic deep learning models*, Nat Genet 55, 2056 (2023). [paper](https://www.nature.com/articles/s41588-023-01574-w) · [data & code](https://github.com/ni-lab/personalized-expression-benchmark)
- Sasse et al., *Benchmarking of deep neural networks for predicting personal gene expression from DNA sequence highlights shortcomings*, Nat Genet 55, 2060 (2023). [paper](https://www.nature.com/articles/s41588-023-01524-6)
- Karollus, Mauermeier & Gagneur, *Current sequence-based models capture gene expression determinants in promoters but mostly ignore distal enhancers*, Genome Biol 24, 56 (2023). [paper](https://doi.org/10.1186/s13059-023-02899-9)
- Spiro et al., *A scalable approach to investigating sequence-to-function predictions from personal genomes*, Nat Methods 23, 1308 (2026). [paper](https://www.nature.com/articles/s41592-026-03124-8)

A Chinese write-up of the same material is in [REPORT.zh.md](REPORT.zh.md).
