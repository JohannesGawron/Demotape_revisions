"""
Simulate a pool of three patients (pat0, pat1, pat2) with known ground truth, in
the CSV format DemoTape takes as input (Mosaic's filtered_variants.csv): columns
CHR,POS,REF,ALT,REGION,NAME,FREQ followed by one column per cell, named
`cell<i>.<label>` (e.g. `cell17.pat0+pat1`), with
entries `REF:ALT:GT` (GT is ignored by DemoTape and set to a constant).

pat1 and pat2 have the same genotype at exactly `n_same` sites and different
genotypes at exactly `n_diff` sites; pat0's genotype is random, but never equal to
both others (such sites would be filtered out).

Read depths are resampled from a real depth matrix (loci x cells, written by
`export_depth.py`), which may stack several libraries block by block, with
`nan` where a cell's library lacks a locus. Each simulated site takes the
depths of a real locus, each simulated cell those of one real cell per block.
This keeps the depth features a parametric model misses (see
notebooks/read_depth.qmd).

Alt reads at heterozygous entries follow a beta-binomial mixture with allelic
dropout (see notebooks/het_sites.qmd), with parameters (m, kappa, d, eps) drawn
per site from independent Gaussians on an unconstrained scale. Homozygous
entries are binomial with a per-site error rate (see notebooks/hom_sites.qmd).
"""

import argparse

import numpy as np
import numpy.typing as npt
import polars as pl

# Homozygous entries: per-site error rate e with logit(e) ~ Normal(HOM_ERROR_MU,
# HOM_ERROR_TAU), estimated at homozygous mutated sites (notebooks/hom_sites.qmd,
# "Distribution of the error rate across loci", row "all" / "hom mutated":
# 108 loci of the six libraries) and used for both homozygous genotypes:
# alt ~ Binom(depth, e) for genotype 0, Binom(depth, 1 - e) for genotype 2.
HOM_ERROR_MU = -6.3761
HOM_ERROR_TAU = 1.2881

# Heterozygous entries: per-site parameters (logit m, log kappa, logit d,
# logit 2 eps) ~ Normal(HET_MEAN, HET_SD), independently, fitted to the
# per-locus estimates in notebooks/het_sites.qmd ("Per-locus parameter
# distribution", pooled over the six libraries).
HET_MEAN = np.array([-0.017, 2.131, -3.344, -3.833])
HET_SD = np.array([0.14, 0.654, 1.531, 1.338])

# Mosaic variant filter: a site needs depth >= MIN_DEPTH in >= MIN_GENOTYPED_FRACTION of cells.
MIN_DEPTH = 10
MIN_GENOTYPED_FRACTION = 0.5

GT_PLACEHOLDER = 3

# Ranges to draw from when --cells_per_sample / --doublet_rate are not given
# (uniformly, per sample and per run respectively).
CELLS_PER_SAMPLE_RANGE = (500, 3000)
DOUBLET_RATE_RANGE = (0.05, 0.3)


def simulate_patient_genotypes(
    n_same: int, n_diff: int, rng: np.random.Generator
) -> npt.NDArray[np.int64]:
    """Return a (n_same + n_diff) x 3 genotype matrix (0 = wildtype, 1 = het, 2 = hom). The first n_same sites have equal genotypes in pat1 and pat2, the remaining n_diff sites different ones. No site is identical in all three patients."""
    same = rng.integers(0, 3, n_same)
    pat0_same = (same + rng.integers(1, 3, n_same)) % 3  # uniform over the other two

    pat1_diff = rng.integers(0, 3, n_diff)
    pat2_diff = (pat1_diff + rng.integers(1, 3, n_diff)) % 3
    pat0_diff = rng.integers(0, 3, n_diff)

    return np.column_stack(
        [
            np.concatenate([pat0_same, pat0_diff]),
            np.concatenate([same, pat1_diff]),
            np.concatenate([same, pat2_diff]),
        ]
    )


def load_depth(path: str) -> npt.NDArray[np.float64]:
    """Return a depth matrix written by `export_depth.py` as a loci x cells array (`nan` = missing)."""
    return np.loadtxt(path, delimiter=',', ndmin=2)


def blocks(
    real_depth: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Return the block of each cell and of each locus. Cells observing the same loci form a block; a locus belongs to the block of the cells observing it."""
    observed = ~np.isnan(real_depth)
    _, cell_block = np.unique(observed.T, axis=0, return_inverse=True)
    locus_block = cell_block[observed.argmax(axis=1)]
    return cell_block, locus_block


def expit(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Logistic function, the inverse of logit."""
    return 1 / (1 + np.exp(-x))


def draw_het_params(n_sites: int, rng: np.random.Generator) -> npt.NDArray[np.float64]:
    """Return (m, kappa, d, eps) per site as a 4 x n_sites array, drawn from the per-locus Gaussians."""
    logit_m, log_kappa, logit_d, logit_2eps = rng.normal(
        HET_MEAN[:, None], HET_SD[:, None], (4, n_sites)
    )
    return np.array(
        [
            expit(logit_m),
            np.exp(log_kappa),
            expit(logit_d),
            expit(logit_2eps) / 2,
        ]
    )


def draw_hom_error(n_sites: int, rng: np.random.Generator) -> npt.NDArray[np.float64]:
    """Return the error rate of homozygous entries per site, drawn from the per-locus logit-normal."""
    return expit(rng.normal(HOM_ERROR_MU, HOM_ERROR_TAU, n_sites))


def simulate_reads(
    genotypes: npt.NDArray[np.int64],
    depth: npt.NDArray[np.int64],
    het_params: npt.NDArray[np.float64],
    hom_error: npt.NDArray[np.float64],
    rng: np.random.Generator,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Simulate (ref, alt) read counts for a sites x cells genotype matrix, given the depth of each entry, the het parameters (m, kappa, d, eps) and the homozygous error rate of each site."""
    m, kappa, d, eps = (q[:, None] for q in het_params)
    # Het: with probability d one allele drops out (VAF eps or 1 - eps),
    # otherwise the VAF is Beta(m kappa, (1 - m) kappa).
    dropout = rng.random(depth.shape) < d
    het_vaf = np.where(
        dropout,
        np.where(rng.random(depth.shape) < 0.5, eps, 1 - eps),
        rng.beta(m * kappa, (1 - m) * kappa, depth.shape),
    )
    error = hom_error[:, None]
    vaf = np.choose(genotypes, [error, het_vaf, 1 - error])
    alt = rng.binomial(depth, vaf)
    return depth - alt, alt


def simulate_pool(
    n_same: int,
    n_diff: int,
    cells_per_sample: list[int],
    doublet_rate: float,
    real_depth: npt.ArrayLike,
    rng: np.random.Generator,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], list[str]]:
    """Simulate a pool and return the (ref, alt) read counts (sites x barcodes) and the ground-truth label per barcode. Depths are resampled from `real_depth` (loci x cells, `nan` where a cell's block lacks a locus)."""
    genotypes = simulate_patient_genotypes(n_same, n_diff, rng)

    singlets = np.repeat(np.arange(3), cells_per_sample)
    n_singlets = len(singlets)
    n_doublets = round(doublet_rate / (1 - doublet_rate) * n_singlets)
    mixing_ratio = np.array(cells_per_sample) / n_singlets
    doublets = np.sort(rng.choice(3, size=(n_doublets, 2), p=mixing_ratio), axis=1)

    pat_a = np.concatenate([singlets, doublets[:, 0]])
    pat_b = np.concatenate([singlets, doublets[:, 1]])
    is_doublet = np.arange(len(pat_a)) >= n_singlets

    # Each barcode gets one real cell per block (with replacement), doublets a
    # second, different one. Sites are drawn from the real loci that pass the
    # depth filter, among the cells observing them.
    real_depth = np.asarray(real_depth, dtype=np.float64)
    cell_block, locus_block = blocks(real_depth)
    real_cell_a = np.empty((len(pat_a), cell_block.max() + 1), dtype=np.int64)
    real_cell_b = np.empty_like(real_cell_a)
    for k in range(real_cell_a.shape[1]):
        cells = np.flatnonzero(cell_block == k)
        a = rng.integers(0, len(cells), len(pat_a))
        b = (a + rng.integers(1, len(cells), len(pat_b))) % len(cells)
        real_cell_a[:, k], real_cell_b[:, k] = cells[a], cells[b]
    observed = ~np.isnan(real_depth)
    real_loci = np.flatnonzero(
        (real_depth >= MIN_DEPTH).sum(axis=1) / observed.sum(axis=1)
        >= MIN_GENOTYPED_FRACTION
    )

    def simulate(rows: npt.NDArray[np.bool_]):
        # Every barcode gets a cell from pat_a and one from pat_b; singlets keep
        # only the first. Each simulated site gets the depths of a real locus
        # (with replacement), each barcode those of its real cell in the
        # locus's block. Het parameters and the homozygous error rate are drawn
        # per site.
        loci = rng.choice(real_loci, rows.sum())
        het_params = draw_het_params(rows.sum(), rng)
        hom_error = draw_hom_error(rows.sum(), rng)

        def depth(real_cell):
            cells = real_cell[:, locus_block[loci]].T  # sites x barcodes
            return real_depth[loci[:, None], cells].astype(np.int64)

        # Doublets as in the previous DemoTape paper: equal genotypes are kept,
        # any other combination becomes het; the depth is the rounded mean of
        # both cells. (Singlets have pat_a == pat_b.)
        genotype_a, genotype_b = genotypes[rows][:, pat_a], genotypes[rows][:, pat_b]
        genotype = np.where(genotype_a == genotype_b, genotype_a, 1)
        depth_a, depth_b = depth(real_cell_a), depth(real_cell_b)
        barcode_depth = np.where(is_doublet, np.rint((depth_a + depth_b) / 2), depth_a)
        return simulate_reads(
            genotype, barcode_depth.astype(np.int64), het_params, hom_error, rng
        )

    ref, alt = simulate(np.ones(len(genotypes), dtype=bool))
    # Re-simulate sites (with a new real locus) that would not pass the Mosaic
    # depth filter, e.g. because of doublet averaging.
    failing = ((ref + alt) >= MIN_DEPTH).mean(axis=1) < MIN_GENOTYPED_FRACTION
    while failing.any():
        ref[failing], alt[failing] = simulate(failing)
        failing = ((ref + alt) >= MIN_DEPTH).mean(axis=1) < MIN_GENOTYPED_FRACTION

    labels = [
        f'pat{a}+pat{b}' if doublet else f'pat{a}'
        for a, b, doublet in zip(pat_a, pat_b, is_doublet, strict=True)
    ]
    return ref, alt, labels


def write_csv(
    ref: npt.NDArray[np.int64],
    alt: npt.NDArray[np.int64],
    labels: list[str],
    output: str,
    rng: np.random.Generator,
) -> None:
    """Write the pool in DemoTape's input format, with synthetic site coordinates and numbered cells."""
    n_sites = ref.shape[0]
    bases = np.array(list('ACGT'))
    ref_idx = rng.integers(0, 4, n_sites)
    alt_idx = (ref_idx + rng.integers(1, 4, n_sites)) % 4  # any base but REF

    columns = {
        'CHR': ['1'] * n_sites,  # without 'chr', as in Mosaic's output
        'POS': np.arange(1, n_sites + 1) * 1000,
        'REF': bases[ref_idx],
        'ALT': bases[alt_idx],
        # Amplicon name and frequency, as in Mosaic's output; each simulated site
        # gets its own amplicon.
        'REGION': [f'AMPL{i + 1}' for i in range(n_sites)],
        'NAME': [f'AMPL{i + 1}' for i in range(n_sites)],
        'FREQ': [0] * n_sites,
    }
    for i, label in enumerate(labels):
        columns[f'cell{i}.{label}'] = [
            f'{r}:{a}:{GT_PLACEHOLDER}'
            for r, a in zip(ref[:, i], alt[:, i], strict=True)
        ]
    pl.DataFrame(columns).write_csv(output)


def parse_args():
    """Parse command-line arguments for the pool simulation."""
    parser = argparse.ArgumentParser(
        description='Simulate a pool of three patients in DemoTape input format.'
    )
    parser.add_argument(
        '--n_same',
        type=int,
        required=True,
        help='Number of sites where pat1 and pat2 have the same genotype.',
    )
    parser.add_argument(
        '--n_diff',
        type=int,
        required=True,
        help='Number of sites where pat1 and pat2 have different genotypes.',
    )
    parser.add_argument(
        '--cells_per_sample',
        type=int,
        nargs=3,
        default=None,
        help='Number of singlets for pat0, pat1 and pat2. Default: drawn uniformly '
        f'from {CELLS_PER_SAMPLE_RANGE[0]}-{CELLS_PER_SAMPLE_RANGE[1]} per sample.',
    )
    parser.add_argument(
        '--doublet_rate',
        type=float,
        default=None,
        help='Fraction of cells that are doublets. Default: drawn uniformly from '
        f'{DOUBLET_RATE_RANGE[0]}-{DOUBLET_RATE_RANGE[1]}.',
    )
    parser.add_argument(
        '--depth_source',
        type=str,
        required=True,
        help='Path to a depth matrix (from export_depth.py) to resample read depths from.',
    )
    parser.add_argument('--seed', type=int, default=None, help='Random seed.')
    parser.add_argument(
        '--output', type=str, required=True, help='Path to the output CSV file.'
    )
    return parser.parse_args()


def main(args):
    """Simulate one pool and write it to `output`."""
    rng = np.random.default_rng(args.seed)
    # Draw unspecified settings first, so that they are reproducible given the seed.
    cells_per_sample = args.cells_per_sample
    if cells_per_sample is None:
        low, high = CELLS_PER_SAMPLE_RANGE
        cells_per_sample = rng.integers(low, high + 1, 3).tolist()
    doublet_rate = args.doublet_rate
    if doublet_rate is None:
        doublet_rate = rng.uniform(*DOUBLET_RATE_RANGE)

    ref, alt, labels = simulate_pool(
        args.n_same,
        args.n_diff,
        cells_per_sample,
        doublet_rate,
        load_depth(args.depth_source),
        rng,
    )
    write_csv(ref, alt, labels, args.output, rng)


if __name__ == '__main__':
    args = parse_args()
    main(args)
