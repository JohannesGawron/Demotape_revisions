"""
Simulate a pool of three patients (pat0, pat1, pat2) with known ground truth, in
the CSV format DemoTape takes as input: columns CHR,POS,REF,ALT followed by one
column per barcode, named `<barcode>.<label>` (e.g. `ACGT....pat0+pat1`), with
entries `REF:ALT:GT` (GT is ignored by DemoTape and set to a constant).

pat1 and pat2 have the same genotype at exactly `n_same` sites and different
genotypes at exactly `n_diff` sites; pat0's genotype is random, but never equal to
both others (such sites would be filtered out).
"""

import argparse

import numpy as np
import numpy.typing as npt
import polars as pl

# Placeholder read model, to be replaced: depth ~ NegBinom, alt ~ Binom(depth, VAF).
MEAN_DEPTH = 50
DEPTH_DISPERSION = 2
SEQ_ERROR = 0.01
VAF = np.array([SEQ_ERROR, 0.5, 1 - SEQ_ERROR])  # indexed by genotype 0/1/2

# Mosaic variant filter: a site needs depth >= MIN_DEPTH in >= MIN_GENOTYPED_FRACTION of cells.
MIN_DEPTH = 10
MIN_GENOTYPED_FRACTION = 0.5

GT_PLACEHOLDER = 3
BARCODE_LENGTH = 12


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


def simulate_reads(
    genotypes: npt.NDArray[np.int64], rng: np.random.Generator
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Simulate (ref, alt) read counts for a sites x cells genotype matrix."""
    p = DEPTH_DISPERSION / (DEPTH_DISPERSION + MEAN_DEPTH)
    depth = rng.negative_binomial(DEPTH_DISPERSION, p, size=genotypes.shape)
    alt = rng.binomial(depth, VAF[genotypes])
    return depth - alt, alt


def simulate_pool(
    n_same: int,
    n_diff: int,
    cells_per_sample: list[int],
    doublet_rate: float,
    rng: np.random.Generator,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], list[str]]:
    """Simulate a pool and return the (ref, alt) read counts (sites x barcodes) and the ground-truth label per barcode."""
    genotypes = simulate_patient_genotypes(n_same, n_diff, rng)

    singlets = np.repeat(np.arange(3), cells_per_sample)
    n_singlets = len(singlets)
    n_doublets = round(doublet_rate / (1 - doublet_rate) * n_singlets)
    mixing_ratio = np.array(cells_per_sample) / n_singlets
    doublets = np.sort(rng.choice(3, size=(n_doublets, 2), p=mixing_ratio), axis=1)

    pat_a = np.concatenate([singlets, doublets[:, 0]])
    pat_b = np.concatenate([singlets, doublets[:, 1]])
    is_doublet = np.arange(len(pat_a)) >= n_singlets

    def simulate(rows: npt.NDArray[np.bool_]):
        # Every barcode gets a cell from pat_a and one from pat_b; singlets keep
        # only the first, doublets get the rounded mean of both.
        ref_a, alt_a = simulate_reads(genotypes[rows][:, pat_a], rng)
        ref_b, alt_b = simulate_reads(genotypes[rows][:, pat_b], rng)
        ref = np.where(is_doublet, np.rint((ref_a + ref_b) / 2), ref_a)
        alt = np.where(is_doublet, np.rint((alt_a + alt_b) / 2), alt_a)
        return ref.astype(np.int64), alt.astype(np.int64)

    ref, alt = simulate(np.ones(len(genotypes), dtype=bool))
    # Re-simulate sites that would not pass the Mosaic depth filter.
    failing = ((ref + alt) >= MIN_DEPTH).mean(axis=1) < MIN_GENOTYPED_FRACTION
    while failing.any():
        ref[failing], alt[failing] = simulate(failing)
        failing = ((ref + alt) >= MIN_DEPTH).mean(axis=1) < MIN_GENOTYPED_FRACTION

    labels = [
        f'pat{a}+pat{b}' if doublet else f'pat{a}'
        for a, b, doublet in zip(pat_a, pat_b, is_doublet, strict=True)
    ]
    return ref, alt, labels


def random_barcodes(n: int, rng: np.random.Generator) -> list[str]:
    """Return n unique random nucleotide barcodes."""
    barcodes: dict[str, None] = {}
    while len(barcodes) < n:
        barcodes[''.join(rng.choice(list('ACGT'), BARCODE_LENGTH))] = None
    return list(barcodes)


def write_csv(
    ref: npt.NDArray[np.int64],
    alt: npt.NDArray[np.int64],
    labels: list[str],
    output: str,
    rng: np.random.Generator,
) -> None:
    """Write the pool in DemoTape's input format, with synthetic site coordinates and barcodes."""
    n_sites = ref.shape[0]
    bases = np.array(list('ACGT'))
    ref_idx = rng.integers(0, 4, n_sites)
    alt_idx = (ref_idx + rng.integers(1, 4, n_sites)) % 4  # any base but REF

    columns = {
        'CHR': ['chr1'] * n_sites,
        'POS': np.arange(1, n_sites + 1) * 1000,
        'REF': bases[ref_idx],
        'ALT': bases[alt_idx],
    }
    barcodes = random_barcodes(len(labels), rng)
    for i, (barcode, label) in enumerate(zip(barcodes, labels, strict=True)):
        columns[f'{barcode}.{label}'] = [
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
        default=[1500, 1500, 1500],
        help='Number of singlets for pat0, pat1 and pat2.',
    )
    parser.add_argument(
        '--doublet_rate',
        type=float,
        default=0.25,
        help='Fraction of barcodes that are doublets.',
    )
    parser.add_argument('--seed', type=int, default=None, help='Random seed.')
    parser.add_argument(
        '--output', type=str, required=True, help='Path to the output CSV file.'
    )
    return parser.parse_args()


def main(args):
    """Simulate one pool and write it to `output`."""
    rng = np.random.default_rng(args.seed)
    ref, alt, labels = simulate_pool(
        args.n_same, args.n_diff, args.cells_per_sample, args.doublet_rate, rng
    )
    write_csv(ref, alt, labels, args.output, rng)


if __name__ == '__main__':
    args = parse_args()
    main(args)
