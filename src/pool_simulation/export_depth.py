"""
Export the read depths of real libraries as one anonymised depth matrix, the
`--depth_source` of `simulate_pool.py`.

The libraries are stacked block by block: rows are the loci of all libraries,
columns the cells of all libraries, and entries are `nan` where a cell's library
lacks a locus. Rows and columns are shuffled and the file has no header, so it
contains no positions, alt counts, genotypes or barcodes, only depths.
"""

import argparse

import numpy as np
import numpy.typing as npt
import polars as pl


def read_filtered_variants(path: str) -> npt.NDArray[np.int64]:
    """Return the read depths (RO + AD) of a `*.filtered_variants.csv` file as a loci x cells array."""
    df = pl.read_csv(path)
    entries = df.select(df.columns[7:])
    ref = entries.select(pl.all().str.split(':').list.get(0).cast(pl.Int64))
    alt = entries.select(pl.all().str.split(':').list.get(1).cast(pl.Int64))
    return (ref + alt).to_numpy()


def stack_blocks(
    depths: list[npt.NDArray[np.int64]], rng: np.random.Generator
) -> npt.NDArray[np.float64]:
    """Stack loci x cells depth matrices block-diagonally (`nan` elsewhere) and shuffle rows and columns."""
    stacked = np.full(
        (sum(d.shape[0] for d in depths), sum(d.shape[1] for d in depths)), np.nan
    )
    row, col = 0, 0
    for d in depths:
        stacked[row : row + d.shape[0], col : col + d.shape[1]] = d
        row, col = row + d.shape[0], col + d.shape[1]
    return stacked[rng.permutation(row)][:, rng.permutation(col)]


def parse_args():
    """Parse command-line arguments for the depth export."""
    parser = argparse.ArgumentParser(
        description='Export real read depths as one anonymised depth matrix.'
    )
    parser.add_argument(
        '--inputs',
        type=str,
        nargs='+',
        required=True,
        help='Paths to *.filtered_variants.csv files, one per library.',
    )
    parser.add_argument('--seed', type=int, default=None, help='Random seed.')
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to the output CSV file (gzipped if it ends in .gz).',
    )
    return parser.parse_args()


def main(args):
    """Stack the input libraries and write the depth matrix to `output`."""
    rng = np.random.default_rng(args.seed)
    stacked = stack_blocks([read_filtered_variants(p) for p in args.inputs], rng)
    np.savetxt(args.output, stacked, fmt='%.0f', delimiter=',')


if __name__ == '__main__':
    args = parse_args()
    main(args)
