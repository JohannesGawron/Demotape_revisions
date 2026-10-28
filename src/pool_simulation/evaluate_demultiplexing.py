"""
Evaluate DemoTape's demultiplexing of a simulated pool: how well pat1 and pat2
are separated, measured by the adjusted Rand index (ARI) between the true
patient and the assigned cluster.

Only cells that are pat1 or pat2 singlets are used; pat0 and all doublets are
treated as nuisance and ignored. Their assigned cluster is taken as DemoTape
reports it, so a pat1/pat2 cell assigned to the pat0 cluster or to a doublet
cluster counts as an error.
"""

import argparse

import numpy as np
import polars as pl


def read_assignments(path: str) -> tuple[list[str], list[str]]:
    """Return the cell names and assigned clusters of a DemoTape `*.assignments.tsv` file."""
    with open(path) as f:
        rows = {line.split('\t')[0]: line.rstrip('\n').split('\t')[1:] for line in f}
    return rows['Barcode'], rows['Cluster']


def adjusted_rand_index(labels_a: list[str], labels_b: list[str]) -> float:
    """Adjusted Rand index between two labelings of the same items (Hubert & Arabie, 1985)."""
    _, a = np.unique(labels_a, return_inverse=True)
    _, b = np.unique(labels_b, return_inverse=True)
    table = np.zeros((a.max() + 1, b.max() + 1))
    np.add.at(table, (a, b), 1)

    def pairs(counts):
        return (counts * (counts - 1) / 2).sum()

    index = pairs(table)
    expected = pairs(table.sum(axis=1)) * pairs(table.sum(axis=0)) / pairs(table.sum())
    max_index = (pairs(table.sum(axis=1)) + pairs(table.sum(axis=0))) / 2
    return float((index - expected) / (max_index - expected))


def evaluate(path: str) -> dict:
    """ARI of pat1 vs. pat2 singlets for one assignments file, with the number of cells used."""
    cells, clusters = read_assignments(path)
    truth = [cell.rsplit('.', 1)[1] for cell in cells]  # cell<i>.<ground truth>
    keep = [t in ('pat1', 'pat2') for t in truth]
    truth = [t for t, k in zip(truth, keep, strict=True) if k]
    assigned = [c for c, k in zip(clusters, keep, strict=True) if k]
    return {
        'file': path,
        'n_cells': len(truth),
        'ari_pat1_pat2': adjusted_rand_index(truth, assigned),
    }


def parse_args():
    """Parse command-line arguments for the evaluation."""
    parser = argparse.ArgumentParser(
        description='Adjusted Rand index of pat1 vs. pat2 in DemoTape assignments.'
    )
    parser.add_argument(
        '--assignments',
        type=str,
        nargs='+',
        required=True,
        help='DemoTape *.assignments.tsv files.',
    )
    parser.add_argument(
        '--output', type=str, required=True, help='Path to the output TSV file.'
    )
    return parser.parse_args()


def main(args):
    """Evaluate every assignments file and write one row per file to `output`."""
    pl.DataFrame([evaluate(path) for path in args.assignments]).write_csv(
        args.output, separator='\t'
    )


if __name__ == '__main__':
    args = parse_args()
    main(args)
