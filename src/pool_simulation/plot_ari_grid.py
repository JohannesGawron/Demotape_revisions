"""
Plot the ARI of pat1 vs. pat2 (from `evaluate_demultiplexing.py`) over the
simulation grid: one boxplot over the seeds per combination of n_same (rows)
and n_diff (columns). n_same, n_diff and the seed are read from the pool file
names (`same<n_same>_diff<n_diff>_seed<seed>`). The styling follows the plots of
INSCITE-py (`inscite_py/types.py`).
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

TEXT_COLOR = '#1D3557'  # as in INSCITE-py


def read_results(path: str) -> pl.DataFrame:
    """Read the ARI table and add n_same, n_diff and seed, parsed from the file names."""
    df = pl.read_csv(path, separator='\t')
    return df.with_columns(
        pl.col('file').str.extract(rf'{name}(\d+)').cast(pl.Int64).alias(column)
        for name, column in [('same', 'n_same'), ('diff', 'n_diff'), ('seed', 'seed')]
    )


def plot_grid(df: pl.DataFrame, output: str) -> None:
    """Plot one boxplot of the ARI per (n_same, n_diff) combination and save it to `output`."""
    n_same = sorted(df['n_same'].unique())
    n_diff = sorted(df['n_diff'].unique())
    fig, axes = plt.subplots(
        len(n_same),
        len(n_diff),
        figsize=(14, 12),
        facecolor='white',
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    rng = np.random.default_rng(0)  # jitter of the points
    for i, s in enumerate(n_same):
        for j, d in enumerate(n_diff):
            ax = axes[i, j]
            ari = df.filter((pl.col('n_same') == s) & (pl.col('n_diff') == d))[
                'ari_pat1_pat2'
            ].to_numpy()
            ax.boxplot(ari, widths=0.5, showfliers=False)
            ax.scatter(
                rng.normal(1, 0.05, len(ari)), ari, s=12, alpha=0.5, color=TEXT_COLOR
            )
            ax.set_xticks([])
            ax.tick_params(axis='both', which='major', labelsize=14)
            if i == 0:
                ax.set_title(
                    f'n_diff = {d}',
                    fontsize=16,
                    fontweight='semibold',
                    pad=10,
                    color=TEXT_COLOR,
                )
            if j == 0:
                ax.set_ylabel(
                    f'n_same = {s}',
                    fontsize=16,
                    fontweight='semibold',
                    labelpad=10,
                    color=TEXT_COLOR,
                )
    axes[0, 0].set_ylim(min(-0.05, df['ari_pat1_pat2'].min() - 0.05), 1.05)
    fig.suptitle(
        'Adjusted Rand index of pat1 vs. pat2',
        fontsize=16,
        fontweight='semibold',
        color=TEXT_COLOR,
    )
    fig.tight_layout()
    fig.savefig(output, dpi=300)


def parse_args():
    """Parse command-line arguments for the plot."""
    parser = argparse.ArgumentParser(
        description='Plot the ARI of pat1 vs. pat2 as a grid of boxplots.'
    )
    parser.add_argument(
        '--ari',
        type=str,
        required=True,
        help='ARI table from evaluate_demultiplexing.py.',
    )
    parser.add_argument(
        '--output', type=str, required=True, help='Path to the output figure.'
    )
    return parser.parse_args()


def main(args):
    """Read the ARI table and plot it."""
    plot_grid(read_results(args.ari), args.output)


if __name__ == '__main__':
    args = parse_args()
    main(args)
