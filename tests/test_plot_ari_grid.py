"""Unit tests for pool_simulation.plot_ari_grid."""

import polars as pl

from pool_simulation import plot_ari_grid as pg


def write_ari_table(path):
    rows = [
        {
            'file': f'results/pool_grid/demultiplexed/same{s}_diff{d}_seed{seed}.assignments.tsv',
            'n_cells': 3000,
            'ari_pat1_pat2': 0.1 * d,
        }
        for s in (1, 10)
        for d in (2, 5)
        for seed in range(3)
    ]
    pl.DataFrame(rows).write_csv(path, separator='\t')


def test_read_results_parses_grid_parameters(tmp_path):
    path = tmp_path / 'ari.tsv'
    write_ari_table(path)

    df = pg.read_results(str(path))

    assert sorted(df['n_same'].unique()) == [1, 10]
    assert sorted(df['n_diff'].unique()) == [2, 5]
    assert sorted(df['seed'].unique()) == [0, 1, 2]


def test_plot_grid_writes_figure(tmp_path):
    path = tmp_path / 'ari.tsv'
    write_ari_table(path)
    output = tmp_path / 'ari_grid.png'

    pg.plot_grid(pg.read_results(str(path)), str(output))

    assert output.stat().st_size > 0
