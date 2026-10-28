"""Unit tests for pool_simulation.evaluate_demultiplexing."""

import types

import polars as pl
import pytest

from pool_simulation import evaluate_demultiplexing as ev


def test_adjusted_rand_index_known_values():
    # Reference values as given by sklearn.metrics.adjusted_rand_score.
    assert ev.adjusted_rand_index(['a', 'a', 'b', 'b'], ['1', '1', '0', '0']) == 1.0
    assert ev.adjusted_rand_index(
        ['a', 'a', 'b', 'b'], ['0', '0', '1', '2']
    ) == pytest.approx(0.5714285714)
    assert ev.adjusted_rand_index(
        ['a', 'a', 'b', 'b'], ['0', '1', '0', '1']
    ) == pytest.approx(-0.5)


def write_assignments(path, cells, clusters):
    path.write_text(
        'Barcode\t'
        + '\t'.join(cells)
        + '\nCluster\t'
        + '\t'.join(clusters)
        + '\nOrder\t'
        + '\t'.join(str(i) for i in range(len(cells)))
    )


def test_evaluate_ignores_pat0_and_doublets(tmp_path):
    path = tmp_path / 'pool.assignments.tsv'
    # pat1 and pat2 perfectly separated; pat0 and the doublets are assigned
    # arbitrarily and must not affect the result.
    write_assignments(
        path,
        [
            'cell0.pat1',
            'cell1.pat1',
            'cell2.pat2',
            'cell3.pat2',
            'cell4.pat0',
            'cell5.pat1+pat2',
            'cell6.pat1+pat1',
        ],
        ['2', '2', '0', '0', '2', '1', '0'],
    )

    result = ev.evaluate(str(path))

    assert result['n_cells'] == 4
    assert result['ari_pat1_pat2'] == 1.0


def test_pat1_cell_in_doublet_cluster_counts_as_error(tmp_path):
    path = tmp_path / 'pool.assignments.tsv'
    write_assignments(
        path,
        ['cell0.pat1', 'cell1.pat1', 'cell2.pat2', 'cell3.pat2'],
        ['0', '0+1', '1', '1'],
    )

    assert ev.evaluate(str(path))['ari_pat1_pat2'] < 1.0


def test_main_writes_one_row_per_file(tmp_path):
    paths = []
    for i in range(2):
        path = tmp_path / f'pool{i}.assignments.tsv'
        write_assignments(
            path,
            ['cell0.pat1', 'cell1.pat1', 'cell2.pat2', 'cell3.pat2'],
            ['0', '0', '1', '1'],
        )
        paths.append(str(path))
    output = tmp_path / 'ari.tsv'

    ev.main(types.SimpleNamespace(assignments=paths, output=str(output)))

    df = pl.read_csv(output, separator='\t')
    assert df.columns == ['file', 'n_cells', 'ari_pat1_pat2']
    assert df.height == 2
