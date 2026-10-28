"""Unit tests for pool_simulation.export_depth."""

import types

import numpy as np

from pool_simulation import export_depth as ed
from pool_simulation import simulate_pool as sp


def test_read_filtered_variants(tmp_path):
    path = tmp_path / 'library.filtered_variants.csv'
    path.write_text(
        'CHR,POS,REF,ALT,REGION,NAME,FREQ,AAA-1,CCC-1\n'
        '1,100,A,C,r,n,0,10:5:1,0:0:3\n'
        '2,200,G,T,r,n,0,7:0:0,3:4:1\n'
    )

    assert ed.read_filtered_variants(str(path)).tolist() == [[15, 0], [7, 7]]


def test_stack_blocks_keeps_each_library_as_a_block():
    a = np.arange(6).reshape(2, 3)
    b = np.arange(100, 104).reshape(1, 4)
    stacked = ed.stack_blocks([a, b], np.random.default_rng(0))

    assert stacked.shape == (3, 7)
    assert np.isnan(stacked).sum() == 2 * 4 + 1 * 3
    cell_block, locus_block = sp.blocks(stacked)
    observed = [
        sorted(stacked[locus_block == k][:, cell_block == k].ravel()) for k in (0, 1)
    ]
    assert sorted(observed) == sorted([sorted(a.ravel()), sorted(b.ravel())])


def test_export_round_trip(tmp_path):
    rows = [f'1,{i},A,C,r,n,0,' + ','.join([f'{i}:1:1'] * 3) for i in range(2)]
    inputs = []
    for name in ['a', 'b']:
        path = tmp_path / f'{name}.filtered_variants.csv'
        path.write_text(
            'CHR,POS,REF,ALT,REGION,NAME,FREQ,X-1,Y-1,Z-1\n' + '\n'.join(rows) + '\n'
        )
        inputs.append(str(path))
    output = tmp_path / 'depth.csv.gz'
    ed.main(types.SimpleNamespace(inputs=inputs, seed=0, output=str(output)))

    depth = sp.load_depth(str(output))
    assert depth.shape == (4, 6)
    assert sorted(depth[~np.isnan(depth)]) == [1] * 6 + [2] * 6
