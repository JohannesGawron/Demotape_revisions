"""Unit tests for pool_simulation.simulate_pool."""

import types

import numpy as np
import polars as pl

from pool_simulation import simulate_pool as sp


def test_patient_genotypes_have_exact_same_and_diff_counts():
    genotypes = sp.simulate_patient_genotypes(30, 20, np.random.default_rng(0))

    assert genotypes.shape == (50, 3)
    assert (genotypes[:30, 1] == genotypes[:30, 2]).all()
    assert (genotypes[30:, 1] != genotypes[30:, 2]).all()
    # No site is identical in all three patients (this also excludes 0/0/0).
    assert not (
        (genotypes[:, 0] == genotypes[:, 1]) & (genotypes[:, 1] == genotypes[:, 2])
    ).any()


def test_pool_has_requested_cells_and_doublet_rate():
    ref, alt, labels = sp.simulate_pool(
        10, 5, [300, 200, 100], 0.25, np.random.default_rng(0)
    )

    n_doublets = sum('+' in label for label in labels)
    assert len(labels) == 600 + 200
    assert n_doublets / len(labels) == 0.25
    assert [labels.count(f'pat{i}') for i in range(3)] == [300, 200, 100]
    assert ref.shape == alt.shape == (15, len(labels))
    assert 'pat0+pat0' in labels or 'pat1+pat1' in labels or 'pat2+pat2' in labels


def test_every_site_passes_depth_filter():
    ref, alt, _ = sp.simulate_pool(
        20, 20, [100, 100, 100], 0.25, np.random.default_rng(0)
    )

    genotyped = (ref + alt) >= sp.MIN_DEPTH
    assert (genotyped.mean(axis=1) >= sp.MIN_GENOTYPED_FRACTION).all()


def test_doublet_counts_are_rounded_mean(monkeypatch):
    # Deterministic reads: ref = 10 * genotype + 1, alt = 10 * genotype + 11.
    monkeypatch.setattr(sp, 'simulate_reads', lambda g, rng: (10 * g + 1, 10 * g + 11))
    rng = np.random.default_rng(0)
    ref, alt, labels = sp.simulate_pool(5, 5, [50, 50, 50], 0.5, rng)
    genotypes = sp.simulate_patient_genotypes(5, 5, np.random.default_rng(0))

    for i, label in enumerate(labels):
        a, b = (
            (int(p[-1]) for p in label.split('+'))
            if '+' in label
            else (int(label[-1]),) * 2
        )
        expected_ref = np.rint((genotypes[:, a] + genotypes[:, b]) * 10 / 2 + 1)
        assert (ref[:, i] == expected_ref).all()
        assert (alt[:, i] - ref[:, i] == 10).all()


def test_csv_format_and_seed_reproducibility(tmp_path):
    def run(name):
        args = types.SimpleNamespace(
            n_same=3,
            n_diff=2,
            cells_per_sample=[4, 4, 4],
            doublet_rate=0.25,
            seed=1,
            output=str(tmp_path / name),
        )
        sp.main(args)
        return pl.read_csv(args.output, infer_schema_length=0)

    df = run('a.csv')

    assert df.columns[:4] == ['CHR', 'POS', 'REF', 'ALT']
    assert df.height == 5
    assert len(df.columns) == 4 + 12 + 4
    assert all(c.split('.')[1].startswith('pat') for c in df.columns[4:])
    ref, alt, gt = df[df.columns[4]][0].split(':')
    assert ref.isdigit() and alt.isdigit() and gt == '3'
    assert df.equals(run('b.csv'))
