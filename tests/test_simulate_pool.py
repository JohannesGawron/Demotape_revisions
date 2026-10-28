"""Unit tests for pool_simulation.simulate_pool."""

import types

import numpy as np
import polars as pl

from pool_simulation import simulate_pool as sp

# Stand-in for a real library: 30 loci x 200 cells, the first 5 loci too shallow
# to pass the depth filter.
REAL_DEPTH = np.random.default_rng(42).poisson(50, (30, 200))
REAL_DEPTH[:5] = 3


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
        10, 5, [300, 200, 100], 0.25, REAL_DEPTH, np.random.default_rng(0)
    )

    n_doublets = sum('+' in label for label in labels)
    assert len(labels) == 600 + 200
    assert n_doublets / len(labels) == 0.25
    assert [labels.count(f'pat{i}') for i in range(3)] == [300, 200, 100]
    assert ref.shape == alt.shape == (15, len(labels))
    assert 'pat0+pat0' in labels or 'pat1+pat1' in labels or 'pat2+pat2' in labels


def test_every_site_passes_depth_filter():
    ref, alt, _ = sp.simulate_pool(
        20, 20, [100, 100, 100], 0.25, REAL_DEPTH, np.random.default_rng(0)
    )

    genotyped = (ref + alt) >= sp.MIN_DEPTH
    assert (genotyped.mean(axis=1) >= sp.MIN_GENOTYPED_FRACTION).all()


def test_singlet_depths_are_resampled_from_passing_real_loci():
    ref, alt, labels = sp.simulate_pool(
        10, 10, [100, 100, 100], 0.25, REAL_DEPTH, np.random.default_rng(0)
    )
    singlet_depths = (ref + alt)[:, ['+' not in label for label in labels]]

    # Each site's depths come from a single real locus that passes the filter.
    for site in singlet_depths:
        assert any(set(site) <= set(locus) for locus in REAL_DEPTH[5:])


def test_doublet_genotypes_follow_the_paper(monkeypatch):
    # Return the genotype the reads are simulated from as ref, and enough alt
    # reads to pass the depth filter.
    monkeypatch.setattr(
        sp,
        'simulate_reads',
        lambda g, depth, het_params, hom_error, rng: (g, 0 * g + 100),
    )
    rng = np.random.default_rng(0)
    ref, _, labels = sp.simulate_pool(5, 5, [50, 50, 50], 0.5, REAL_DEPTH, rng)
    genotypes = sp.simulate_patient_genotypes(5, 5, np.random.default_rng(0))

    for i, label in enumerate(labels):
        a, b = (
            (int(p[-1]) for p in label.split('+'))
            if '+' in label
            else (int(label[-1]),) * 2
        )
        # Equal genotypes are kept, any other combination becomes het.
        expected = np.where(genotypes[:, a] == genotypes[:, b], genotypes[:, a], 1)
        assert (ref[:, i] == expected).all()


def test_sites_take_depths_from_one_block():
    # Two blocks with constant depths 20 and 40, nan outside the blocks.
    real_depth = np.full((10, 50), np.nan)
    real_depth[:4, :20] = 20
    real_depth[4:, 20:] = 40
    ref, alt, _ = sp.simulate_pool(
        20, 20, [50, 50, 50], 0.25, real_depth, np.random.default_rng(0)
    )

    depth = ref + alt
    assert (depth == depth[:, :1]).all()  # no mixing of blocks within a site
    assert set(depth[:, 0]) == {20, 40}


def test_doublets_get_two_cells_per_block(monkeypatch):
    # Every cell has its own depth, so a doublet's depth is the mean of two
    # different cells only if they differ.
    real_depth = np.full((4, 6), np.nan)
    real_depth[:2, :3] = [[10, 20, 40]] * 2
    real_depth[2:, 3:] = [[100, 200, 400]] * 2
    monkeypatch.setattr(
        sp,
        'simulate_reads',
        lambda g, depth, het_params, hom_error, rng: (depth, 0 * depth),
    )
    ref, _, labels = sp.simulate_pool(
        2, 2, [20, 20, 20], 0.5, real_depth, np.random.default_rng(0)
    )

    doublets = ref[:, ['+' in label for label in labels]]
    assert not np.isin(doublets, [10, 20, 40, 100, 200, 400]).any()


def test_load_depth(tmp_path):
    path = tmp_path / 'depth.csv'
    path.write_text('15,nan\n7,7\n')

    np.testing.assert_array_equal(sp.load_depth(str(path)), [[15, np.nan], [7, 7]])


def test_csv_format_and_seed_reproducibility(tmp_path):
    depth_source = tmp_path / 'depth.csv'
    np.savetxt(depth_source, np.full((3, 6), 40), fmt='%.0f', delimiter=',')

    def run(name):
        args = types.SimpleNamespace(
            n_same=3,
            n_diff=2,
            cells_per_sample=[4, 4, 4],
            doublet_rate=0.25,
            depth_source=str(depth_source),
            seed=1,
            output=str(tmp_path / name),
        )
        sp.main(args)
        return pl.read_csv(args.output, infer_schema_length=0)

    df = run('a.csv')

    assert df.columns[:7] == ['CHR', 'POS', 'REF', 'ALT', 'REGION', 'NAME', 'FREQ']
    assert df.height == 5
    assert len(df.columns) == 7 + 12 + 4
    assert all(c.split('.')[1].startswith('pat') for c in df.columns[7:])
    assert df.columns[7].startswith('cell0.') and df['CHR'][0] == '1'
    ref, alt, gt = df[df.columns[7]][0].split(':')
    assert ref.isdigit() and alt.isdigit() and gt == '3'
    assert df.equals(run('b.csv'))


def test_unspecified_cells_and_doublet_rate_are_drawn_reproducibly(tmp_path):
    depth_source = tmp_path / 'depth.csv'
    np.savetxt(depth_source, np.full((3, 6), 40), fmt='%.0f', delimiter=',')

    def run(name, seed):
        args = types.SimpleNamespace(
            n_same=1,
            n_diff=1,
            cells_per_sample=None,
            doublet_rate=None,
            depth_source=str(depth_source),
            seed=seed,
            output=str(tmp_path / name),
        )
        sp.main(args)
        labels = [c.split('.')[1] for c in pl.read_csv(args.output).columns[7:]]
        singlets = [labels.count(f'pat{i}') for i in range(3)]
        return singlets, 1 - sum(singlets) / len(labels)

    singlets, doublet_rate = run('a.csv', seed=1)

    assert all(500 <= n <= 3000 for n in singlets)
    assert 0.05 - 1e-3 <= doublet_rate <= 0.3 + 1e-3  # up to rounding of n_doublets
    assert run('b.csv', seed=1) == (singlets, doublet_rate)
    assert run('c.csv', seed=2) != (singlets, doublet_rate)


def test_het_params_are_valid_and_centred():
    m, kappa, d, eps = sp.draw_het_params(10_000, np.random.default_rng(0))

    assert ((m > 0) & (m < 1)).all() and (kappa > 0).all()
    assert ((d > 0) & (d < 1)).all() and ((eps > 0) & (eps < 0.5)).all()
    np.testing.assert_allclose(np.median(np.log(kappa)), sp.HET_MEAN[1], atol=0.05)
    np.testing.assert_allclose(np.median(m), sp.expit(sp.HET_MEAN[0]), atol=0.01)


def test_het_reads_full_dropout_without_error():
    depth = np.full((5, 1000), 30)
    het_params = np.array([[0.5] * 5, [20] * 5, [1.0] * 5, [0.0] * 5])
    _, alt = sp.simulate_reads(
        np.ones_like(depth), depth, het_params, np.zeros(5), np.random.default_rng(0)
    )

    assert set(np.unique(alt)) == {0, 30}


def test_het_reads_without_dropout_concentrate_at_m():
    depth = np.full((1, 5000), 10_000)
    het_params = np.array([[0.3], [1e6], [0.0], [0.01]])
    _, alt = sp.simulate_reads(
        np.ones_like(depth), depth, het_params, np.zeros(1), np.random.default_rng(0)
    )

    np.testing.assert_allclose(alt / depth, 0.3, atol=0.03)


def test_hom_reads_use_the_site_error_rate():
    depth = np.full((4, 5000), 100)
    genotypes = np.array([[0] * 5000, [2] * 5000, [0] * 5000, [2] * 5000])
    het_params = sp.draw_het_params(4, np.random.default_rng(0))
    hom_error = np.array([0.001, 0.001, 0.05, 0.05])
    _, alt = sp.simulate_reads(
        genotypes, depth, het_params, hom_error, np.random.default_rng(0)
    )

    np.testing.assert_allclose(
        (alt / depth).mean(axis=1), [0.001, 0.999, 0.05, 0.95], atol=0.002
    )


def test_hom_error_follows_the_logit_normal():
    error = sp.draw_hom_error(10_000, np.random.default_rng(0))
    logit_error = np.log(error / (1 - error))

    assert ((error > 0) & (error < 1)).all()
    np.testing.assert_allclose(logit_error.mean(), sp.HOM_ERROR_MU, atol=0.05)
    np.testing.assert_allclose(logit_error.std(), sp.HOM_ERROR_TAU, atol=0.05)
