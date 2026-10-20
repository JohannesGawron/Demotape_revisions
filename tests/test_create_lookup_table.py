"""Unit tests for mutation_call_variability.create_lookup_table."""

import types
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl

from mutation_call_variability import create_lookup_table


def make_args(out_dir, repetitions):
    return types.SimpleNamespace(
        input=['GFB-4397_GFB-4401.loom'],
        repetitions=repetitions,
        output_folder=str(out_dir),
        output_folder_lookup=str(out_dir),
    )


def run_main_and_read_counts(args, n_cells):
    """Run main() against a mocked loom file and return the sampled cell counts."""
    fake_dataset = MagicMock()
    fake_dataset.shape = (500, n_cells)
    fake_connect = MagicMock()
    fake_connect.return_value.__enter__.return_value = fake_dataset
    fake_connect.return_value.__exit__.return_value = False

    with patch.object(create_lookup_table.loompy, 'connect', fake_connect):
        create_lookup_table.main(args)

    out_table = pl.read_csv(
        f'{args.output_folder_lookup}/subsample_lookup_table.tsv', separator='\t'
    )
    return out_table['cell_count'].to_numpy()


def test_repeated_calls_do_not_produce_identical_samples(tmp_path):
    """main() creates `rng = np.random.default_rng()` fresh (unseeded) on every
    call, so calling it twice must not replay the exact same sequence of
    sampled cell counts. This guards against a regression where the RNG is
    seeded with a fixed value inside the function on every call.
    """
    n_cells = 1000
    repetitions = 500

    out_dir_1 = tmp_path / 'run1'
    out_dir_1.mkdir()
    counts_1 = run_main_and_read_counts(make_args(out_dir_1, repetitions), n_cells)

    out_dir_2 = tmp_path / 'run2'
    out_dir_2.mkdir()
    counts_2 = run_main_and_read_counts(make_args(out_dir_2, repetitions), n_cells)

    assert counts_1.shape == counts_2.shape == (repetitions,)
    assert not np.array_equal(counts_1, counts_2)


def test_samples_are_within_valid_cell_range(tmp_path):
    n_cells = 250
    repetitions = 300

    counts = run_main_and_read_counts(make_args(tmp_path, repetitions), n_cells)

    assert len(counts) == repetitions
    assert counts.min() >= 1
    assert counts.max() <= n_cells
