#!/usr/bin/env python3
"""Compute the Jaccard index between the variant sets of two
`*.filtered_variants.csv` tables, where each variant is uniquely identified
by its (CHR, POS, REF, ALT) tuple.
"""

import argparse
from pathlib import Path
from mutation_call_variability import commodity
import polars as pl


def load_variant_set(path: str) -> set[tuple[str, int, str, str]]:
    df = pl.read_csv(
        path,
        columns=["CHR", "POS", "REF", "ALT"],
        dtypes={"CHR": pl.Utf8, "POS": pl.Int64, "REF": pl.Utf8, "ALT": pl.Utf8},
    )
    return set(df.iter_rows())


def jaccard_index(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute the Jaccard index between the variant sets of "
        "two filtered_variants.csv tables (variants identified by CHR, POS, REF, ALT)."
    )
    parser.add_argument("table_a", type=str, help="Path to the first filtered_variants.csv file.")
    parser.add_argument("table_b", type=str, help="Path to the second filtered_variants.csv file.")
    return parser.parse_args()

def compare_variant_calls(dataset1_file: str, dataset2_file: str) -> float: 
    """Compute the Jaccard index between the variant sets of two filtered_variants.csv tables."""
    set_a = load_variant_set(dataset1_file)
    set_b = load_variant_set(dataset2_file)
    return jaccard_index(set_a, set_b)


def compute_jaccard_indices(lookup_table_file: str, full_variant_sets_assignment: dict[str, str]) -> None:
    lookup_table = commodity.load_lookup_table(lookup_table_file)

    output_table = pl.DataFrame(
        schema={
            'hash': pl.Utf8,
            'paper_id': pl.Utf8,
            'n_cells': pl.Int64,
            'jaccard_index': pl.Float64,
        }
    )
    for sample in lookup_table:
        paper_ID = commodity.file_name2metadata(sample, 'paper_id')
        n_cells = commodity.file_name2metadata(sample, 'n_cells')
        hash_value = commodity.file_name2metadata(sample, 'hash')

        full_set_path = full_variant_sets_assignment.get(paper_ID)
        jaccard_index_value = compare_variant_calls(sample, full_set_path)

        output_table = output_table.vstack(pl.DataFrame({
            'hash': [hash_value],
            'paper_id': [paper_ID],
            'n_cells': [n_cells],
            'jaccard_index': [jaccard_index_value],
        }))

    return output_table
