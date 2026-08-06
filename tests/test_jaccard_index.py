"""Unit tests for mutation_call_variability.jaccard_index."""

import csv

import pytest

from mutation_call_variability.jaccard_index import (
    compare_variant_calls,
    compute_jaccard_indices,
    jaccard_index,
    load_variant_set,
)


def write_variant_csv(path, rows, header=("CHR", "POS", "REF", "ALT")):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# jaccard_index()
# ---------------------------------------------------------------------------


def test_jaccard_index_both_empty():
    assert jaccard_index(set(), set()) == 1.0


def test_jaccard_index_identical_sets():
    s = {("1", 100, "A", "T"), ("2", 200, "C", "G")}
    assert jaccard_index(s, s) == 1.0


def test_jaccard_index_disjoint_sets():
    a = {("1", 100, "A", "T")}
    b = {("2", 200, "C", "G")}
    assert jaccard_index(a, b) == 0.0


def test_jaccard_index_partial_overlap():
    a = {("1", 1, "A", "T"), ("1", 2, "A", "T"), ("1", 3, "A", "T")}
    b = {("1", 2, "A", "T"), ("1", 3, "A", "T"), ("1", 4, "A", "T")}
    # intersection = {2, 3} (size 2), union = {1, 2, 3, 4} (size 4)
    assert jaccard_index(a, b) == pytest.approx(2 / 4)


def test_jaccard_index_one_empty_one_nonempty():
    assert jaccard_index(set(), {("1", 1, "A", "T")}) == 0.0


def test_jaccard_index_subset():
    a = {("1", 1, "A", "T")}
    b = {("1", 1, "A", "T"), ("1", 2, "A", "T")}
    assert jaccard_index(a, b) == pytest.approx(1 / 2)


def test_jaccard_index_symmetric():
    a = {("1", 1, "A", "T"), ("1", 5, "A", "T")}
    b = {("1", 1, "A", "T"), ("1", 9, "A", "T")}
    assert jaccard_index(a, b) == jaccard_index(b, a)


# ---------------------------------------------------------------------------
# load_variant_set()
# ---------------------------------------------------------------------------


def test_load_variant_set_basic(tmp_path):
    path = tmp_path / "a.filtered_variants.csv"
    write_variant_csv(
        path,
        [
            ("1", 100, "A", "T"),
            ("2", 200, "C", "G"),
            ("X", 300, "G", "A"),
        ],
    )
    result = load_variant_set(str(path))
    assert result == {
        ("1", 100, "A", "T"),
        ("2", 200, "C", "G"),
        ("X", 300, "G", "A"),
    }
    pos = next(v for v in result if v[0] == "1")[1]
    assert isinstance(pos, int)


def test_load_variant_set_deduplicates_rows(tmp_path):
    path = tmp_path / "b.filtered_variants.csv"
    write_variant_csv(
        path,
        [
            ("1", 100, "A", "T"),
            ("1", 100, "A", "T"),
            ("2", 200, "C", "G"),
        ],
    )
    result = load_variant_set(str(path))
    assert len(result) == 2


def test_load_variant_set_ignores_extra_columns(tmp_path):
    path = tmp_path / "c.filtered_variants.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["CHR", "POS", "REF", "ALT", "REGION", "NAME"])
        writer.writerow(["1", 100, "A", "T", "chr1:100", "AMPL1"])
    result = load_variant_set(str(path))
    assert result == {("1", 100, "A", "T")}


def test_load_variant_set_handles_late_appearing_non_numeric_chromosome(tmp_path):
    """CHR is digit-only for the first 120 rows and only turns alphabetic
    (X/Y) afterwards. Polars' default schema inference only scans the
    first 100 rows, which infers CHR as Int64 and then fails to parse
    "X"/"Y" later in the file unless the full file is scanned.
    """
    path = tmp_path / "d.filtered_variants.csv"
    rows = [(str((i % 22) + 1), 1000 + i, "A", "T") for i in range(120)]
    rows.append(("X", 5000, "G", "C"))
    rows.append(("Y", 5001, "C", "G"))
    write_variant_csv(path, rows)

    result = load_variant_set(str(path))

    assert ("X", 5000, "G", "C") in result
    assert ("Y", 5001, "C", "G") in result
    assert len(result) == 122


# ---------------------------------------------------------------------------
# compare_variant_calls()
# ---------------------------------------------------------------------------


def test_compare_variant_calls_end_to_end(tmp_path):
    path_a = tmp_path / "a.filtered_variants.csv"
    path_b = tmp_path / "b.filtered_variants.csv"
    write_variant_csv(
        path_a,
        [("1", 1, "A", "T"), ("1", 2, "A", "T"), ("1", 3, "A", "T")],
    )
    write_variant_csv(
        path_b,
        [("1", 2, "A", "T"), ("1", 3, "A", "T"), ("1", 4, "A", "T")],
    )
    result = compare_variant_calls(str(path_a), str(path_b))
    assert result == pytest.approx(2 / 4)


# ---------------------------------------------------------------------------
# compute_jaccard_indices()
# ---------------------------------------------------------------------------


def test_compute_jaccard_indices_integration(tmp_path):
    full_a = tmp_path / "S1.full.filtered_variants.csv"
    full_b = tmp_path / "S2.full.filtered_variants.csv"
    write_variant_csv(
        full_a,
        [
            ("1", 1, "A", "T"),
            ("1", 2, "A", "T"),
            ("1", 3, "A", "T"),
            ("1", 4, "A", "T"),
        ],
    )
    write_variant_csv(full_b, [("2", 10, "C", "G"), ("2", 20, "C", "G")])

    sample_a = tmp_path / "S1.50.deadbeef.filtered_variants.csv"
    write_variant_csv(sample_a, [("1", 1, "A", "T"), ("1", 2, "A", "T")])

    sample_b = tmp_path / "S2.30.cafebabe.filtered_variants.csv"
    write_variant_csv(sample_b, [("2", 10, "C", "G"), ("2", 20, "C", "G")])

    lookup_table_path = tmp_path / "lookup_table.tsv"
    with open(lookup_table_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["output_name"])
        writer.writerow([str(sample_a)])
        writer.writerow([str(sample_b)])

    full_variant_sets_assignment = {"S1": str(full_a), "S2": str(full_b)}

    output_table = compute_jaccard_indices(str(lookup_table_path), full_variant_sets_assignment)

    rows = {row["paper_id"]: row for row in output_table.to_dicts()}

    assert rows["S1"]["n_cells"] == 50
    assert rows["S1"]["hash"] == "deadbeef"
    assert rows["S1"]["jaccard_index"] == pytest.approx(0.5)

    assert rows["S2"]["n_cells"] == 30
    assert rows["S2"]["hash"] == "cafebabe"
    assert rows["S2"]["jaccard_index"] == pytest.approx(1.0)
