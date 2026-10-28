"""Load the real Tapestri libraries analysed by the notebooks in this directory."""

import polars as pl

LIBRARY_DIR = '/cluster/work/bewi/members/jgawron/INTeRCePT/pipeline_output/mondrian/libraries'
LIBRARIES = [
    'QGF-211698_QGF-211701',
    'QGF-211699_QGF-211702',
    'QGF-211700_QGF-211703',
    'GFB-4395_GFB-4399',
    'GFB-4396_GFB-4400',
    'GFB-4397_GFB-4401',
]


def load_counts(path):
    """Return (depth, alt, gt) DataFrames: a `variant` column plus one column per cell, in identical row and column order."""
    df = pl.read_csv(path)
    cells = df.columns[7:]
    variant = pl.format('{}:{}:{}/{}', 'CHR', 'POS', 'REF', 'ALT').alias('variant')

    def field(cell, k):
        return pl.col(cell).str.split(':').list.get(k).cast(pl.Int64)

    depth = df.select(variant, *((field(c, 0) + field(c, 1)).alias(c) for c in cells))
    alt = df.select(variant, *(field(c, 1) for c in cells))
    gt = df.select(variant, *(field(c, 2) for c in cells))
    return depth, alt, gt


def load_data():
    """Return {library: (depth, alt, gt)} for all libraries."""
    return {
        b: load_counts(f'{LIBRARY_DIR}/{b}/dna/variants/{b}.filtered_variants.csv')
        for b in LIBRARIES
    }
