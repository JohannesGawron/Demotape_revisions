from pathlib import Path
import polars as pl

def load_lookup_table(lookup_table_file: str) -> list[str]:
    lookup_table = pl.read_csv(lookup_table_file, separator='\t')

    return lookup_table['output_name'].to_list()

def file_name2metadata(file_name: str, metadata_key: str) -> str:
    """Extract metadata from a file name based on the provided key."""
    metadata_mapping = {
        'paper_id': 0,
        'n_cells': 1,
        'hash': 2
    }
    try:
        metadata_idx = metadata_mapping.get(metadata_key)
    except KeyError:
        raise ValueError(f"Invalid metadata_key: {metadata_key}. Valid keys are: {list(metadata_mapping.keys())}")
    
    return Path(file_name).stem.split('.')[metadata_idx] if metadata_idx != 1 else int(Path(file_name).stem.split('.')[metadata_idx])