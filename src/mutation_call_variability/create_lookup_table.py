import numpy as np
from numpy.typing import NDArray
import loompy
import argparse
import logging
import polars as pl
from pathlib import Path
import uuid
import yaml

dataset2paperID = {'GFB-4397_GFB-4401': 'S1', 'GFB-4395_GFB-4399': 'S2', 'GFB-4396_GFB-4400': 'S3'}

def parse_args():
    '''Parse command-line arguments for cell subsampling.'''
    parser = argparse.ArgumentParser(description='Subsample cells and apply variant calling.')
    parser.add_argument('--input', type=str, nargs='+', required=True, help='Path to one or more input loom files.')
    parser.add_argument('--repetitions', type=int, default=1000, help='Number of subsampling repetitions.')
    parser.add_argument('--output_folder', type=str, required=True, help='Path to the output folder of the pipeline.')
    parser.add_argument('--output_folder_lookup', type=str, required=True, help='Path to the output folder for the lookup table.')
    args = parser.parse_args()
    if isinstance(args.input, str):
        args.input = [args.input]

    return args


def get_paperPID(dataset_name: str) -> str:
    '''Return the paper identifier for a dataset name.'''
    dataset_name = Path(dataset_name).stem.split('.')[0]
    
    return dataset2paperID.get(dataset_name)


def get_n_cells(dataset_name: str) -> int:
    with loompy.connect(dataset_name) as ds:
        return ds.shape[1]


def create_subsample_index(samples: np.ndarray[int], input_file: str, args) -> tuple[list[str], str]:
    paperPID = get_paperPID(input_file)
    output_filenames = [
        f'{args.output_folder}/{paperPID}.{sample}.{uuid.uuid4().hex[:3]}.filtered_variants.csv'
        for sample in samples
    ]
    
    return output_filenames, paperPID

def main(args):
        output_table = pl.DataFrame(
            schema={
                'cell_count': pl.Int64,
                'dataset': pl.Utf8,
                'output_name': pl.Utf8,
            }
        )
        for input_file in args.input:
            n_cells = get_n_cells(input_file)
            samples = np.random.uniform(1, n_cells + 1, args.repetitions).astype(int)
            output_filenames, paperPID = create_subsample_index(samples, input_file, args)
            output_table = output_table.vstack(pl.DataFrame({'cell_count': samples, 'dataset': [paperPID] * len(samples), 'output_name': output_filenames}))

        output_table.write_csv(f'{Path(args.output_folder_lookup)}/subsample_lookup_table.tsv', separator = '\t')        

if __name__ == '__main__':     
    args = parse_args()
    main(args)






