import argparse
import uuid
from pathlib import Path

import loompy
import numpy as np
import numpy.typing as npt
import polars as pl

dataset2paperID = {
    'GFB-4397_GFB-4401': 'S1',
    'GFB-4395_GFB-4399': 'S2',
    'GFB-4396_GFB-4400': 'S3',
}


def parse_args():
    """Parse command-line arguments for cell subsampling."""
    parser = argparse.ArgumentParser(
        description='Subsample cells and apply variant calling.'
    )
    parser.add_argument(
        '--input',
        type=str,
        nargs='+',
        required=True,
        help='Path to one or more input loom files.',
    )
    parser.add_argument(
        '--repetitions',
        type=int,
        default=1000,
        help='Number of subsampling repetitions.',
    )
    parser.add_argument(
        '--output_folder',
        type=str,
        required=True,
        help='Path to the output folder of the pipeline.',
    )
    parser.add_argument(
        '--output_folder_lookup',
        type=str,
        required=True,
        help='Path to the output folder for the lookup table.',
    )
    args = parser.parse_args()
    if isinstance(args.input, str):
        args.input = [args.input]

    return args


def get_paperPID(dataset_name: str) -> str:
    """Return the paper identifier for a dataset name."""
    dataset_name = Path(dataset_name).stem.split('.')[0]

    if dataset_name not in dataset2paperID:
        raise KeyError(f'Unknown dataset name: {dataset_name}')

    return dataset2paperID[dataset_name]


def get_n_cells(dataset_name: str) -> int:
    """Return the width of a loom dataset along axis 1, interpreted as the number of cells."""
    with loompy.connect(dataset_name) as ds:
        return int(ds.shape[1])


def create_subsample_index(
    samples: npt.NDArray[np.integer], input_file: str, args
) -> tuple[list[str], str]:
    """Create a list of output filenames for subsampled datasets from input file, by subsampling numbers of cells according to samples, and adds a unique hash to each filename."""
    paperPID = get_paperPID(input_file)
    output_filenames = [
        f'{args.output_folder}/{paperPID}.{sample}.{uuid.uuid4().hex[:3]}.filtered_variants.csv'
        for sample in samples
    ]

    return output_filenames, paperPID


def main(args):
    """This creates a lookup table for subsampled dataset names. Each `input` gets `repetitions` many file names placed into `output_folder`, and the resulting lookup is saved into `output_folder_lookup` under the name `subsample_lookup_table.tsv`."""
    output_table = pl.DataFrame(
        schema={
            'cell_count': pl.Int64,
            'dataset': pl.Utf8,
            'output_name': pl.Utf8,
        }
    )
    rng = np.random.default_rng()
    for input_file in args.input:
        n_cells = get_n_cells(input_file)
        samples = rng.uniform(1, n_cells + 1, args.repetitions).astype(int)
        output_filenames, paperPID = create_subsample_index(samples, input_file, args)
        output_table = output_table.vstack(
            pl.DataFrame(
                {
                    'cell_count': samples,
                    'dataset': [paperPID] * len(samples),
                    'output_name': output_filenames,
                }
            )
        )

    output_table.write_csv(
        f'{Path(args.output_folder_lookup)}/subsample_lookup_table.tsv', separator='\t'
    )


if __name__ == '__main__':
    args = parse_args()
    main(args)
