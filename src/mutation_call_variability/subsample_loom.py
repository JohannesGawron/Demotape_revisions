"""
A script to downsample loom files to a subset of cells.
"""

import ctypes
from pathlib import Path

import loompy
import numpy as np

from mutation_call_variability import commodity

libc = ctypes.CDLL('libc.so.6')

paper2datasetPID = {
    'S1': 'GFB-4397_GFB-4401',
    'S2': 'GFB-4395_GFB-4399',
    'S3': 'GFB-4396_GFB-4400',
}
rng = np.random.default_rng()


def subsample_loom(input_dataset_name: str, output_dataset_name: str) -> None:
    """Subsample cells from a loom dataset and save the result to a new loom file. The number of cells in the subsampled loom is inferred from the output_dataset_name, which should follow the format: dataset.n_cells.hash.loom."""
    with loompy.connect(input_dataset_name, mode='r') as ds:
        n_cells = ds.shape[1]
        n_cells_subsample = commodity.file_name2metadata(output_dataset_name, 'n_cells')
        if not isinstance(n_cells_subsample, int):
            raise TypeError(
                f'The variable output_dataset_name needs to have the right format.\nCorrect format: dataset.INT.hash.loom.\n Currently: {output_dataset_name}'
            )

        cells_subsample = rng.choice(
            np.arange(n_cells), size=n_cells_subsample, replace=False
        )
        cells_subsample = np.sort(cells_subsample)

        with loompy.new(
            str(Path(output_dataset_name).with_suffix('.loom').resolve())
        ) as dsout:
            for _, _, view in ds.scan(
                items=np.array(cells_subsample), axis=1, batch_size=256
            ):
                dsout.add_columns(view.layers, col_attrs=view.ca, row_attrs=view.ra)
                libc.malloc_trim(0)
