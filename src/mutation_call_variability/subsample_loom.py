import numpy as np
import loompy
from pathlib import Path
import ctypes
from mutation_call_variability import commodity
libc = ctypes.CDLL("libc.so.6")

paper2datasetPID = {'S1': 'GFB-4397_GFB-4401', 'S2': 'GFB-4395_GFB-4399', 'S3': 'GFB-4396_GFB-4400'}

def subsample_loom(input_dataset_name: str, output_dataset_name: str) -> None:
    with loompy.connect(input_dataset_name, mode = 'r') as ds:
        n_cells = ds.shape[1]
        samplePID = commodity.file_name2metadata(output_dataset_name, 'paper_id')
        samplePID = paper2datasetPID.get(samplePID)
        n_cells_subsample = commodity.file_name2metadata(output_dataset_name, 'n_cells')
        cells_subsample = np.random.choice(np.arange(n_cells), size = n_cells_subsample, replace = False)
        cells_subsample = sorted(cells_subsample)

        with loompy.new(str(Path(output_dataset_name).with_suffix('.loom').resolve())) as dsout:
            for (_, _, view) in ds.scan(items=np.array(cells_subsample), axis=1, batch_size = 256):
                dsout.add_columns(view.layers, col_attrs=view.ca, row_attrs=view.ra)
                libc.malloc_trim(0)