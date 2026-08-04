# mutation_call_variability

Investigating how mutation calls vary in single-cell DNA sequencing data as a
function of cell count.

## Background

This project addresses concerns raised in the revisions of **DemoTape**, regarding
the claim that clonal structures and evolutionary patterns inferred from
multiplexed data closely match those from individually sequenced samples.

The reviewers noted noticeable discrepancies in SNP detection between
demultiplexed and individually sequenced samples, which could indicate
misidentified mutations that in turn might affect downstream clonal inference.
While some of these differences are attributable to filtering thresholds, the
analysis here tests the alternative explanation: that the discrepancies are
driven simply by **cell count**, and do not introduce systematic biases in
genotype reconstruction.

The approach is to repeatedly subsample cells from the full datasets, re-run
variant calling on each subsample, and measure the concordance between the
variants called on the subsample and those called on the full dataset
(Jaccard index). If the discrepancies are a cell-count effect, concordance should
depend on the number of cells and nothing else.

Input data are MissionBio Tapestri single-cell DNA panel datasets in `.loom`
format. Three samples are analysed, named as in the DemoTape paper:

| Paper ID | Dataset PID         |
| -------- | ------------------- |
| `S1`     | `GFB-4397_GFB-4401` |
| `S2`     | `GFB-4395_GFB-4399` |
| `S3`     | `GFB-4396_GFB-4400` |

## Repository layout

```
src/mutation_call_variability/
├── create_lookup_table.py   # Draw random subsample sizes -> lookup table (TSV)
├── subsample_loom.py        # Subsample cells from a loom file into a new loom file
├── mosaic_processing.py     # MissionBio-mosaic-equivalent variant filtering (CLI)
├── jaccard_index.py         # Jaccard index between two variant call sets
└── commodity.py             # Shared helpers (lookup table I/O, filename parsing)

workflow/
├── Snakefile                # Pipeline definition
├── config/params.yaml       # Input/output paths, mosaic filtering thresholds
├── config/subsample_lookup_table.tsv  # Generated: one row per subsample
└── profiles/config.yaml     # SLURM executor profile (account, retries, defaults)

notebooks/
└── regression.qmd           # Statistical analysis of the preprocessed data
```

## Setup

Dependencies are managed with [pixi](https://pixi.sh/). The workspace defines a
single `default` environment and targets `linux-64` only (it is meant to run on
the ETH Euler cluster).

```bash
pixi install
```

### Cluster-specific configuration

**Point pixi's cache at scratch.** With `PIXI_CACHE_DIR` on `/cluster/project`,
`pixi install` can hang until the uv lock times out (300 s), and even plain
`pixi run` invocations become erratically slow (observed 0.4 s vs. 37 s for the
same no-op command). Add this to your shell profile on the cluster:

```bash
export PIXI_CACHE_DIR=/scratch/$USER-pixi-cache
```

**`snakemake-executor-plugin-slurm-jobstep` is pinned to `<0.6.1`.** Version
0.6.1 strips `SLURM_MEM_*` environment variables before launching the nested
`srun` step (intended to fix a GPU-only conflict, but applied unconditionally).
The nested step then cannot inherit the job's memory allocation and falls back to
the cluster's default mem-per-cpu, so jobs are OOM-killed at ~1 GB regardless of
what `mem_mb_per_cpu` requests. Do not lift this pin without re-testing;
`repro_oom_bug/` contains a minimal reproduction. For details, see this [PR](https://github.com/snakemake/snakemake-executor-plugin-slurm-jobstep/pull/51).

## Running the analysis

### 1. Generate the subsample lookup table

Draws `--repetitions` random cell counts per input dataset and writes
`subsample_lookup_table.tsv`, which defines the pipeline's targets. Each output
filename carries a short random hash so that repeated draws of the same cell
count do not collide.

```bash
pixi run python src/mutation_call_variability/create_lookup_table.py \
    --input <sample1>.cells.loom <sample2>.cells.loom <sample3>.cells.loom \
    --repetitions 100 \
    --output_folder_lookup workflow/config/ \
    --output_folder results
```

See `workflow/config/README` for the exact invocation used to produce the
committed lookup table.

### 2. Run the workflow

```bash
pixi run workflow-dry   # dry run: show what would be executed
pixi run workflow       # submit rules as SLURM jobs
```

The pipeline has four rules:

| Rule                     | Does                                                              |
| ------------------------ | ----------------------------------------------------------------- |
| `subsample_loom`         | Subsample cells from the full loom file to the target cell count   |
| `mosaic_subsample`       | Variant calling + filtering on each subsampled loom file           |
| `mosaic_full`            | Variant calling + filtering on the full (un-subsampled) dataset    |
| `compute_jaccard_indices`| Jaccard index of each subsample's variant set vs. its full dataset |

Final output is `<output_directory>/jaccard_indices.tsv` with columns `hash`,
`paper_id`, `n_cells`, `jaccard_index`.

Memory is requested via `mem_mb_per_cpu` and scales with the retry attempt
(`get_mem_mb` in the Snakefile doubles it each time), because peak usage varies
with subsample size. Euler rejects `--mem` (memory-by-node), so resources must be
expressed per-CPU.

### 3. Analyse the results

```bash
pixi run docs   # renders notebooks/ with Quarto
```

`notebooks/regression.qmd` checks that the drawn cell counts really are uniformly
distributed (Q-Q plots, one panel per sample) and then regresses the Jaccard
index on cell count.

## Development

```bash
pixi run lint            # ruff check
pixi run format          # ruff format
pixi run typecheck       # pyright (fast, local)
pixi run typecheck-ci    # mypy (used in CI/pre-commit)
pixi run docstring-cov   # interrogate, --fail-under=80
pixi run test            # pytest
pixi run check           # all of the above
pixi run hooks           # install pre-commit hooks
```

Note that `tests/` is currently empty, so `pixi run test` (and therefore
`pixi run check`) collects nothing.

## License

MIT — see [LICENSE](LICENSE).
