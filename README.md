# DemoTape revision

Analyses carried out in response to the reviewers' comments on the **DemoTape**
paper.

Aspects covered so far:

- **Mutation call variability** — how mutation calls in single-cell DNA
  sequencing data vary as a function of cell count (see below).
- **Pool simulation** — synthetic pools of three patients with known ground
  truth, to benchmark demultiplexing (see [Pool simulation](#pool-simulation)).

## Mutation call variability

### Background

This analysis addresses concerns raised in the revisions of **DemoTape**, regarding
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

## Pool simulation

`src/pool_simulation/simulate_pool.py` simulates one pool of three patients
(`pat0`, `pat1`, `pat2`) with known ground truth, in DemoTape's input format.
The genotypes are synthetic; read depths are resampled from real libraries, and
the alt reads follow models fitted in `notebooks/`.

### Usage

**Environment.** The commands below use the pixi environment (`pixi run …`).
The simulation scripts only need Python, numpy and polars, so a small conda
environment works as well; `envs/pool_simulation.yml` lists these dependencies,
with versions as in `pixi.toml`:

```bash
pixi install
#Alternatively with conda: conda env create -f envs/pool_simulation.yml
#conda activate pool_simulation
#PYTHONPATH=src python -m pool_simulation.simulate_pool  
```

`PYTHONPATH=src` is needed with conda because the conda environment does not install this
repository's package (pixi does).

**1. Export the real read depths (once).** In order to realistically simulate read depths, I create a pool of read depths from real data that approximates the read depth distribution. `export_depth.py` combines the
`*.filtered_variants.csv` files of one or more libraries into one depth matrix.
It contains only depths (no header, positions, alt counts, genotypes or
barcodes), with rows and columns shuffled. Re-run it only to change the
libraries.

```bash
L=/cluster/work/bewi/members/jgawron/INTeRCePT/pipeline_output/mondrian/libraries
pixi run python -m pool_simulation.export_depth \
    --inputs $(for b in GFB-4395_GFB-4399 GFB-4396_GFB-4400 GFB-4397_GFB-4401; do echo $L/$b/dna/variants/$b.filtered_variants.csv; done) \
    --seed 0 --output results/depth.csv.gz
```

| Option | Description |
| --- | --- |
| `--inputs` | `*.filtered_variants.csv` files, one per library (required) |
| `--seed` | Random seed for the shuffling |
| `--output` | Output file; gzipped if it ends in `.gz` (required) |

**2. Simulate a pool.**

```bash
pixi run python -m pool_simulation.simulate_pool \
    --n_same 50 --n_diff 20 \
    --cells_per_sample 1500 1500 1500 --doublet_rate 0.25 \
    --depth_source results/depth.csv.gz \
    --seed 0 --output results/pool_same50_diff20_seed0.csv
```

| Option | Description |
| --- | --- |
| `--n_same` | Number of sites where pat1 and pat2 have the same genotype (required) |
| `--n_diff` | Number of sites where pat1 and pat2 have different genotypes (required) |
| `--cells_per_sample` | Number of singlets for pat0, pat1 and pat2; unequal numbers give unbalanced mixing. If omitted, drawn uniformly from 500–3000 per sample |
| `--doublet_rate` | Fraction of all cells that are doublets, added on top of the singlets. If omitted, drawn uniformly from 0.05–0.3 per run |
| `--depth_source` | Depth matrix from step 1 (required) |
| `--seed` | Random seed; the same seed and options reproduce the same pool, including drawn cell numbers and doublet rate |
| `--output` | Output CSV file (required) |

For a grid over the number of shared (x) and distinguishing (y) sites, with
several repetitions each:

```bash
for x in 1 2 5 10; do
  for y in 1 2 5 10; do
    for seed in 0 1 2 3 4 5 6 7 8 9; do
      pixi run python -m pool_simulation.simulate_pool \
          --n_same $x --n_diff $y --depth_source results/depth.csv.gz \
          --seed $seed --output results/pools/pool_same${x}_diff${y}_seed${seed}.csv
    done
  done
done
```

**3. Demultiplex with DemoTape.**

```bash
pixi run python /cluster/work/bewi/members/jgawron/INTeRCePT/mincedTapestri/workflow/scripts/demultiplex_distance.py \
    --input results/pool_same50_diff20_seed0.csv -n 3 \
    --output results/pool_same50_diff20_seed0_demultiplexed --output_plot
```

This DemoTape version reads the ground truth from the cell names (see the
output format below) and needs pandas < 3, which `pixi.toml` pins.

When I set the flag `--output_plot`, the demultiplexing script `demultiplex_distance.py` (commit #93ec5c4) crashed. 

**Local fix in DemoTape (not committed in the mincedTapestri repository):**
with ground-truth labels in the cell names, `--output_plot` crashed in
`plot_heatmap` (`TypeError: '<=' not supported between instances of 'int' and
'list'`), because line 410 nested the row colours one level too deep for
seaborn's `clustermap`. The line was changed from

```python
r_colors = [get_row_cols(self.true_cl), get_row_cols(assignment_str)]
```

to

```python
r_colors = [*get_row_cols(self.true_cl), *get_row_cols(assignment_str)]
```

The heatmap now shows four colour columns (true cluster, true doublet partner,
assigned cluster, assigned doublet partner). Only the plot is affected; the
assignments and profiles are unchanged.

**4. Evaluate the demultiplexing.** `evaluate_demultiplexing.py` computes the
adjusted Rand index (ARI) between the true patient and DemoTape's assigned
cluster, for pat1 and pat2 singlets only: how well these two patients are
separated. pat0 and all doublets are ignored as nuisance. A pat1/pat2 cell that
DemoTape assigns to the pat0 cluster or to a doublet cluster counts as an
error. It takes any number of `*.assignments.tsv` files and writes one row per
file (`file`, `n_cells`, `ari_pat1_pat2`).

```bash
pixi run python -m pool_simulation.evaluate_demultiplexing \
    --assignments results/pool_same50_diff20_seed0_demultiplexed.assignments.tsv \
    --output results/ari.tsv
```

### Output format

A CSV with columns `CHR,POS,REF,ALT,REGION,NAME,FREQ`, as in Mosaic's
`filtered_variants.csv`, followed by one column per cell:

- **Variant columns:** `CHR` is `1` (without `chr`),
  `POS`, `REF`/`ALT` are random distinct bases, `REGION` and `NAME`
  are a synthetic amplicon per site (`AMPL1`, `AMPL2`, …), and `FREQ` is 0.
- **Cell columns:** named `cell<i>.<ground truth>`, where the ground truth is
  `pat0`, `pat1` or `pat2` for singlets and `patI+patJ` for doublets,
  including homotypic ones such as `pat0+pat0`.
- **Entries:** `REF:ALT:GT`, with the simulated read counts and a constant GT
  of `3`, which DemoTape ignores.

### Model

- Genotypes are germline only. pat1 and pat2 have the same genotype at exactly
  `--n_same` sites and different genotypes at exactly `--n_diff` sites. pat0 is
  random, but no site is identical in all three patients (such sites would be
  filtered out).
- Doublets are added on top of the singlets so that exactly `--doublet_rate` of
  all cells (barcodes) are doublets. Each doublet's patients are drawn in proportion to
  the mixing ratio. As in the previous DemoTape paper, a doublet's genotype is
  that of its two cells if they agree, and heterozygous otherwise (0+1, 0+2 and
  1+2 all become 1); its depth is the rounded mean of the two cells' depths,
  and its reads are simulated from that genotype and depth.
- Sites failing the Mosaic depth filter (depth ≥ 10 in ≥ 50% of cells) are
  re-simulated, so all sites are well covered.
- Read depths are resampled from a real depth matrix (`--depth_source`):
  each site takes the depths of a real locus, each cell those of a real cell.
  The matrix can stack several libraries block by block (`nan` where a cell's
  library lacks a locus); each cell then gets one real cell per library, and
  library effects are not modelled (they are negligible, see
  `notebooks/read_depth.qmd`). A negative-binomial model missed the depth cap
  at about 50 reads and the dropout.
- Alternative reads at heterozygous entries follow a beta-binomial mixture with
  allelic dropout, with parameters (m, kappa, d, eps) drawn per site from
  Gaussians fitted to per-locus estimates (`HET_MEAN`, `HET_SD`; see
  `notebooks/het_sites.qmd`). Homozygous entries are binomial with an error
  rate drawn per site, logit(error) ~ Normal(-6.376, 1.288) (`HOM_ERROR_MU`,
  `HOM_ERROR_TAU`), estimated at homozygous mutated sites and used for both
  homozygous genotypes (see `notebooks/hom_sites.qmd`).

## Repository layout

```
src/mutation_call_variability/
├── create_lookup_table.py   # Draw random subsample sizes -> lookup table (TSV)
├── subsample_loom.py        # Subsample cells from a loom file into a new loom file
├── mosaic_processing.py     # MissionBio-mosaic-equivalent variant filtering (CLI)
├── jaccard_index.py         # Jaccard index between two variant call sets
└── commodity.py             # Shared helpers (lookup table I/O, filename parsing)

envs/
└── pool_simulation.yml      # Conda environment for the pool simulation

src/pool_simulation/
├── export_depth.py          # Export real libraries as one anonymised depth matrix (CLI)
├── simulate_pool.py         # Simulate a 3-patient pool in DemoTape input format (CLI)
└── evaluate_demultiplexing.py  # ARI of pat1 vs. pat2 in DemoTape's assignments (CLI)

workflow/
├── Snakefile                # Pipeline definition
├── config/params.yaml       # Input/output paths, mosaic filtering thresholds
├── config/subsample_lookup_table.tsv  # Generated: one row per subsample
└── profiles/config.yaml     # SLURM executor profile (account, retries, defaults)

notebooks/
├── _quarto.yml              # Quarto project, so that freeze: auto skips unchanged notebooks
├── regression.qmd           # Statistical analysis of the preprocessed data
├── load_data.py             # Shared loader for the real libraries (read depth, het sites)
├── read_depth.qmd           # Read depth model; why the simulator resamples real depths
├── het_sites.qmd            # Beta-binomial + allelic dropout model at heterozygous sites
└── hom_sites.qmd            # Error rate at homozygous mutated and wildtype sites
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

## Running the analysis (mutation_call_variability only)

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
pixi run notebooks     # renders notebooks/ with Quarto (only changed notebooks are re-executed)
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
