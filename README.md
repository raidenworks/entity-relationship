# ERD Pipeline

A lightweight pipeline to profile CSV tables, infer primary/foreign keys, and render:

- An ERD diagram (`ERD.svg`) with table schemas and links.
- An edges diagram (`EDGES.svg`) based on the generated `edges.csv`.
- A schema table (`tables_schema.csv`) when ERD rendering is enabled.

## Quick Start

1. Create the environment (online build machine):
   - Conda (recommended): `conda env create -f environment.yml`
   - Activate: `conda activate erd-env`

2. Configure inputs in `config.yml`:
   - `input_csv_dir`: path to your CSV folder
   - `csv_glob`: filename pattern for CSVs
   - `output_dir`: where artifacts are written

   Rendering toggles:
   - `render_erd`: when true, produces `tables_schema.csv` and `ERD.svg`
   - `render_edges_diagram`: when true, produces `EDGES.svg`

3. Run the pipeline:
   - `python run_erd_pipeline.py --config config.yml`

4. Outputs (under `output_dir`):
   - `edges.csv` (always)
   - `EDGES.svg` (if `render_edges_diagram: true`)
   - `tables_schema.csv`, `ERD.svg` (if `render_erd: true`)
   - `pipeline.log`

## Offline Deployment (conda-pack)

### Conda environment set-up

- On the online machine:
  - run `conda pack -n erd-env -o erd-env.tar.gz` to pack your conda environment
  - download this repository as a zip file
- On the offline target machine:
  - place the generated `erd-env.tar.gz` file into your target folder, e.g. `erd`
  - unzip the repository into `erd` folder
  - double-click `unpack_env.bat` to execute the batch file which extracts `erd-env.tar.gz` to an `erd-env` subfolder which then runs conda-unpack
  ```
  erd
  ├ output
  | └ .gitkeep
  ├ .gitattributes
  ├ .gitignore
  ├ config.yml
  ├ environment.yml
  ├ erd-env.tar.gz
  ├ README.md
  ├ run_erd_pipeline.py
  ├ run_pipeline.bat
  └ unpack_env.bat
  ```

### Run script

- Activate: `conda activate erd-env`
  - Navigate to where you place this repo and
  - Run `python.exe run_erd_pipeline.py --config config.yml`\
  or
  - Double-click `run_pipeline.bat`

## Performance & Heuristics

- Fast datetime parsing with format/epoch detection; falls back to chunked parsing with progress if needed.
- Sampling-based type inference to avoid full-column scans:
  - `heuristics.COLUMN_SAMPLE_SIZE`: number of random non-null rows per column
  - `heuristics.SAMPLE_RANDOM_SEED`: set for reproducible sampling
  - `heuristics.INFERENCE_THRESHOLD`: classification threshold on samples
  - `heuristics.DATE_LIKE_THRESHOLD`: pre-check to attempt datetime parsing only when likely

## Coverage Metric

Coverage in both ERD/EDGES diagrams is unique-based: the fraction of unique child keys present in the parent key domain.

## Notes

- Generated artifacts in `output/` and logs are ignored by git (see `.gitignore`).
- Adjust `environment.yml` to your Python/pandas versions as needed (pandas >= 2.2 recommended).


## Windows Helpers

- Unpack a conda-packed env into `erd-env` next to the tarball and fix paths:
  - `unpack_env.bat path\\to\\erd-env.tar.gz`

- Run the pipeline with the packed env:
  - `run_pipeline.bat [config.yml]`
  - Defaults to `config.yml` in the repo root if omitted.




