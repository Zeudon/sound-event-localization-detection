## Copilot / AI agent quick instructions for this repo

Purpose: short, actionable guidance so an AI coding agent can be productive immediately.

- Entry points
  - `SMR_SELD.ipynb` — primary working artifact. Open and run cells to reproduce experiments. Inspect the notebook's top cells to learn required imports and runtime assumptions.

- Data layout and important paths
  - `foa_dev/` — contains dev train/test folders for SONY/TAU. Typical subfolders: `dev-train-sony/`, `dev-test-tau/`, etc.
  - `metadata_dev/` — CSV metadata organized by dataset split. Example filenames: `fold4_room23_mix001.csv`, `fold4_room24_mix016.csv`, `fold4_room10_mix001.csv`. Filenames follow pattern `fold{N}_room{M}_mix{XXX}.csv`.
  - When referencing data, use the relative paths above and preserve existing CSV name patterns; many scripts and the notebook expect them.

- Project structure & big picture
  - The repo is notebook-first: experiments and data exploration are in `SMR_SELD.ipynb` rather than packaged scripts.
  - Dataflow: CSV metadata in `metadata_dev/*` points to raw audio / mixes under `foa_dev/*` (train/test folders). The notebook consumes metadata CSVs to build datasets and launch training/evaluation runs.

- Conventions and patterns the agent should follow
  - Do not re-organize or rename `metadata_dev` CSV files — callers (notebook and potential scripts) use exact filenames.
  - Prefer minimal, reversible edits: add a small helper Python module (e.g., `src/utils.py`) when extracting repeated logic from the notebook.
  - Keep changes local and testable: add a tiny notebook cell or a short script that demonstrates correctness against a small CSV subset.

- Developer workflows discovered
  - No repo-level dependency file found (no `requirements.txt`, `pyproject.toml`, `environment.yml`, or `setup.py`). To figure required packages, inspect the import cells in `SMR_SELD.ipynb` and ask the user before installing system-level packages.
  - Primary validation is running notebook cells and confirming data reads from `metadata_dev/` without error.

- Integration points & external dependencies
  - The code relies on local CSV metadata and local audio/mix files under `foa_dev/*`. There are no obvious external service APIs in the repo.

- Useful, concrete search examples for the agent
  - Search the repo for `fold` or `metadata_dev` to find where code reads CSV files.
  - Look for `SMR_SELD` or cell-level comments in `SMR_SELD.ipynb` to find experiment hyperparameters.

- What not to change without confirmation
  - Do not modify CSV filenames or move `foa_dev` / `metadata_dev` directories.
  - Do not run wide-ranging installs or network calls; ask the user for the environment to use.

- When you make changes
  - Add a one-line README or a notebook cell that documents how to run the change (inputs/outputs, example command or cell order).
  - Prefer adding small unit tests or a short demonstrator script (e.g., `scripts/check_metadata.py`) that reads a sample CSV and prints basic schema validation.

If anything here is unclear or you want the instructions to emphasize a different workflow (for example, converting the notebook into python scripts, adding dependency manifests, or creating CI), tell me which direction you prefer and I will update or extend this file.
