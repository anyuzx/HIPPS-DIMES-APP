# HIPPS-DIMES APP

Streamlit workbench for running HIPPS-DIMES locally and inspecting the results interactively.

This repository contains the app layer only. The numerical model and optimization code live in the main `HIPPS-DIMES` repository. The app calls the HIPPS-DIMES Python API directly, so you can configure runs, launch optimization, and inspect the returned matrices, structures, dynamics, and mechanics from one interface.

## Features

- Run `run_optimization()` directly from a Streamlit UI
- Inspect distance maps, contact maps, and connectivity matrices
- Compare target matrices against HIPPS-DIMES output
- Review convergence curves and live optimization progress
- Visualize sampled 3D structures
- Compute dynamics and mechanics observables from the final connectivity matrix
- Browse local files and inspect available `.cool` / `.mcool` groups before launching a run

## Repository scope

- This repo is a local desktop app, not a hosted web service.
- Input files are read from your local filesystem.
- No data is uploaded anywhere by the app itself.
- You still need the main `HIPPS-DIMES` package installed in the same Python environment.

## Recommended layout

Clone the app repo next to the core repo:

```text
workspace/
|-- HIPPS-DIMES
`-- HIPPS-DIMES-APP
```

The app can also import HIPPS-DIMES from an installed package, but the side-by-side layout above is the simplest setup for development.

## Installation

### Option 1: `uv` virtual environment

```bash
git clone <HIPPS-DIMES-url>
git clone <HIPPS-DIMES-APP-url>

cd HIPPS-DIMES-APP
uv venv
source .venv/bin/activate
uv pip install -e ../HIPPS-DIMES
uv pip install -e .
```

### Option 2: `pip` virtual environment

```bash
git clone <HIPPS-DIMES-url>
git clone <HIPPS-DIMES-APP-url>

cd HIPPS-DIMES-APP
python -m venv .venv
source .venv/bin/activate
pip install -e ../HIPPS-DIMES
pip install -e .
```

### Option 3: Conda or mamba environment

```bash
git clone <HIPPS-DIMES-url>
git clone <HIPPS-DIMES-APP-url>

mamba create -n hipps-dimes-app python=3.11 pip
mamba activate hipps-dimes-app
pip install -e ./HIPPS-DIMES
pip install -e ./HIPPS-DIMES-APP
```

## Launch

```bash
cd HIPPS-DIMES-APP
streamlit run app.py
```

## Quick start

1. Set `Input file path` to a local file.
2. Choose the correct `Input type` and `Input format`.
3. For `cooler` and `.hic` inputs, fill in `Selection / region`.
4. Adjust optimization parameters in the sidebar.
5. Click `Run HIPPS-DIMES`.

## Supported inputs

### Contact maps

Use:

- `Input type = cmap`
- `Input format = text`, `npy`, `cooler`, or `hic`

For `cooler` and `hic`, `Selection / region` is required.

### Distance maps

Use:

- `Input type = dmap`
- `Input format = text` or `npy`

### Squared distance maps

Use:

- `Input type = ddmap`
- `Input format = text` or `npy`

## File browser and cooler groups

The sidebar includes a local filesystem browser to help populate `Input file path`.

- `Directory`: browse a folder
- `Go`, `Up`, `Home`, `Sync`: navigate quickly
- `Files` -> `Use file`: copy the selected file into `Input file path`
- `Cooler groups / resolutions`: inspect available groups in `.cool` / `.mcool` files and append the selected group to `Input file path` automatically

Examples:

```text
/path/to/contact_map.txt
/path/to/contact_map.npy
/path/to/data.cool
/path/to/data.mcool::/resolutions/10000
/path/to/data.hic
```

## App tabs

### Overview

Run metadata, output availability, and download actions.

### Matrices

- Final distance map
- Final contact map
- Final connectivity matrix
- Target vs HIPPS-DIMES matrix comparisons

### Convergence

Loss and entropy across optimization iterations.

### 3D Structure

Sampled structures from the final connectivity matrix.

### Dynamics

Theory-based dynamics observables derived from the optimized model.

### Mechanics

Bulk and per-locus modulus calculations from the optimized connectivity matrix.

## Notes

- Large `.cool` and `.hic` inputs are better handled as local paths than as browser uploads.
- GPU options appear automatically when HIPPS-DIMES detects a supported GPU environment.
- The app keeps the most recent run in memory for interactive inspection.
- For noisy runs with `save_steps`, HIPPS-DIMES requires an `Output prefix`.

## Troubleshooting

### The app cannot import HIPPS-DIMES

Make sure both repositories are installed into the same environment:

```bash
pip install -e ../HIPPS-DIMES
pip install -e .
```

### My `.mcool` file does not run

Make sure:

- `Input type = cmap`
- `Input format = cooler`
- `Selection / region` is filled in
- the correct cooler group or resolution is selected in the sidebar

### My `.hic` input does not run

Check:

- `Input type = cmap`
- `Input format = hic`
- `Selection / region` is filled in
- `Hi-C binsize`, normalization, and unit match the file
