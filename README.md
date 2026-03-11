# HIPPS-DIMES GUI

A standalone Streamlit app for running HIPPS-DIMES and inspecting the results locally.

This repo is separate from the main HIPPS-DIMES codebase. The app calls the HIPPS-DIMES Python API directly, so you can launch runs, inspect matrices, review convergence, and explore structure, dynamics, and mechanics in one place.

## What the app does

- Runs `run_optimization()` directly instead of shelling out to the CLI
- Visualizes distance maps, contact maps, and connectivity matrices
- Compares target vs HIPPS-DIMES result in the Matrices tab
- Shows convergence curves from the iteration series
- Renders sampled 3D structures from returned `xyzs`
- Computes theory-based dynamics and mechanics plots from the final connectivity matrix

## Repo layout

The intended layout is:

```text
Code/
|-- HIPPS-DIMES
`-- HIPPS-DIMES-GUI
```

The app can try to import HIPPS-DIMES from the sibling repo automatically, but the clean setup is to install both repos into the same Python environment.

## Installation

### Recommended: mamba environment + uv install

```bash
mamba create -n hipps-dimes-gui python=3.11 pip
mamba activate hipps-dimes-gui
mamba install -c conda-forge uv

uv pip install -e /Users/guangshi/Library/CloudStorage/Dropbox/Documents/Work-Document/Code/HIPPS-DIMES
uv pip install -e /Users/guangshi/Library/CloudStorage/Dropbox/Documents/Work-Document/Code/HIPPS-DIMES-GUI
```

### uv virtual environment

```bash
cd /Users/guangshi/Library/CloudStorage/Dropbox/Documents/Work-Document/Code/HIPPS-DIMES-GUI
uv venv
source .venv/bin/activate
uv pip install -e ../HIPPS-DIMES
uv pip install -e .
```

### pip virtual environment

```bash
cd /Users/guangshi/Library/CloudStorage/Dropbox/Documents/Work-Document/Code/HIPPS-DIMES-GUI
python -m venv .venv
source .venv/bin/activate
pip install -e ../HIPPS-DIMES
pip install -e .
```

## Launch

```bash
cd /Users/guangshi/Library/CloudStorage/Dropbox/Documents/Work-Document/Code/HIPPS-DIMES-GUI
streamlit run app.py
```

If you are using a virtual environment, activate it first.

## Quick start

1. Set the top `Input file path` field to your file.
2. Choose the correct `Input type` and `Input format`.
3. For `cooler` and `.hic` inputs, fill in `Selection / region`.
4. Set optimization parameters in the sidebar.
5. Click `Run HIPPS-DIMES`.

## Input fields

### `Input file path`

This is the actual path that HIPPS-DIMES will use. If you already know the full path, paste it here directly.

Examples:

```text
/path/to/contact_map.txt
/path/to/contact_map.npy
/path/to/data.cool
/path/to/data.mcool::/resolutions/10000
/path/to/data.hic
```

### `Browse local files`

This is a filesystem navigator that helps fill in the top `Input file path`. It does not replace the top field.

- `Directory`: folder to browse, not a file path
- `Go`: open the folder typed in `Directory`
- `Up`: go to the parent folder
- `Home`: go to your home directory
- `Sync`: jump the browser to the folder implied by the current `Input file path`
- `Hidden`: show dotfiles and dotfolders
- `All files`: show all suffixes; otherwise the browser filters to likely HIPPS-DIMES inputs such as `.txt`, `.csv`, `.npy`, `.cool`, `.mcool`, `.hic`
- `Folders` + `Open folder`: navigate into a selected folder
- `Files` + `Use file`: copy the selected file path into the top `Input file path`

Important:

- If you paste a full file path into `Directory`, `Go` will not work, because that control expects a folder.
- For multires cooler files, use `Use file` first, then manually append the group to the top field, for example:

```text
/path/to/data.mcool::/resolutions/10000
```

## Input types and formats

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

## What appears in the tabs

### Matrices

- `Final distance map`: lower triangle is the target dmap, upper triangle is the HIPPS-DIMES result
- `Final contact map`: lower triangle is the target cmap, upper triangle is the HIPPS-DIMES result
- `Connectivity matrix`: final learned connectivity matrix

### Convergence

- Loss and entropy across iterations

### 3D Structure

- Sampled structures from the final connectivity matrix

### Dynamics and Mechanics

- Theory-based observables derived from the final connectivity matrix

## Practical notes

- For large `.cool` and `.hic` files, path-based loading is better than a browser upload widget.
- If GPU support is available in the environment, the sidebar exposes it automatically.
- The default sample path points to `../HIPPS-DIMES/data/IMR90_chr21-28-30Mb.csv` when that file exists.
- If XYZ writing fails for a run, try clearing `Output prefix` or turning on `Skip XYZ generation`.

## Troubleshooting

### I pasted a file path into `Directory` and nothing happened

`Directory` expects a folder path. Paste the file path into the top `Input file path` field instead, or paste the parent folder into `Directory` and browse from there.

### My `.cool` or `.hic` input does not run

Check:

- `Input type = cmap`
- `Input format` matches the file
- `Selection / region` is filled in for `cooler` and `hic`

### My `.mcool` file does not work directly

Use the multires group syntax in `Input file path`, for example:

```text
/path/to/data.mcool::/resolutions/10000
```
