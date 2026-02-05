# ML-based-surrogate-fluid-flow-modeling-for-airfoil~# 2D_ML_CFD

Short description
- 2D_ML_CFD is a small research codebase to generate 2D airfoil geometries, produce signed-distance fields (SDFs), run (template) OpenFOAM cases to create CFD fields on a regular grid, and train a U-Net style model to predict velocity/pressure fields from SDF inputs.

Repository layout
- `data_gen_auto.py`: orchestrates CFD runs (OpenFOAM) using an `airFoil2D` template case, interpolates results onto a grid and saves `u.npy`, `v.npy`, `p.npy`, and `sdf.npy` per case.
- `naca_airfoil_generation.py`: generates NACA-4 airfoil coordinate arrays and saves them to `airfoils_npy`.
- `sdf_gen.py`: generates grid-aligned signed-distance fields (`*_sdf.npy`) from airfoil coordinates.
- `npy_to_stl.py`: converts 2D `.npy` coordinate files into thin 3D `.stl` meshes for meshing.
- `npy_to_stl.py`, `naca_airfoil_generation.py`, `sdf_gen.py`, `data_gen_auto.py` form the pipeline that produces dataset entries.
- `dataset_utils.py`: `AirfoilCFDDataset` dataset loader used by training and testing scripts.
- `model_arch_attn.py`: UNet-like model (with attention gates). The code expects an `AirfoilUNet` model.
- `train.py`: training loop to train the model on dataset folders.
- `test.py`: quick visualization script to load a checkpoint and compare predictions with ground truth.
- `airFoil2D/`: OpenFOAM template case used by `data_gen_auto.py`.

Quick requirements
- Python 3.8+
- PyTorch (tested with 1.10+)
- numpy, matplotlib, shapely, trimesh, pyvista
- OpenFOAM (system `blockMesh`, `snappyHexMesh`, `simpleFoam`, `surfaceFeatures` must be on PATH) for `data_gen_auto.py`

Install (minimal)
```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy matplotlib shapely trimesh pyvista torch torchvision tqdm
```
If you plan to run `data_gen_auto.py` you must also install and configure OpenFOAM separately (not handled by this repo).

Dataset expectations
- The training code expects a `dataset/` directory (see `DATASET_DIR` in `train.py` and `test.py`).
- Each case should be a directory containing at least:
  - `u.npy`, `v.npy`, `p.npy` — 2D arrays (H x W)
  - `sdf.npy` — signed-distance field (H x W)
  - optional `meta.json`, and preview PNGs

Typical data pipeline
1. Generate airfoil coordinates: `python naca_airfoil_generation.py` → `airfoils_npy/` (.npy files)
2. Generate SDFs from coordinates: `python sdf_gen.py` → `sdf_airfoil/` (or `sdf_airfoil_new/` depending on script)
3. Convert `.npy` to `.stl` for meshing: `python npy_to_stl.py` → `stl_airfoil/`
4. Run CFD + interpolate onto grid: `python data_gen_auto.py` → creates per-case folders under `dataset_new/` (this step requires OpenFOAM and can be slow)

Training
- Edit `DATASET_DIR` in `train.py` to point to your dataset directory.
- Example:
```bash
python train.py
```
- Checkpoints are saved to `./checkpoints` (see `CHECKPOINT_DIR` in `train.py`).

Testing / Visualization
- `test.py` loads `./checkpoints/best_model.pt` by default; adjust `CHECKPOINT` and `DATASET_DIR` as needed.
```bash
python test.py
```

Notes & troubleshooting
- `train.py` / `test.py` import `AirfoilUNet` from `model_arch`. In this repo the file is named `model_arch_attn.py`. Rename the file to `model_arch.py` or change the imports in `train.py`/`test.py` to `from model_arch_attn import AirfoilUNet`.
- `data_gen_auto.py` uses `pyvista.OpenFOAMReader` to read the final time step. Ensure OpenFOAM writes a case file with `.OpenFOAM` marker and that PyVista can access your OpenFOAM installation.
- Running `data_gen_auto.py` will call `blockMesh`, `surfaceFeatures`, `snappyHexMesh`, and `simpleFoam` in your shell. These require correct OpenFOAM setup and privileges.

Suggested next steps
- Verify/rename `model_arch_attn.py` to `model_arch.py` or update imports.
- Create a small synthetic dataset entry to test `dataset_utils.py` / `train.py` quickly before running the full CFD pipeline.

License & contact
- This repository contains research code. No license file included — add one if you intend to share publicly.

If you'd like, I can:
- run quick fixes (rename `model_arch_attn.py` → `model_arch.py`) to make imports consistent,
- create a `requirements.txt` and a small example dataset entry to test training.
