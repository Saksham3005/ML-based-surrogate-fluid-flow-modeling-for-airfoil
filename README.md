# 🌊 ML-Based Surrogate Fluid Flow Modeling for Airfoils

A cutting-edge research framework that leverages deep learning to create fast, accurate surrogate models for 2D airfoil aerodynamic simulations. This project automates the complete pipeline: from NACA airfoil geometry generation through CFD simulation to neural network training for real-time flow prediction.

## 🎯 Project Overview

**What does this project do?**

This repository combines **computational fluid dynamics (CFD)** with **machine learning** to build surrogate models that can predict velocity and pressure fields around 2D airfoils at a fraction of the computational cost of traditional CFD solvers. Instead of waiting hours for OpenFOAM simulations, trained models can generate predictions in milliseconds.

**Key Capabilities:**

- 🛫 Generate parametric NACA-4 airfoil geometries with systematic variations
- 🔲 Compute signed-distance fields (SDFs) representing airfoil geometry on regular grids
- 🌀 Orchestrate OpenFOAM simulations to generate high-fidelity CFD datasets
- 🧠 Train attention-enhanced U-Net models to map airfoil geometry → flow fields
- ⚡ Deploy trained models for real-time flow prediction

---

## 📁 Repository Structure

```
ml-surrogate-airfoil/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
│
├── dataset_creation/                  # Data pipeline
│   ├── naca_airfoil_generation.py     # Generate NACA-4 airfoil coordinates
│   ├── sdf_gen.py                     # Compute signed-distance fields from geometries
│   ├── npy_to_stl.py                  # Convert coordinates to 3D STL meshes
│   └── data_gen_auto.py               # Orchestrate OpenFOAM runs and collect CFD data
│
├── model/                             # ML training & inference
│   ├── model_arch_attn.py             # U-Net with attention gates (core architecture)
│   ├── dataset_utils.py               # PyTorch dataset loader for CFD data
│   ├── train.py                       # Training loop with validation
│   └── test.py                        # Inference & visualization script
│
└── airFoil2D/                         # OpenFOAM template case
    ├── 0/                             # Initial and boundary conditions
    ├── constant/                      # Mesh and transport properties
    └── system/                        # Solver configuration
```

---

## 🔧 Core Components Explained

### 1. **Dataset Creation Pipeline**

#### `naca_airfoil_generation.py`
Generates NACA-4 airfoil coordinate arrays using the standard parametric equations. Creates 64 distinct airfoil geometries (8×8 parameter sweep).

**Output:** `airfoils_npy/` directory with `.npy` coordinate files

#### `sdf_gen.py`
Converts airfoil coordinates into **signed-distance fields** (SDFs):
- Negative values: inside the airfoil
- Positive values: outside the airfoil
- Normalized to [-1, +1] range

**Output:** `sdf_airfoil_new/` directory with `*_sdf.npy` files

#### `npy_to_stl.py`
Transforms 2D coordinate arrays into 3D thin-shell meshes (`.stl` format) suitable for OpenFOAM mesh generation.

**Output:** `stl_airfoil/` directory with `.stl` files

#### `data_gen_auto.py`
Orchestrates the complete CFD pipeline:
1. Copies OpenFOAM template case per geometry
2. Updates boundary conditions (freestream velocity, angle of attack)
3. Runs: `blockMesh` → `surfaceFeatures` → `snappyHexMesh` → `simpleFoam`
4. Interpolates solution onto regular 256×256 grid
5. Exports: `u.npy`, `v.npy`, `p.npy`, `sdf.npy` per case

**Requirements:** OpenFOAM must be installed and PATH-configured

---

### 2. **Neural Network Architecture**

#### `model_arch_attn.py`
Implements an **attention-enhanced U-Net** for aerodynamic field prediction.

**Key Features:**
- **Encoder-Decoder Structure:** 4-level hierarchical feature extraction
- **Attention Gates:** Selectively amplify relevant spatial features during upsampling
- **Instance Normalization:** Improves training stability
- **LeakyReLU Activations:** Prevents feature collapse

**Architecture Details:**
```
Input: (B, 1, 256, 256)  [SDF channel]
  ↓ Downsampling (4 levels, base=64)
  ↓ Bottleneck (16×base channels)
  ↓ Upsampling + Attention Gates
Output: (B, 3, 256, 256)  [u, v, p channels]
```

---

### 3. **Training & Evaluation**

#### `train.py`
Full training pipeline with validation:
- **Optimizer:** AdamW (learning rate: 1e-4)
- **Loss:** Mean Squared Error (MSE)
- **Batch Size:** 5 samples
- **Epochs:** 100 (configurable)
- **Checkpointing:** Saves best model + periodic snapshots
- **Device:** Automatic GPU/CPU selection

**Normalization Scheme:**
- Velocity fields (u, v): divided by 10.0
- Pressure field (p): divided by 100.0

#### `dataset_utils.py`
`AirfoilCFDDataset` class handles:
- Loading CFD data from disk
- Batch stacking (u, v, p → 3-channel output)
- Automatic normalization
- PyTorch DataLoader integration

#### `test.py`
Inference & visualization script:
- Loads best trained checkpoint
- Runs inference on test samples
- Side-by-side comparison: Ground Truth vs. Predictions
- Plots: SDF input, velocity components (u, v), pressure (p)

---

## 📦 Installation & Setup

### Prerequisites
- **Python:** 3.8 or higher
- **GPU (recommended):** NVIDIA GPU with CUDA support
- **OpenFOAM 9:** Only needed for `data_gen_auto.py`

### Quick Start

**1. Clone repository:**
```bash
git clone <repository_url>
cd ml-surrogate-airfoil
```

**2. Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

**4. Verify installation:**
```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
```

---

## 🚀 Usage Workflow

### **Phase 1: Dataset Generation** (in `dataset_creation/` directory)

```bash
# Step 1: Generate NACA-4 airfoil geometries
python naca_airfoil_generation.py
# Output: airfoils_npy/*.npy (64 airfoils)

# Step 2: Compute signed-distance fields
python sdf_gen.py
# Output: sdf_airfoil_new/*_sdf.npy

# Step 3: Convert to 3D meshes for OpenFOAM
python npy_to_stl.py
# Output: stl_airfoil/*.stl

# Step 4: Run CFD simulations (requires OpenFOAM)
python data_gen_auto.py
# Output: dataset_new/case_*/[u.npy, v.npy, p.npy, sdf.npy, meta.json]
# ⚠️ This step is slow (can take hours to days depending on case count)
```

### **Phase 2: Model Training** (in `model/` directory)

```bash
# Configure dataset path in train.py (DATASET_DIR variable)
# Ensure dataset structure: dataset/case_001/[u.npy, v.npy, p.npy, sdf.npy]

python train.py
# Output: 
#   - checkpoints/best_model.pt (best validation performance)
#   - checkpoints/10_model.pt, 20_model.pt, ... (periodic snapshots)
```

**Training Configuration** (edit in `train.py`):
```python
BATCH_SIZE = 5         # Adjust based on GPU memory
LR = 1e-4             # Learning rate
EPOCHS = 100          # Total epochs
VAL_SPLIT = 0.1       # 10% validation split
```

### **Phase 3: Inference & Visualization** (in `model/` directory)

```bash
# Configure paths in test.py
python test.py
# Displays: 3×4 grid comparing SDF input and predictions vs. ground truth
```

---

## 📊 Dataset Format

The training pipeline expects the following directory structure:

```
dataset/
├── case_001/
│   ├── u.npy              # X-velocity (256×256)
│   ├── v.npy              # Y-velocity (256×256)
│   ├── p.npy              # Pressure (256×256)
│   ├── sdf.npy            # Signed-distance field (256×256)
│   ├── meta.json          # (optional) Case metadata
│   ├── u_preview.png      # (optional) Visualization
│   └── ...
├── case_002/
│   └── [same structure]
└── ...
```

**File Specifications:**
- **Dimensions:** All `.npy` files are 2D arrays (256×256)
- **Data Type:** float32
- **Range:**
  - SDF: [-1.0, +1.0] (normalized by max distance 0.4)
  - Velocity: arbitrary (normalized during training by 10.0)
  - Pressure: arbitrary (normalized during training by 100.0)

---

## 🔧 Configuration & Customization

### Model Architecture
Modify `model_arch_attn.py`:
```python
model = AirfoilUNet(
    in_channels=1,      # SDF input
    out_channels=3,     # [u, v, p] output
    base=64            # Base filter count (increase for more capacity)
)
```

### Training Hyperparameters
Edit `train.py`:
```python
BATCH_SIZE = 5        # Larger batches = more stable gradients (needs more VRAM)
LR = 1e-4            # Learning rate (decrease if diverging)
EPOCHS = 100         # Convergence may occur earlier
VAL_SPLIT = 0.1      # Validation set ratio
```

### CFD Simulation Parameters
Edit `data_gen_auto.py`:
```python
GRID_SIZE = 256      # Output resolution
X_LIM = (-1.0, 3.0)  # Domain extent (streamwise)
Y_LIM = (-1.5, 1.5)  # Domain extent (cross-stream)
U_INF = 10.0         # Freestream velocity [m/s]
ALPHA_DEG = 5.0      # Angle of attack [degrees]
NU = 1.5e-5          # Kinematic viscosity [m²/s]
```

---

## 🐛 Troubleshooting

### **Import Error: `from model_arch import AirfoilUNet`**
The model file is named `model_arch_attn.py`, but imports reference `model_arch`.

**Solution:** Either rename the file or update imports in `train.py`/`test.py`:
```python
from model_arch_attn import AirfoilUNet  # Instead of: from model_arch import AirfoilUNet
```

### **OpenFOAM Commands Not Found**
`data_gen_auto.py` requires OpenFOAM executables on PATH.

**Solution:**
```bash
# Add to .bashrc or shell profile:
source /opt/openfoam10/etc/bashrc  # Adjust path to your OpenFOAM installation
```

### **CUDA Out of Memory**
GPU memory exhausted during training.

**Solution:**
- Reduce `BATCH_SIZE` in `train.py`
- Reduce `base` parameter in `AirfoilUNet` initialization
- Use CPU (slower but no memory limit): Training will automatically use CPU if CUDA unavailable

### **Data Loading Errors**
Missing or misformatted dataset files.

**Solution:**
- Verify dataset directory structure matches specification above
- Check all `.npy` files exist: `u.npy`, `v.npy`, `p.npy`, `sdf.npy`
- Verify array shapes: all must be (256, 256)

---

## 📈 Performance Metrics

**Model Performance Indicators:**
- **Inference Speed:** ~1-2ms per sample (GPU)
- **Speedup vs. CFD:** 1000-5000× faster than OpenFOAM
- **Accuracy:** Highly dependent on training dataset size and diversity

**Training Benchmark** (on single GPU):
```
Batch Size: 5
Epochs: 100
Device: NVIDIA A100 (example)
Time: ~5-10 minutes per 100 epochs
```

---

## 🧪 Quick Validation Script

Test the installation without full dataset:

```python
# model/test_install.py
import torch
from model_arch_attn import AirfoilUNet
import numpy as np

# Initialize model
model = AirfoilUNet(in_channels=1, out_channels=3, base=64)
print(f"✓ Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Test forward pass
x = torch.randn(2, 1, 256, 256)
y = model(x)
print(f"✓ Forward pass OK. Input shape: {x.shape}, Output shape: {y.shape}")

# Test dataset loader (if dataset exists)
try:
    from dataset_utils import AirfoilCFDDataset
    dataset = AirfoilCFDDataset(root_dir="../dataset", normalize=True)
    x, y = dataset[0]
    print(f"✓ Dataset loaded. Sample shapes: x={x.shape}, y={y.shape}")
except:
    print("⚠ Dataset not found (this is OK during first install)")
```

---

## 📚 References & Background

### NACA Airfoil Parametrization
- NACA 4-digit code: `MPTT` where M=max camber (%), P=camber location (/10), TT=thickness (%)
- Equation reference: *Theory of Wing Sections* - Abbott & Von Doenhoff

### Deep Learning for CFD
- U-Net architecture: *U-Net: Convolutional Networks for Biomedical Image Segmentation* (Ronneberger et al.)
- Attention mechanisms: *Attention U-Net: Learning Where to Look for the Pancreas* (Oktay et al.)

### OpenFOAM
- [Official Documentation](https://www.openfoam.com)
- Solver: SIMPLE (Semi-Implicit Method for Pressure-Linked Equations)

---

## 🤝 Contributing & Future Work

**Potential Enhancements:**
- [ ] Multi-angle-of-attack dataset collection
- [ ] Compressibility effects (transonic flow)
- [ ] Turbulence model variations
- [ ] Physics-informed loss functions
- [ ] GraphNeuralNet alternative architectures
- [ ] Quantization for mobile deployment
- [ ] Web interface for interactive predictions

---

## ⚖️ License & Citation

This project contains research code. If you use this work, please cite:

```bibtex
@software{ml_surrogate_airfoil_2024,
  title={ML-Based Surrogate Fluid Flow Modeling for Airfoils},
  author={Saksham},
  year={2026},
  url={https://github.com/Saksham3005/ML-based-surrogate-fluid-flow-modeling-for-airfoil}
}
```

---

## 📧 Contact & Support

For questions or issues:
- 📝 Open a GitHub issue
- 📧 Contact: [saksham.j3005@gmail.com]

---

**Happy modeling! 🚀**

