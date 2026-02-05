import os
import shutil
import subprocess
import numpy as np
import pyvista as pv
import json
from pyvista import OpenFOAMReader
import matplotlib.pyplot as plt


STL_DIR = "./stl_airfoil"
SDF_DIR = "./sdf_airfoil"
TEMPLATE_CASE = "./airFoil2D"
OUT_DIR = "./dataset_new"
NPY_DIR = "./airfoils_npy"

GRID_SIZE = 256
X_LIM = (-1.0, 3.0)
Y_LIM = (-1.5, 1.5)

U_INF = 10.0
ALPHA_DEG = 5.0
NU = 1.5e-5



x = np.linspace(*X_LIM, GRID_SIZE)
y = np.linspace(*Y_LIM, GRID_SIZE)
X, Y = np.meshgrid(x, y)
QUERY_POINTS = np.c_[X.ravel(), Y.ravel(), np.zeros_like(X).ravel()]



def save_field_plot(field, title, fname, vmin=None, vmax=None, cmap="viridis"):
    plt.figure(figsize=(5, 3))
    plt.imshow(
        field,
        origin="lower",
        extent=[X_LIM[0], X_LIM[1], Y_LIM[0], Y_LIM[1]],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax
    )
    plt.colorbar()
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close()


def run(cmd, cwd):
    subprocess.run(cmd, cwd=cwd, shell=True, check=True)

def write_U_file(case_dir):
    alpha = np.deg2rad(ALPHA_DEG)
    Ux = U_INF * np.cos(alpha)
    Uy = U_INF * np.sin(alpha)

    U_txt = f"""
FoamFile
{{
    format ascii;
    class volVectorField;
    object U;
}}

dimensions [0 1 -1 0 0 0 0];
internalField uniform ({Ux} {Uy} 0);

boundaryField
{{
    inlet {{ type fixedValue; value uniform ({Ux} {Uy} 0); }}
    outlet {{ type zeroGradient; }}
    top {{ type slip; }}
    bottom {{ type slip; }}
    airfoil {{ type noSlip; }}
    front {{ type empty; }}
    back {{ type empty; }}
}}
"""
    with open(os.path.join(case_dir, "0/U"), "w") as f:
        f.write(U_txt)

def log_failure(name, reason):
    with open("failed_cases.txt", "a") as f:
        f.write(f"{name}: {reason}\n")

# MAIN LOOP

os.makedirs(OUT_DIR, exist_ok=True)

for stl_file in sorted(os.listdir(STL_DIR)):
    if not stl_file.endswith(".stl"):
        continue

    name = stl_file.replace(".stl", "")
    print(f"\n🚀 Processing {name}")

    case_dir = os.path.join(OUT_DIR, name)

    # Safety: skip if already processed
    if os.path.exists(os.path.join(case_dir, "u.npy")):
        print(f"⚠️  Skipping {name} (already exists)")
        continue

    try:

        shutil.copytree(TEMPLATE_CASE, case_dir)

        foam_marker = os.path.join(case_dir, f"{name}.OpenFOAM")
        open(foam_marker, "w").close()

        shutil.copy(
            os.path.join(STL_DIR, stl_file),
            os.path.join(case_dir, "constant/triSurface/airfoil.stl")
        )

        sdf = np.load(os.path.join(SDF_DIR, f"{name}_sdf.npy"))
        write_U_file(case_dir)

        # running CFD

        run("blockMesh", case_dir)
        run("surfaceFeatures", case_dir)
        run("snappyHexMesh -overwrite", case_dir)
        run("simpleFoam", case_dir)

        reader = OpenFOAMReader(foam_marker)
        reader.enable_all_patch_arrays()
        reader.enable_all_cell_arrays()

        reader.set_active_time_value(reader.time_values[-1])
        foam = reader.read()
        grid = foam["internalMesh"]

        points_pd = pv.PolyData(QUERY_POINTS)
        dx = (X_LIM[1] - X_LIM[0]) / GRID_SIZE
        dy = (Y_LIM[1] - Y_LIM[0]) / GRID_SIZE
        radius = 2.5 * max(dx, dy)

        interp = points_pd.interpolate(
            grid,
            radius=radius,
            sharpness=2.0
        )


        U = interp["U"]
        p = interp["p"]

        if U.shape[0] != GRID_SIZE * GRID_SIZE:
            raise RuntimeError("Interpolation size mismatch")

        u = U[:, 0].reshape(GRID_SIZE, GRID_SIZE)
        v = U[:, 1].reshape(GRID_SIZE, GRID_SIZE)
        p = p.reshape(GRID_SIZE, GRID_SIZE)

        u = np.nan_to_num(u)
        v = np.nan_to_num(v)
        p = np.nan_to_num(p)


        from shapely.geometry import Polygon, Point

        airfoil_xy = np.load(os.path.join(NPY_DIR, f"{name}.npy"))  # or wherever
        airfoil_poly = Polygon(airfoil_xy)

        solid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)

        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                pt = Point(X[i, j], Y[i, j])
                if airfoil_poly.contains(pt):
                    solid[i, j] = True


        u[solid] = 0.0
        v[solid] = 0.0
        p[solid] = 0.0


        if np.isnan(u).any() or np.isnan(v).any() or np.isnan(p).any():
            raise RuntimeError("NaNs detected")

        if np.max(np.abs(u)) > 5 * U_INF:
            raise RuntimeError("Velocity blow-up detected")


        np.save(os.path.join(case_dir, "u.npy"), u)
        np.save(os.path.join(case_dir, "v.npy"), v)
        np.save(os.path.join(case_dir, "p.npy"), p)
        np.save(os.path.join(case_dir, "sdf.npy"), sdf)

        save_field_plot(
            u,
            title="u-velocity",
            fname=os.path.join(case_dir, "u.png"),
            vmin=0.0,
            vmax=1.5 * U_INF,
            cmap="jet"
        )

        save_field_plot(
            v,
            title="v-velocity",
            fname=os.path.join(case_dir, "v.png"),
            vmin=-0.5 * U_INF,
            vmax=0.5 * U_INF,
            cmap="seismic"
        )

        save_field_plot(
            p,
            title="pressure",
            fname=os.path.join(case_dir, "p.png"),
            cmap="viridis"
        )


        with open(os.path.join(case_dir, "meta.json"), "w") as f:
            json.dump({
                "U_inf": U_INF,
                "alpha_deg": ALPHA_DEG,
                "nu": NU
            }, f, indent=2)

        for folder in ["constant", "system", "processor0"]:
            path = os.path.join(case_dir, folder)
            if os.path.exists(path):
                shutil.rmtree(path)

        print(f"Done: {name}")

    except Exception as e:
        print(f"Failed: {name} → {e}")
        log_failure(name, str(e))
        if os.path.exists(case_dir):
            shutil.rmtree(case_dir)
        continue
