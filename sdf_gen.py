import numpy as np
from shapely.geometry import Polygon, Point
from shapely import contains_xy

def close_airfoil(airfoil_xy):
    if not np.allclose(airfoil_xy[0], airfoil_xy[-1]):
        airfoil_xy = np.vstack([airfoil_xy, airfoil_xy[0]])
    return airfoil_xy



def generate_sdf(
    airfoil_xy,
    grid_size=256,
    xlim=(-1.0, 3.0),
    ylim=(-1.5, 1.5),
    d_max=0.4
):
    x = np.linspace(xlim[0], xlim[1], grid_size)
    y = np.linspace(ylim[0], ylim[1], grid_size)
    X, Y = np.meshgrid(x, y)

    airfoil_xy = close_airfoil(airfoil_xy)
    airfoil = Polygon(airfoil_xy)

    if not airfoil.is_valid:
        raise ValueError("Invalid airfoil polygon")

    inside = contains_xy(airfoil, X, Y)

    dist = np.zeros_like(X, dtype=np.float32)
    for i in range(grid_size):
        for j in range(grid_size):
            dist[i, j] = airfoil.exterior.distance(
                Point(X[i, j], Y[i, j])
            )

    sdf = dist
    sdf[inside] *= -1.0

    sdf = np.clip(sdf, -d_max, d_max)
    sdf = sdf / d_max

    return X, Y, sdf


out_dir = "./airfoils_npy"
import os, sys

list_of_airfoils = os.listdir(out_dir)
for fname in list_of_airfoils:
    if fname.endswith("sdf.npy") == False:
        airfoil_xy = np.load(os.path.join(out_dir, fname))
        # airfoil_xy = close_airfoil(airfoil_xy)
        X, Y, sdf = generate_sdf(airfoil_xy)

        sdf_fname = fname.replace(".npy", "_sdf.npy")
        np.save(os.path.join("./sdf_airfoil_new", sdf_fname), sdf)
        print(f"Saved SDF to {sdf_fname}")

