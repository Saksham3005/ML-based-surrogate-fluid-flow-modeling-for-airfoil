import numpy as np
import trimesh
import os

airfoil_dir = os.listdir("./airfoils_npy")

for c in airfoil_dir:
    if c.endswith("sdf.npy") == False:
        airfoil_path = os.path.join("./airfoils_npy", c)
        airfoil = np.load(airfoil_path)
        # Close geometry
        if not np.allclose(airfoil[0], airfoil[-1]):
            airfoil = np.vstack([airfoil, airfoil[0]])

        # Extrude 2D airfoil slightly in z
        z_thickness = 0.01
        vertices = []
        faces = []

        for i in range(len(airfoil) - 1):
            p1 = airfoil[i]
            p2 = airfoil[i+1]

            v0 = [p1[0], p1[1], -z_thickness]
            v1 = [p2[0], p2[1], -z_thickness]
            v2 = [p2[0], p2[1],  z_thickness]
            v3 = [p1[0], p1[1],  z_thickness]

            idx = len(vertices)
            vertices += [v0, v1, v2, v3]
            faces += [
                [idx, idx+1, idx+2],
                [idx, idx+2, idx+3]
            ]

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.export("./stl_airfoil/" + c.replace(".npy", ".stl"))
