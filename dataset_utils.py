import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset


class AirfoilCFDDataset(Dataset):
    def __init__(
        self,
        root_dir,
        normalize=True,
        dtype=torch.float32
    ):
        """
        root_dir: path to dataset/
        """
        self.root_dir = root_dir
        self.cases = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])

        self.normalize = normalize
        self.dtype = dtype

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        case_dir = os.path.join(self.root_dir, self.cases[idx])


        sdf = np.load(os.path.join(case_dir, "sdf.npy"))
        u   = np.load(os.path.join(case_dir, "u.npy"))
        v   = np.load(os.path.join(case_dir, "v.npy"))
        p   = np.load(os.path.join(case_dir, "p.npy"))


        x = sdf[None, :, :]
        y = np.stack([u, v, p], axis=0)

        # --------------------
        if self.normalize:

            y[0] /= 10.0  # u
            y[1] /= 10.0  # v


            y[2] /= 100.0

        x = torch.tensor(x, dtype=self.dtype)
        y = torch.tensor(y, dtype=self.dtype)

        return x, y
    
if __name__ == "__main__":

    from torch.utils.data import DataLoader

    dataset = AirfoilCFDDataset(
        root_dir="../dataset",
        normalize=True
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # Test
    x, y = next(iter(loader))
    print(x.shape)  # (B, 1, H, W)
    print(y.shape)  # (B, 3, H, W)

    import matplotlib.pyplot as plt

    x, y = dataset[0]




    import matplotlib.pyplot as plt

    x, y = dataset[0]

    extent = [-1.0, 3.0, -1.5, 1.5]  # MUST match SDF/CFD domain

    plt.figure(figsize=(12,4))

    plt.subplot(1,4,1)
    plt.title("SDF")
    plt.imshow(
        x[0],
        origin="lower",
        extent=extent,
        cmap="seismic"
    )
    plt.colorbar()

    plt.subplot(1,4,2)
    plt.title("u")
    plt.imshow(
        y[0],
        origin="lower",
        extent=extent,
        cmap="jet"
    )
    plt.colorbar()

    plt.subplot(1,4,3)
    plt.title("v")
    plt.imshow(
        y[1],
        origin="lower",
        extent=extent,
        cmap="jet"
    )
    plt.colorbar()

    plt.subplot(1,4,4)
    plt.title("p")
    plt.imshow(
        y[2],
        origin="lower",
        extent=extent,
        cmap="jet"
    )
    plt.colorbar()

    plt.tight_layout()
    plt.show()
