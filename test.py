import torch
import numpy as np
import matplotlib.pyplot as plt
from dataset_utils import AirfoilCFDDataset
from model_arch import AirfoilUNet

DATASET_DIR = "../dataset"
CHECKPOINT = "./checkpoints/best_model.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

X_LIM = (-1.0, 3.0)
Y_LIM = (-1.5, 1.5)
EXTENT = [*X_LIM, *Y_LIM]


dataset = AirfoilCFDDataset(
    root_dir=DATASET_DIR,
    normalize=True
)

idx = 255  # change this for testing on other entries
x, y_gt = dataset[idx]


model = AirfoilUNet(in_channels=1, out_channels=3, base=64)
ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
model.load_state_dict(ckpt["model"])
model.to(DEVICE)
model.eval()


with torch.no_grad():
    pred = model(x.unsqueeze(0).to(DEVICE)).cpu()[0]


sdf = x[0].numpy()

u_gt, v_gt, p_gt = y_gt.numpy()
u_pr, v_pr, p_pr = pred.numpy()


def plot_field(ax, field, title, cmap="jet"):
    im = ax.imshow(
        field,
        origin="lower",
        extent=EXTENT,
        cmap=cmap
    )
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar(im, ax=ax, fraction=0.046)


fig, axes = plt.subplots(3, 4, figsize=(18, 10))

plot_field(axes[0,0], sdf, "SDF", cmap="seismic")
axes[0,1].axis("off")
axes[0,2].axis("off")
axes[0,3].axis("off")

plot_field(axes[1,0], u_gt, "u (GT)")
plot_field(axes[1,1], v_gt, "v (GT)")
plot_field(axes[1,2], p_gt, "p (GT)")
axes[1,3].axis("off")

plot_field(axes[2,0], u_pr, "u (Pred)")
plot_field(axes[2,1], v_pr, "v (Pred)")
plot_field(axes[2,2], p_pr, "p (Pred)")
axes[2,3].axis("off")

plt.tight_layout()
plt.show()
