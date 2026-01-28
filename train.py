import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset_utils import AirfoilCFDDataset
from model_arch import AirfoilUNet


DATASET_DIR = "../dataset"
CHECKPOINT_DIR = "./checkpoints"

BATCH_SIZE = 5
LR = 1e-4
EPOCHS = 100
VAL_SPLIT = 0.1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


dataset = AirfoilCFDDataset(
    root_dir=DATASET_DIR,
    normalize=True
)
n_val = int(len(dataset) * VAL_SPLIT)
n_train = len(dataset) - n_val

train_ds, val_ds = random_split(dataset, [n_train, n_val])

train_loader = DataLoader(
    train_ds,   
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_ds,
    batch_size=1,
    shuffle=False
)



print(f"Dataset: {len(dataset)} samples")
print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")


model = AirfoilUNet(
    in_channels=1,
    out_channels=3,
    base=64
).to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)


def loss_fn(pred, target):
    return F.mse_loss(pred, target)


best_val = float("inf")

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
    for x, y in pbar:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        pred = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        pbar.set_postfix(train_loss=loss.item())

    train_loss /= len(train_loader)

    # -------------------------
    # VALIDATION
    # -------------------------
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            pred = model(x)
            val_loss += loss_fn(pred, y).item()

    val_loss /= max(1, len(val_loader))

    print(
        f"Epoch {epoch:04d} | "
        f"Train: {train_loss:.6e} | "
        f"Val: {val_loss:.6e}"
    )
    if epoch % 10 == 0:
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"{epoch}_model.pt")
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_loss": val_loss,
            },
            ckpt_path
        )
        print(f"Saved checkpoint @ epoch {epoch}")

    if val_loss < best_val:
        best_val = val_loss
        ckpt_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_loss": val_loss,
            },
            ckpt_path
        )
        print(f"Saved best model @ epoch {epoch}")
