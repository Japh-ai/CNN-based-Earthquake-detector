# %% --- Imports ---

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay


from dataset import SeismicDataset
from model import SeismicCNN
from train import fit, evaluate

import pandas as pd

eq = pd.read_csv(EQ_CSV)
eq.head()

noise = pd.read_csv(NOISE_CSV)
noise.shape

# %%
# --- Configuration ---
EQ_HDF5 = "../data/chunk2.hdf5"  # HDF5 file containing earthquake waveforms
EQ_CSV = "../data/chunk2.csv"  # CSV file containing metadata for earthquake waveforms
NOISE_HDF5 = "../data/chunk1.hdf5"  # HDF5 file containing noise waveforms
NOISE_CSV = "../data/chunk1.csv"  # CSV file containing metadata for noise waveforms
MAX_SAMPLES = 8000
BATCH_SIZE = 32
NUM_EPOCHS = 15
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {DEVICE}")

# %%
# --- Load Dataset ---
dataset = SeismicDataset(
    EQ_HDF5, EQ_CSV, NOISE_HDF5, NOISE_CSV, max_samples=MAX_SAMPLES
)

# --- Train/Val/Test Split ---
train_size = int(0.70 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])
type(train_set)

# --- Create Dataloaders ---
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

# %%=
# --- Initialize Model ---

model = SeismicCNN()

print(model)

# %%
# Load the model (if you want to skip training and just evaluate)
# model = SeismicCNN()
# model.load_state_dict(torch.load(f"../models/seismic_cnn_model.pth"))
# model.eval()
# %%


# --- Train Model ---
print(
    f"Training on: {DEVICE}, with {len(train_loader.dataset)} training samples, {len(val_loader.dataset)} validation samples, and {len(test_loader.dataset)} test samples."
)

model, train_losses, val_losses = fit(
    model,
    train_loader,
    val_loader,
    num_epochs=NUM_EPOCHS,
    lr=LR,
    criterion=nn.BCELoss(),
    device=DEVICE,
)

# %%
# --- Plot training and validation loss curves ---
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Training loss", color="blue")
plt.plot(val_losses, label="Validation loss", color="red")
plt.title("Training and validation loss curves", fontsize=14)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()


# %%
# Store the model
torch.save(model.state_dict(), f"../models/seismic_cnn_model.pth")


# %% --- Evaluate on test set ---
test_loss, test_acc = evaluate(model, test_loader, nn.BCELoss(), device=DEVICE)
print(f"Test loss: {test_loss:.4f}  Test accuracy: {test_acc:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(train_losses, label="Training Loss", color="blue", linewidth=2)
plt.plot(val_losses, label="Validation Loss", color="red", linewidth=2)
plt.axhline(
    test_loss,
    label=f"Test Loss: {test_loss:.4f}",
    linestyle="--",
    color="green",
    linewidth=2,
)
plt.title("Training, Validation, and Test Loss Curves", fontsize=14, fontweight="bold")
plt.xlabel("Epochs", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.legend(fontsize=11, loc="upper right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("../reports/loss_curves.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# --- Classification Report and Confusion Matrix ---
y_true = []
y_pred = []
model.eval()
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        outputs = model(X_batch).squeeze()
        predictions = (outputs > 0.5).float()  # Threshold at 0.5
        y_true.extend(y_batch.cpu().numpy())
        y_pred.extend(predictions.cpu().numpy())
print("Classification Report:")
print(
    classification_report(y_true, y_pred, target_names=["Noise", "Earthquake"])
)  # target names are put in the ascending sorted order of the labels (0 → Noise, 1 → Earthquake)


fig, ax = plt.subplots(figsize=(6.5, 5.5))
ConfusionMatrixDisplay.from_predictions(
    y_true,
    y_pred,
    display_labels=["Noise", "Earthquake"],
    cmap="Blues",
    ax=ax,  # The ax parameter allows you to customize the plot (title, labels, etc.) after it's created.
    values_format="d",  # Show integer counts in the confusion matrix cells.
)
ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
ax.set_xlabel("Predicted label", fontsize=11)
ax.set_ylabel("True label", fontsize=11)
plt.tight_layout()
plt.savefig("../reports/confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.show()


# %%
fig, axes = plt.subplots(4, 4, figsize=(18, 12))
start = 35
for i, ax in enumerate(axes.flat):
    idx = start + i  # skip the first 35 examples.
    waveform = all_waveforms[idx].numpy()
    true_label = int(all_labels[idx])
    pred_prob = all_preds[idx]
    correct = true_label == int(pred_prob >= 0.5)

    for ch in range(3):
        ax.plot(waveform[ch], lw=0.4, alpha=0.85)

    for spine in ax.spines.values():
        spine.set_edgecolor("green" if correct else "red")
        spine.set_linewidth(2.5)

    true_str = "EQ" if true_label == 1 else "Noise"
    ax.set_title(
        f"True: {true_str}  |  Pred: {pred_prob:.2f}",
        fontsize=7.5,
        color="green" if correct else "red",
        fontweight="bold",
    )
    ax.set_xticks([])
    ax.set_yticks([])

fig.suptitle(
    "Test Set Predictions:  Green = correct  |  Red = wrong",
    fontsize=13,
    fontweight="bold",
)
plt.tight_layout()
plt.savefig("../reports/test_predictions.png", dpi=150, bbox_inches="tight")
plt.show()

# %%


# %%
