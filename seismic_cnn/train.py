#####
import torch
import torch.nn as nn


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()  # dropout ON, batchnorm uses batch stats
    total_loss, correct = 0.0, 0
    # "loader" yields batches — each iteration gives you X (a batch of waveforms) and y (their labels).

    for (
        X,
        y,
    ) in loader:
        X, y = X.to(device), y.to(device).unsqueeze(1)

        optimizer.zero_grad()  # Clears gradients from the previous batch.
        # PyTorch accumulates gradients by default, so without this the gradients from batch 1 would add to batch 2, batch 3

        preds = model(
            X
        )  # Forward pass — runs the input through all layers and gets predictions. Shape: (batch, 1), values between 0 and 1.

        loss = criterion(
            preds, y
        )  # Compares predictions to true labels and computes a single number — how wrong the model was on this batch.

        loss.backward()  # Backpropagation — computes the gradient of the loss with respect to every weight in the network.
        # After this, every weight has a .grad value.

        optimizer.step()  # Uses those gradients to update the weights.

        total_loss += loss.item() * len(X)
        correct += (preds.round() == y).sum().item()

    n = len(loader.dataset)  # number of samples in the whole dataset.
    return (
        total_loss / n,
        correct / n,
    )  # returns average loss and accuracy for the epoch


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device).unsqueeze(1)
            preds = model(
                X
            )  # No stored gradients needed for evaluation, so we wrap in torch.no_grad().

            loss = criterion(preds, y)
            total_loss += loss.item() * len(X)
            correct += (preds.round() == y).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


def fit(
    model,
    train_loader,
    val_loader,
    num_epochs=15,
    lr=1e-3,
    criterion=nn.BCELoss(),
    device="cuda",
):
    # Binary Cross-Entropy Loss, standard for binary classification.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Adam optimizer.
    model.to(device)
    print(f"Training on: {next(model.parameters()).device}")

    train_losses, val_losses = [], []
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch:02d}/{num_epochs} | "
            f"Train loss: {train_loss:.4f}  acc: {train_acc:.4f} | "
            f"Val loss: {val_loss:.4f}  acc: {val_acc:.4f}"
        )
        train_losses.append(train_loss)
        val_losses.append(val_loss)
    return model, train_losses, val_losses
