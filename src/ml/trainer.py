import torch
import torch.nn as nn
import time
import os

def train(model, train_loader, val_loader, logger, path_to_load,
          epochs=50, lr=1e-3, patience=5):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    epochs_no_improve = 0

    start_time = time.time()

    for epoch in range(epochs):

        model.train()
        train_loss = 0

        for X, Y in train_loader:
            pred = model(X)
            loss = loss_fn(pred, Y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0

        with torch.no_grad():
            for X, Y in val_loader:
                pred = model(X)
                loss = loss_fn(pred, Y)
                val_loss += loss.item()

        logger.info(
            f"Epoch {epoch} | "
            f"Train Loss {train_loss:.6f} | "
            f"Val Loss {val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            os.makedirs(path_to_load, exist_ok=True)
            torch.save(model.state_dict(), path_to_load + "model_data.pt")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            logger.info("Early stopping triggered")
            break

    total_time = time.time() - start_time
    logger.info(f"Training finished in {total_time:.2f} seconds")