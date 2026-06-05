import torch
import os


def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(model, path, device="cpu"):
    model.load_state_dict(torch.load(path, map_location=device))
    return model