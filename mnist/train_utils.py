import torch
from torch.utils.data import DataLoader


def train_loop(dataloader: DataLoader, model, loss_fn, optimizer, device: torch.device):
    model.to(device)
    model.train()
    total_loss = 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def test_loop(dataloader: DataLoader, model, loss_fn, device: torch.device):
    model.to(device)
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    accuracy = correct / size
    return test_loss, accuracy


def train_one_epoch(
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    model,
    loss_fn,
    optimizer,
    device,
):
    train_loss = train_loop(train_dataloader, model, loss_fn, optimizer, device)
    test_loss, accuracy = test_loop(test_dataloader, model, loss_fn, device)
    return train_loss, test_loss, accuracy


def train_many_epochs(
    epochs: int,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    model,
    loss_fn,
    optimizer,
    device,
    writer=None,
):
    for epoch in range(epochs):
        train_loss, test_loss, accuracy = train_one_epoch(
            train_dataloader, test_dataloader, model, loss_fn, optimizer, device
        )
        print(
            f"Epoch {epoch + 1}/{epochs}  "
            f"train_loss: {train_loss:.4f}  "
            f"test_loss: {test_loss:.4f}  "
            f"accuracy: {accuracy:.4f}"
        )
        if writer:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/test", test_loss, epoch)
            writer.add_scalar("Accuracy", accuracy, epoch)

    return model
