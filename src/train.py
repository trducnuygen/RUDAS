import torch
import torch.nn as nn
import torch.optim as optim


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0

    criterion = nn.CrossEntropyLoss()

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


def train_model(model, train_loader, val_loader, device, epochs=10, lr=1e-3):
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        loss = train_one_epoch(model, train_loader, optimizer, device)
        acc = evaluate(model, val_loader, device)

        print(f"Epoch {epoch+1}: Loss={loss:.4f}, Acc={acc:.4f}")

    return model