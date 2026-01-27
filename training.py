import torch
import torch.nn as nn
from tqdm import tqdm


def train_one_epoch(model, dataloader, optimizer, criteria, device):
    model.train()
    current_loss = 0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criteria(outputs, labels)
        loss.backward()
        optimizer.step()

        current_loss = current_loss + loss.item() * images.size(0)
        _, pred = torch.max(outputs, 1)
        correct = correct + (pred == labels).sum().item()
        total = total + labels.size(0)

    epoch_loss = current_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def validate_one_epoch(model, dataloader, criteria, device):
    model.eval()
    current_loss = 0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criteria(outputs, labels)
        current_loss = current_loss + loss.item() * images.size(0)
        _, pred = torch.max(outputs, 1)
        correct = correct + (pred == labels).sum().item()
        total = total + labels.size(0)
    epoch_loss = current_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def fit():
    pass


def evaluate():
    pass
