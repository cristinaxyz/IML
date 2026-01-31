import torch
import torch.nn as nn
import torch.optim as optim

from model import CNN, ConvolutionBlockGroupNorm
from processing import PreprocessingData
from training import train_one_epoch, validate_one_epoch
# https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
# Early stopping utility gotten form stackoverflow and slightly modified for our purposes 
class EarlyStopper:
    def __init__(self, patience: int = 10, min_delta: float = 0.0, max_overfit: float = 0.65):
        self.patience = patience
        self.min_delta = min_delta
        self.max_overfit = max_overfit  #tolerable overfitting gap gotten from the training logs 
        self.best_loss = float("inf")
        self.counter = 0
    
    def __call__(self, val_loss: float, train_loss: float) -> bool:
        # Check if overfitting exceeds threshold
        overfit_gap = val_loss - train_loss
        if overfit_gap > self.max_overfit:
            print(f"Overfitting detected: gap={overfit_gap:.4f} > {self.max_overfit}")
            return True
        
        # Check if validation loss improved
        if val_loss < (self.best_loss - self.min_delta):
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
            return False

def main():
    datamodel = PreprocessingData(3404)
    train_loader, val_loader, test_loader = datamodel.split_data()

    model = CNN(num_classes=11)
    criteria = nn.CrossEntropyLoss()

    learning_rate = 0.001 

    optimizer = optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4
    )

    epochs = 110
    best_value_loss = float("inf")

    epochs_not_improving = 5
    waiting_index = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)

    early_stop = EarlyStopper(patience=10, min_delta=0.001, max_overfit=0.65)
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criteria, device=device
        )
        validation_loss, validation_acc = validate_one_epoch(
            model, val_loader, criteria, device=device
        )
        
        if early_stop(validation_loss, train_loss):
            print(f"Early stopping triggered at epoch {epoch}")
            break

        if validation_loss < best_value_loss:
            best_value_loss = validation_loss
            waiting_index = 0
            torch.save(model.state_dict(), "best.pth")
        else:
            waiting_index = waiting_index + 1
            if waiting_index == epochs_not_improving:
                learning_rate = learning_rate * 0.5
                waiting_index = 0

        print(
            f"Learning Rate: {learning_rate}, waiting index: {waiting_index}/{epochs_not_improving}"
        )
        print(f"Train Loss: {train_loss}, Validation Loss: {validation_loss}")
        print(
            f"Train Accuracy: {train_acc}, Validation Accuracy: {validation_acc}"
        )
        # change learning rate

    torch.save(model.state_dict(), "final.pth")


if __name__ == "__main__":
    main()
