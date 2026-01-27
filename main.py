from processing import PreprocessingData
from model import CNN
from training import train_one_epoch, validate_one_epoch
import torch 
import torch.optim as optim
import torch.nn as nn


def main():
    datamodel = PreprocessingData(3404, transform=True)
    train_loader, val_loader, test_loader = datamodel.split_data(workers=4)

    model = CNN(num_classes=11)

    criteria = nn.CrossEntropyLoss()

    learning_rate = 0.001

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

    epochs = 110
    best_value_loss=float('inf')
    
    epochs_not_improving = 5
    waiting_index = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criteria, device=device)
        validation_loss, validation_acc = validate_one_epoch(model, val_loader, criteria, device=device)

        if validation_loss < best_value_loss:
            best_value_loss = validation_loss
            waiting_index = 0
            torch.save(model.state_dict(), "best.pth")
        else:
            waiting_index = waiting_index + 1
            if waiting_index == epochs_not_improving:
                learning_rate = learning_rate * 0.5

        print(train_loss, validation_loss)
        print(train_acc, validation_acc)

        #change learning rate
    


if __name__ == "__main__":
    main()
