from processing import PreprocessingData
from model import CNN
from training import train_one_epoch, validate_one_epoch
import torch.optim as optim
import torch.nn as nn


def main():
    datamodel = PreprocessingData(3404, transform=True)
    train_loader, val_loader, test_loader = datamodel.split_data()

    model = CNN(num_classes=10)

    criteria = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

    epochs = 110
    best_value_loss=float('inf')
    
    epochs_not_improving = 5
    waiting_index = 0

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criteria)
        validation_loss, validation_acc = validate_one_epoch(model, val_loader, optimizer, criteria)

        #change learning rate
    


if __name__ == "__main__":
    main()
