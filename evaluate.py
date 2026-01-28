import matplotlib.pyplot as plt
import numpy as np
import torch

from model import CNN
from processing import PreprocessingData


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    datamodel = PreprocessingData(3404, transform=True)
    train_loader, val_loader, test_loader = datamodel.split_data()

    loader = val_loader
    # loader = test_loader

    num_classes = len(datamodel.dataset.classes)
    class_names = list(datamodel.dataset.classes)

    model = CNN(num_classes=num_classes)
    state = torch.load("best.pth", map_location=device)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)

    accuracy = (preds == labels).sum() / labels.shape[0]

    # Confusion matrix
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(labels, preds):
        cm[int(t), int(p)] += 1

    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion matrix (rows=true, columns=predicted):")
    print(cm)

    print("Per-class accuracy:")
    for i, name in enumerate(class_names):
        total = cm[i].sum()
        correct = int(cm[i, i])
        acc = correct / total if total > 0 else 0.0
        print(f"  {name}: {acc:.4f} ({correct}/{total})")

    # normalize rows to percentages for accurate color mapping
    cm_pct = np.zeros_like(cm, dtype=float)
    for i in range(cm.shape[0]):
        row_total = cm[i].sum()
        if row_total > 0:
            cm_pct[i] = cm[i] / row_total * 100.0

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        cm_pct, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=100
    )
    ax.figure.colorbar(im, ax=ax)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(class_names, rotation=90, fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)

    # overlay percentages per true class on the cells
    try:
        for i in range(cm.shape[0]):
            row_total = cm[i].sum()
            for j in range(cm.shape[1]):
                val = cm[i, j]
                pct = (val / row_total * 100.0) if row_total > 0 else 0.0
                color = "white" if pct > 50.0 else "black"
                ax.text(
                    j,
                    i,
                    f"{pct:.1f}%",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=7,
                )
    except Exception:
        pass

    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close(fig)
    print("Saved confusion_matrix.png")


if __name__ == "__main__":
    main()
