import sys

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
    state_dir = sys.argv[1] if len(sys.argv) > 1 else "best.pth"
    state = torch.load(state_dir, map_location=device)
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

    # Confusion matrix
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(labels, preds):
        cm[int(t), int(p)] += 1
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

    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    print("Metrics per class (accuracy precision recall f1):")
    for i, name in enumerate(class_names):
        tp = int(cm[i, i])
        predicted = cm[:, i].sum()
        actual = cm[i].sum()
        precision = tp / predicted if predicted > 0 else 0.0
        recall = tp / actual if actual > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        accuracies.append(tp / actual if actual > 0 else 0.0)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        print(
            f"{name}: {accuracies[i]:.4f} {precisions[i]:.4f} {recalls[i]:.4f} {f1s[i]:.4f}"
        )

    mean_acc = np.mean(accuracies)
    mean_p = np.mean(precisions)
    mean_r = np.mean(recalls)
    mean_f1 = np.mean(f1s)
    print(f"Average: {mean_acc:.4f} {mean_p:.4f} {mean_r:.4f} {mean_f1:.4f}")

    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
