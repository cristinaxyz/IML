import re
import sys
from typing import Dict, List

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def parse_training_log_text(text: str) -> Dict[str, List[float]]:
    """Parse training log text and return metric lists per epoch."""

    epoch_starts = [
        m.start() for m in re.finditer(r"^Epoch\s+\d+/\d+", text, re.M)
    ]
    blocks = []
    for i, pos in enumerate(epoch_starts):
        end = epoch_starts[i + 1] if i + 1 < len(epoch_starts) else len(text)
        blocks.append(text[pos:end])

    re_lr = re.compile(r"Learning Rate:\s*([0-9.eE+\-]+)")
    re_train_loss = re.compile(r"Train Loss:\s*([0-9.eE+\-]+)")
    re_val_loss = re.compile(r"Validation Loss:\s*([0-9.eE+\-]+)")
    re_train_acc = re.compile(r"Train Accuracy:\s*([0-9.eE+\-]+)")
    re_val_acc = re.compile(r"Validation Accuracy:\s*([0-9.eE+\-]+)")

    learning_rates: List[float] = []
    train_losses: List[float] = []
    val_losses: List[float] = []
    train_accs: List[float] = []
    val_accs: List[float] = []

    def _first_float(pattern, block):
        m = pattern.search(block)
        return float(m.group(1))

    for blk in blocks:
        try:
            learning_rates.append(_first_float(re_lr, blk))
            train_losses.append(_first_float(re_train_loss, blk))
            val_losses.append(_first_float(re_val_loss, blk))
            train_accs.append(_first_float(re_train_acc, blk))
            val_accs.append(_first_float(re_val_acc, blk))
        except AttributeError:
            pass

    return {
        "learning_rates": learning_rates,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accs": train_accs,
        "val_accs": val_accs,
    }


def parse_training_log_file(path: str) -> Dict[str, List[float]]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return parse_training_log_text(text)


if __name__ == "__main__":
    metrics = parse_training_log_file(sys.argv[1])

    print(
        f"Best validation accuracy: {max(metrics['val_accs']):.4f} in epoch {metrics['val_accs'].index(max(metrics['val_accs'])) + 1} ",
    )
    print(
        f"Best validation loss: {min(metrics['val_losses']):.4f} in epoch {metrics['val_losses'].index(min(metrics['val_losses'])) + 1} ",
    )

    # Plot training and validation loss
    plt.figure(figsize=(12, 5))
    plt.plot(metrics["train_losses"], label="Train Loss")
    plt.plot(metrics["val_losses"], label="Validation Loss")
    plt.plot(metrics["val_accs"], label="Validation Accuracy")
    lrs = metrics.get("learning_rates", [])
    lr_changed = False
    if lrs:
        change_epochs = []
        last = lrs[0]
        for i, lr in enumerate(lrs, start=1):
            if lr != last:
                change_epochs.append(i)
                last = lr
        # draw a vertical dashed line at each change epoch
        for ce in change_epochs:
            plt.axvline(x=ce, color="gray", linestyle=":", linewidth=1)
        if change_epochs:
            lr_changed = True

    plt.xlabel("Epoch")
    plt.title("Training and Validation Loss (LR_init = 0.00025, Batch Norm)")
    handles, labels = plt.gca().get_legend_handles_labels()
    if lr_changed:
        proxy = Line2D([0], [0], color="gray", linestyle=":", linewidth=1)
        handles.append(proxy)
        labels.append("LR change")
    plt.legend(handles, labels)
    plt.savefig("figures/out.png")
