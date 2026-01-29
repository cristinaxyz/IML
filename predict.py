# THIS IS PURE CHATGPT just for fun
# We probably don't need to include this

import argparse
import pathlib
from typing import List

import torch
import torchvision.transforms as T
from PIL import Image

from model import CNN
from processing import IMAGE_SIZE, PreprocessingData, mean, std


def load_image(path: pathlib.Path, transform) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0)


def gather_image_paths(path: pathlib.Path) -> List[pathlib.Path]:
    if path.is_file():
        return [path]
    imgs = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"):
        imgs.extend(sorted(path.rglob(ext)))
    return imgs


def main():
    parser = argparse.ArgumentParser(
        description="Run images through trained model"
    )
    parser.add_argument(
        "path", type=str, help="Image file or directory of images"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best.pth",
        help="Checkpoint file (.pth)",
    )
    parser.add_argument(
        "--topk", type=int, default=3, help="Show top-k predictions per image"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda or cpu (default autodetect)",
    )
    args = parser.parse_args()

    device = torch.device(
        args.device
        if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # get class names from dataset (keeps consistent ordering)
    datamodel = PreprocessingData(3404)
    class_names = list(datamodel.dataset.classes)
    num_classes = len(class_names)

    model = CNN(num_classes=num_classes)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()

    # transform matching evaluation (no augmentation)
    transform = T.Compose(
        [
            T.Resize(IMAGE_SIZE),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )

    path = pathlib.Path(args.path)
    if not path.exists():
        print(f"Path not found: {path}")
        return

    image_paths = gather_image_paths(path)
    if len(image_paths) == 0:
        print("No images found at the given path.")
        return

    with torch.no_grad():
        for img_path in image_paths:
            img_tensor = load_image(img_path, transform).to(device)
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            topk = torch.topk(probs, k=min(args.topk, num_classes))
            indices = topk.indices.cpu().tolist()
            values = topk.values.cpu().tolist()

            print(f"{img_path}:")
            for idx, score in zip(indices, values):
                print(f"  {class_names[idx]}: {score:.4f}")


if __name__ == "__main__":
    main()
