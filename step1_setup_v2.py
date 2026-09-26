"""
Step 1 of AdvLS (digital-only implementation) -- version 2, matched to the
imagenet-mini dataset layout (folders named by synset code, e.g. n01440764).

Usage:
    pip install torch torchvision pillow
    python step1_setup_v2.py ^
        --image_dir "C:\\Users\\Brutikaa\\imagenet-mini\\val" ^
        --class_index "C:\\Users\\Brutikaa\\imagenet_class_index.json" ^
        --output correctly_classified.json

What it does:
    1. Loads imagenet_class_index.json, which maps
       {"0": ["n01440764", "tench"], "1": ["n01443537", "goldfish"], ...}
       into a lookup: synset_code -> class_index (the number ResNet50 outputs).
    2. Walks every synset-coded subfolder under --image_dir (e.g. n01440764/).
    3. Runs ResNet50 (pretrained on ImageNet) on every image inside.
    4. Keeps only the images where the model's top prediction matches the
       folder's true class index -- these are your "correctly classified"
       images, exactly like the paper's 1000-image digital test set.
    5. Saves the surviving images (path, true label, confidence) to a JSON
       manifest that later steps (laser spot generation + GA attack) will
       read from.
"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights


def load_model(device):
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    model.eval().to(device)
    preprocess = weights.transforms()
    categories = weights.meta["categories"]
    return model, preprocess, categories


def load_synset_to_index(class_index_path):
    """imagenet_class_index.json looks like: {"0": ["n01440764","tench"], ...}
    Build a dict: "n01440764" -> 0"""
    with open(class_index_path, "r") as f:
        raw = json.load(f)
    return {synset: int(idx) for idx, (synset, _name) in raw.items()}


def main():
    parser = argparse.ArgumentParser(description="Step 1 (v2): build correctly-classified manifest from imagenet-mini")
    parser.add_argument("--image_dir", required=True,
                         help=r'Path to the val (or train) folder, e.g. "C:\Users\Brutikaa\imagenet-mini\val"')
    parser.add_argument("--class_index", required=True,
                         help=r'Path to imagenet_class_index.json')
    parser.add_argument("--output", default="correctly_classified.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit_per_class", type=int, default=0,
                         help="Optional cap on images checked per class folder (0 = no limit). "
                              "Useful for a quick first run before scaling to all ~1000 images.")
    args = parser.parse_args()

    device = torch.device(args.device)
    model, preprocess, categories = load_model(device)
    synset_to_index = load_synset_to_index(args.class_index)
    print(f"Loaded ResNet50 on {device}. {len(synset_to_index)} synset->index mappings loaded.\n")

    image_root = Path(args.image_dir)
    class_dirs = sorted(d for d in image_root.iterdir() if d.is_dir())

    if not class_dirs:
        print(f"No subfolders found in {image_root}. Double-check the path.")
        return

    manifest = []
    kept, dropped, skipped_folders = 0, 0, 0

    for class_dir in class_dirs:
        synset_code = class_dir.name  # e.g. "n01440764"

        if synset_code not in synset_to_index:
            print(f"[skip folder] '{synset_code}' not found in class index JSON.")
            skipped_folders += 1
            continue

        true_idx = synset_to_index[synset_code]
        true_name = categories[true_idx]

        image_paths = [p for p in class_dir.iterdir()
                       if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        if args.limit_per_class > 0:
            image_paths = image_paths[:args.limit_per_class]

        for img_path in image_paths:
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"[skip file] Could not open {img_path.name}: {e}")
                continue

            x = preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(x)
                probs = torch.softmax(logits, dim=1)
                pred_idx = int(probs.argmax(dim=1))
                confidence = float(probs[0, pred_idx])

            if pred_idx == true_idx:
                manifest.append({
                    "path": str(img_path),
                    "true_label_idx": true_idx,
                    "true_label_name": true_name,
                    "confidence": round(confidence, 4),
                })
                kept += 1
            else:
                dropped += 1

        print(f"[{synset_code}] '{true_name}': checked {len(image_paths)} images so far -- "
              f"running totals: kept={kept}, dropped={dropped}")

    with open(args.output, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone. {kept} correctly classified, {dropped} misclassified, "
          f"{skipped_folders} folders skipped (no class match).")
    print(f"Manifest saved to {args.output} -- this feeds into the laser spot + GA steps next.")


if __name__ == "__main__":
    main()
