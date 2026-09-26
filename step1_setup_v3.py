"""
Step 1 of AdvLS (digital-only implementation) -- version 3.
Same as v2, but:
  - Skips any class folder that can't actually be read (corrupted extraction,
    permission issue, etc.) instead of crashing the whole run.
  - Saves the manifest to disk after every class folder, so if the script
    crashes or you Ctrl+C it, you keep everything processed so far.

Usage:
    python step1_setup_v3.py ^
        --image_dir "C:\\Users\\Brutikaa\\imagenet-mini\\val" ^
        --class_index "C:\\Users\\Brutikaa\\imagenet_class_index.json" ^
        --output correctly_classified.json
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
    with open(class_index_path, "r") as f:
        raw = json.load(f)
    return {synset: int(idx) for idx, (synset, _name) in raw.items()}


def save_manifest(manifest, output_path):
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Step 1 (v3): robust, crash-safe manifest builder")
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--class_index", required=True)
    parser.add_argument("--output", default="correctly_classified.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit_per_class", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(args.device)
    model, preprocess, categories = load_model(device)
    synset_to_index = load_synset_to_index(args.class_index)
    print(f"Loaded ResNet50 on {device}. {len(synset_to_index)} synset->index mappings loaded.\n")

    image_root = Path(args.image_dir)

    try:
        class_dirs = sorted(d for d in image_root.iterdir() if d.is_dir())
    except (FileNotFoundError, OSError) as e:
        print(f"Could not list {image_root}: {e}")
        return

    if not class_dirs:
        print(f"No subfolders found in {image_root}. Double-check the path.")
        return

    manifest = []
    kept, dropped, skipped_folders, broken_folders = 0, 0, 0, 0

    for class_dir in class_dirs:
        synset_code = class_dir.name

        if synset_code not in synset_to_index:
            skipped_folders += 1
            continue

        true_idx = synset_to_index[synset_code]
        true_name = categories[true_idx]

        try:
            image_paths = [p for p in class_dir.iterdir()
                            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        except (FileNotFoundError, OSError) as e:
            print(f"[BROKEN FOLDER] '{synset_code}' could not be read ({e}) -- skipping it and continuing.")
            broken_folders += 1
            continue

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

        # Save progress after every class folder -- a crash from here on
        # never loses more than the current folder's worth of work.
        save_manifest(manifest, args.output)

        print(f"[{synset_code}] '{true_name}': running totals -- "
              f"kept={kept}, dropped={dropped}, broken_folders={broken_folders}")

    save_manifest(manifest, args.output)
    print(f"\nDone. {kept} correctly classified, {dropped} misclassified, "
          f"{skipped_folders} folders had no class match, {broken_folders} folders were unreadable.")
    print(f"Manifest saved to {args.output}")


if __name__ == "__main__":
    main()
