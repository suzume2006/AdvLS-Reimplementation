"""
Step 7 of AdvLS (digital-only implementation): attack transferability,
matching Table 3/4 in the paper.

For each image: run the GA against ResNet50 (the "white-box" target, same
as steps 4-6) to find a successful adversarial spot placement. Then, WITHOUT
any further optimization, feed that exact same adversarial image into several
other pretrained classifiers and check whether they are also fooled. This
measures "transfer" -- whether an attack crafted for one model generalizes
to others (a black-box attack scenario).

Usage:
    python step7_transferability.py --manifest correctly_classified.json
                                     --num_images 20 --color_mode blue
                                     --num_spots 35 --population 30 --generations 100
                                     --target_models vgg19 mobilenet densenet
                                     --output transferability.json

Note: all these torchvision ImageNet models share the same 1000-class index
ordering, so a true_label_idx from ResNet50's manifest is directly comparable
across models -- no relabeling needed.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision.models import (
    vgg19, VGG19_Weights,
    mobilenet_v2, MobileNet_V2_Weights,
    densenet121, DenseNet121_Weights,
    alexnet, AlexNet_Weights,
    resnet101, ResNet101_Weights,
)

from step4_ga import load_model, run_ga
from step5_batch_eval import load_manifest_sample


TARGET_MODEL_LOADERS = {
    "vgg19": (vgg19, VGG19_Weights.IMAGENET1K_V1),
    "mobilenet": (mobilenet_v2, MobileNet_V2_Weights.IMAGENET1K_V1),
    "densenet": (densenet121, DenseNet121_Weights.IMAGENET1K_V1),
    "alexnet": (alexnet, AlexNet_Weights.IMAGENET1K_V1),
    "resnet101": (resnet101, ResNet101_Weights.IMAGENET1K_V1),
}


def load_target_model(name, device):
    """Load one of the transfer-test target models (different architecture
    than the ResNet50 used for the actual attack optimization)."""
    model_fn, weights = TARGET_MODEL_LOADERS[name]
    model = model_fn(weights=weights)
    model.eval().to(device)
    preprocess = weights.transforms()
    return model, preprocess


def check_transfer(model, preprocess, device, adv_image, true_label_idx):
    """Returns True if this model is ALSO fooled by the given adversarial image
    (without any re-optimization -- just a single forward pass)."""
    x = preprocess(adv_image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        pred_idx = int(logits.argmax(dim=1))
    return pred_idx != true_label_idx


def main():
    parser = argparse.ArgumentParser(description="Step 7: attack transferability across models")
    parser.add_argument("--manifest", default="correctly_classified.json")
    parser.add_argument("--num_images", type=int, default=20)
    parser.add_argument("--color_mode", default="random", choices=["random", "red", "green", "blue"])
    parser.add_argument("--num_spots", type=int, default=35)
    parser.add_argument("--population", type=int, default=30)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--target_models", nargs="+", default=["vgg19", "mobilenet", "densenet"],
                         choices=list(TARGET_MODEL_LOADERS.keys()))
    parser.add_argument("--output", default="transferability.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(args.device)

    print("Loading source model (ResNet50, the one being attacked)...")
    source_model, source_preprocess, categories = load_model(device)

    print(f"Loading {len(args.target_models)} target models for transfer testing: {args.target_models}")
    target_models = {}
    for name in args.target_models:
        print(f"  loading {name}...")
        target_models[name] = load_target_model(name, device)

    sample = load_manifest_sample(args.manifest, args.num_images, seed=args.seed)
    print(f"\nLoaded {len(sample)} images (seed={args.seed})")
    print(f"Attack settings: {args.num_spots} spots, color_mode={args.color_mode}, "
          f"population={args.population}, generations={args.generations}\n")

    results = []
    output_file = Path(args.output)
    completed_paths = set()
    if output_file.exists():
        with open(output_file, "r") as f:
            results = json.load(f)
        completed_paths = {r["path"] for r in results}
        print(f"Resuming: found {len(results)} already-completed results\n")

    for i, entry in enumerate(sample):
        if entry["path"] in completed_paths:
            print(f"[{i+1}/{len(sample)}] skip (already done): {Path(entry['path']).name}")
            continue

        try:
            clean_image = Image.open(entry["path"]).convert("RGB")
        except Exception as e:
            print(f"[{i+1}/{len(sample)}] could not open {entry['path']}: {e}")
            continue

        start_time = time.time()
        success, best_group, num_queries, adv_image, pred_idx = run_ga(
            source_model, source_preprocess, device, clean_image, entry["true_label_idx"],
            num_spots=args.num_spots, population_size=args.population,
            max_generations=args.generations, color_mode=args.color_mode,
            verbose=False,
        )
        elapsed = time.time() - start_time

        result = {
            "path": entry["path"],
            "true_label_name": entry["true_label_name"],
            "source_attack_success": success,
            "num_queries": num_queries,
            "seconds": round(elapsed, 2),
            "transfer": {},
        }

        if success and adv_image is not None:
            transfer_summary = []
            for name, (target_model, target_preprocess) in target_models.items():
                transferred = check_transfer(
                    target_model, target_preprocess, device, adv_image, entry["true_label_idx"]
                )
                result["transfer"][name] = transferred
                transfer_summary.append(f"{name}={'FOOLED' if transferred else 'ok'}")
            print(f"[{i+1}/{len(sample)}] SOURCE SUCCESS | {num_queries:4d} queries | {elapsed:5.1f}s | "
                  f"{Path(entry['path']).name} | transfer: {', '.join(transfer_summary)}")
        else:
            print(f"[{i+1}/{len(sample)}] source attack FAILED | {num_queries:4d} queries | "
                  f"{elapsed:5.1f}s | {Path(entry['path']).name} (skipping transfer test)")

        results.append(result)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)

    # Summary: of images where the source attack succeeded, what % also
    # transferred (fooled) each target model?
    successful = [r for r in results if r["source_attack_success"]]
    print("\n" + "=" * 60)
    print(f"Source (ResNet50) ASR: {100.0 * len(successful) / len(results):.1f}% "
          f"({len(successful)}/{len(results)} images)")
    print("-" * 60)
    print(f"{'Target model':<15}{'Transfer rate':>20}")
    for name in args.target_models:
        if not successful:
            break
        transferred_count = sum(1 for r in successful if r["transfer"].get(name, False))
        rate = 100.0 * transferred_count / len(successful)
        print(f"{name:<15}{rate:>19.1f}%")
    print("=" * 60)
    print(f"\nFull per-image results saved to {args.output}")


if __name__ == "__main__":
    main()
