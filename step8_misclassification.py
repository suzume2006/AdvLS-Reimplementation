"""
Step 8 of AdvLS (digital-only implementation): misclassification semantic
analysis, matching Figures 5 and 9 in the paper.

The paper's interesting finding: when laser spots successfully fool the
classifier, the WRONG label it lands on isn't random -- certain categories
(Bubble, Envelope, Petri dish, Hair Slide, etc.) come up far more often than
others, because their visual features (small, round, colorful blobs) happen
to resemble laser spots.

This script runs the GA across a sample of images, and for every SUCCESSFUL
attack, records what wrong label the model landed on. At the end, it tallies
how often each wrong label appeared, so you can see your own version of the
paper's "most common misclassification targets" list.

Usage:
    python step8_misclassification.py --manifest correctly_classified.json
                                       --num_images 40 --color_mode green
                                       --num_spots 35 --population 30 --generations 100
                                       --output misclassification.json
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import torch
from PIL import Image

from step4_ga import load_model, run_ga
from step5_batch_eval import load_manifest_sample


def main():
    parser = argparse.ArgumentParser(description="Step 8: misclassification semantic analysis")
    parser.add_argument("--manifest", default="correctly_classified.json")
    parser.add_argument("--num_images", type=int, default=40)
    parser.add_argument("--color_mode", default="random", choices=["random", "red", "green", "blue"])
    parser.add_argument("--num_spots", type=int, default=35)
    parser.add_argument("--population", type=int, default=30)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--output", default="misclassification.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(args.device)
    model, preprocess, categories = load_model(device)

    sample = load_manifest_sample(args.manifest, args.num_images, seed=args.seed)
    print(f"Loaded {len(sample)} images (seed={args.seed})")
    print(f"Attack settings: {args.num_spots} spots, color_mode={args.color_mode}, "
          f"population={args.population}, generations={args.generations}\n")

    results = []
    completed_paths = set()
    output_file = Path(args.output)
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
            model, preprocess, device, clean_image, entry["true_label_idx"],
            num_spots=args.num_spots, population_size=args.population,
            max_generations=args.generations, color_mode=args.color_mode,
            verbose=False,
        )
        elapsed = time.time() - start_time

        misclassified_as = categories[pred_idx] if success and pred_idx is not None else None

        result = {
            "path": entry["path"],
            "true_label_name": entry["true_label_name"],
            "success": success,
            "misclassified_as": misclassified_as,
            "num_queries": num_queries,
            "seconds": round(elapsed, 2),
        }
        results.append(result)

        if success:
            print(f"[{i+1}/{len(sample)}] SUCCESS | {entry['true_label_name']} -> "
                  f"'{misclassified_as}' | {num_queries} queries | {elapsed:.1f}s")
        else:
            print(f"[{i+1}/{len(sample)}] FAILED  | {entry['true_label_name']} | "
                  f"{num_queries} queries | {elapsed:.1f}s")

        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)

    # Tally the misclassification targets, like the paper's Figure 5.
    successes = [r for r in results if r["success"]]
    label_counts = Counter(r["misclassified_as"] for r in successes)

    print("\n" + "=" * 60)
    print(f"Successful attacks: {len(successes)}/{len(results)} "
          f"({100.0 * len(successes) / len(results):.1f}% ASR)")
    print("-" * 60)
    print("Most common misclassification targets:")
    for label, count in label_counts.most_common(15):
        pct = 100.0 * count / len(successes)
        bar = "#" * count
        print(f"  {label:<25} {count:>3} ({pct:4.1f}%) {bar}")
    print("=" * 60)
    print(f"\nFull results saved to {args.output}")
    print("\nCompare this list to the paper's Figure 5/9 findings (e.g. 'Bubble', "
          "'Envelope', 'Petri dish', 'Hair slide') -- categories with small, round, "
          "shiny, or colorful visual features tend to attract laser-spot misclassifications.")


if __name__ == "__main__":
    main()
