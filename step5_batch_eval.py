"""
Step 5 of AdvLS (digital-only implementation): batch evaluation.

Runs the genetic algorithm (step 4) across a sample of images from your
correctly_classified.json manifest (step 1), for each requested color mode,
and reports:
    - Attack Success Rate (ASR): % of images successfully misclassified
    - Average queries: mean number of model queries needed per image
      (only counted for successful attacks, matching the paper's Table 1)

This reproduces the structure of the paper's Table 1 (digital environment,
random/red/green/blue columns) on your own machine.

Usage:
    python step5_batch_eval.py --manifest correctly_classified.json
                                --num_images 50
                                --color_modes random red green blue
                                --num_spots 20 --population 20 --generations 50
                                --output batch_results.json

Notes:
    - This will take a while: num_images x len(color_modes) full GA runs.
      Start small (e.g. --num_images 20) to get a feel for the runtime
      before committing to a large run.
    - Progress is saved incrementally to --output after every single image,
      so if it crashes or you stop it, you keep everything done so far.
"""

import argparse
import json
import random
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights

from step4_ga import load_model, run_ga


def load_manifest_sample(manifest_path, num_images, seed=42):
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    random.Random(seed).shuffle(manifest)  # fixed seed so re-runs use the same sample
    return manifest[:num_images]


def run_batch(model, preprocess, device, sample, color_modes,
              num_spots, population, generations, output_path):

    # Load any existing partial results so re-running doesn't lose earlier progress
    # or redo work already completed (simple resume: skip entries already present).
    results = []
    completed_keys = set()
    output_file = Path(output_path)
    if output_file.exists():
        with open(output_file, "r") as f:
            results = json.load(f)
        completed_keys = {(r["path"], r["color_mode"]) for r in results}
        print(f"Resuming: found {len(results)} already-completed results in {output_path}\n")

    total_runs = len(sample) * len(color_modes)
    run_count = 0

    for color_mode in color_modes:
        for entry in sample:
            key = (entry["path"], color_mode)
            run_count += 1

            if key in completed_keys:
                print(f"[{run_count}/{total_runs}] skip (already done): "
                      f"{Path(entry['path']).name} [{color_mode}]")
                continue

            try:
                clean_image = Image.open(entry["path"]).convert("RGB")
            except Exception as e:
                print(f"[{run_count}/{total_runs}] could not open {entry['path']}: {e}")
                continue

            start_time = time.time()
            success, best_group, num_queries, adv_image, pred_idx = run_ga(
                model, preprocess, device, clean_image, entry["true_label_idx"],
                num_spots=num_spots, population_size=population,
                max_generations=generations, color_mode=color_mode,
                verbose=False,
            )
            elapsed = time.time() - start_time

            result = {
                "path": entry["path"],
                "true_label_name": entry["true_label_name"],
                "color_mode": color_mode,
                "success": success,
                "num_queries": num_queries,
                "seconds": round(elapsed, 2),
            }
            results.append(result)

            status = "SUCCESS" if success else "FAILED "
            print(f"[{run_count}/{total_runs}] {status} | {color_mode:6s} | "
                  f"{num_queries:4d} queries | {elapsed:5.1f}s | {Path(entry['path']).name}")

            # Save after every single image -- never lose more than one image's work.
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)

    return results


def summarize(results, color_modes):
    print("\n" + "=" * 60)
    print(f"{'Color':<10}{'ASR':>10}{'Avg Queries (successes only)':>35}")
    print("-" * 60)
    for color_mode in color_modes:
        subset = [r for r in results if r["color_mode"] == color_mode]
        if not subset:
            continue
        successes = [r for r in subset if r["success"]]
        asr = 100.0 * len(successes) / len(subset)
        avg_queries = sum(r["num_queries"] for r in successes) / len(successes) if successes else float("nan")
        print(f"{color_mode:<10}{asr:>9.1f}%{avg_queries:>35.1f}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Step 5: batch ASR/query evaluation across many images")
    parser.add_argument("--manifest", default="correctly_classified.json")
    parser.add_argument("--num_images", type=int, default=20)
    parser.add_argument("--color_modes", nargs="+", default=["random"],
                         choices=["random", "red", "green", "blue"])
    parser.add_argument("--num_spots", type=int, default=20)
    parser.add_argument("--population", type=int, default=20)
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--output", default="batch_results.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(args.device)
    model, preprocess, categories = load_model(device)

    sample = load_manifest_sample(args.manifest, args.num_images, seed=args.seed)
    print(f"Loaded {len(sample)} images from {args.manifest} "
          f"(seeded sample, same {args.num_images} images every run with seed={args.seed})")
    print(f"Testing color modes: {args.color_modes}")
    print(f"Total GA runs: {len(sample)} images x {len(args.color_modes)} colors "
          f"= {len(sample) * len(args.color_modes)}\n")

    results = run_batch(
        model, preprocess, device, sample, args.color_modes,
        args.num_spots, args.population, args.generations, args.output,
    )

    summarize(results, args.color_modes)
    print(f"\nFull per-image results saved to {args.output}")


if __name__ == "__main__":
    main()
