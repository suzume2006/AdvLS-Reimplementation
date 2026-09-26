"""
Step 6 of AdvLS (digital-only implementation): number-of-spots ablation,
matching Figure 8 in the paper (ASR vs. number of laser spots, 5 to 100).

Usage:
    python step6_spots_ablation.py --manifest correctly_classified.json
                                    --num_images 20
                                    --spot_counts 5 15 25 35 50
                                    --color_mode random
                                    --population 30 --generations 100
                                    --output spots_ablation.json

Note: the paper sweeps 5-100 in steps of 5 (20 data points) across ~1000
images per point -- that's extremely expensive to fully reproduce. Pick a
handful of representative spot counts (e.g. 5, 15, 25, 35, 50) on a smaller
image sample first, matching the pattern rather than the exact scale.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image

from step4_ga import load_model, run_ga
from step5_batch_eval import load_manifest_sample


def run_ablation(model, preprocess, device, sample, spot_counts, color_mode,
                  population, generations, output_path):

    results = []
    completed_keys = set()
    output_file = Path(output_path)
    if output_file.exists():
        with open(output_file, "r") as f:
            results = json.load(f)
        completed_keys = {(r["path"], r["num_spots"]) for r in results}
        print(f"Resuming: found {len(results)} already-completed results in {output_path}\n")

    total_runs = len(sample) * len(spot_counts)
    run_count = 0

    for num_spots in spot_counts:
        for entry in sample:
            key = (entry["path"], num_spots)
            run_count += 1

            if key in completed_keys:
                print(f"[{run_count}/{total_runs}] skip (already done): "
                      f"{Path(entry['path']).name} [{num_spots} spots]")
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
                "num_spots": num_spots,
                "success": success,
                "num_queries": num_queries,
                "seconds": round(elapsed, 2),
            }
            results.append(result)

            status = "SUCCESS" if success else "FAILED "
            print(f"[{run_count}/{total_runs}] {status} | {num_spots:3d} spots | "
                  f"{num_queries:4d} queries | {elapsed:5.1f}s | {Path(entry['path']).name}")

            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)

    return results


def summarize(results, spot_counts):
    print("\n" + "=" * 60)
    print(f"{'Spots':<10}{'ASR':>10}{'Avg Queries (successes only)':>35}")
    print("-" * 60)
    for num_spots in spot_counts:
        subset = [r for r in results if r["num_spots"] == num_spots]
        if not subset:
            continue
        successes = [r for r in subset if r["success"]]
        asr = 100.0 * len(successes) / len(subset)
        avg_queries = sum(r["num_queries"] for r in successes) / len(successes) if successes else float("nan")
        print(f"{num_spots:<10}{asr:>9.1f}%{avg_queries:>35.1f}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Step 6: number-of-spots ablation")
    parser.add_argument("--manifest", default="correctly_classified.json")
    parser.add_argument("--num_images", type=int, default=20)
    parser.add_argument("--spot_counts", nargs="+", type=int, default=[5, 15, 25, 35, 50])
    parser.add_argument("--color_mode", default="random", choices=["random", "red", "green", "blue"])
    parser.add_argument("--population", type=int, default=30)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--output", default="spots_ablation.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(args.device)
    model, preprocess, categories = load_model(device)

    sample = load_manifest_sample(args.manifest, args.num_images, seed=args.seed)
    print(f"Loaded {len(sample)} images (seed={args.seed}, same sample as step 5 for consistency)")
    print(f"Testing spot counts: {args.spot_counts}, color_mode={args.color_mode}")
    print(f"Total GA runs: {len(sample)} images x {len(args.spot_counts)} spot counts "
          f"= {len(sample) * len(args.spot_counts)}\n")

    results = run_ablation(
        model, preprocess, device, sample, args.spot_counts, args.color_mode,
        args.population, args.generations, args.output,
    )

    summarize(results, args.spot_counts)
    print(f"\nFull per-image results saved to {args.output}")


if __name__ == "__main__":
    main()
