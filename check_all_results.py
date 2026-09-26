import json
from pathlib import Path

results_dir = Path("../results")

print("=" * 60)
print("1. COLOR / ASR COMPARISON (comparison_v2.json)")
print("=" * 60)
with open(results_dir / "comparison_v2.json") as f:
    data = json.load(f)
colors = {}
for r in data:
    colors.setdefault(r['color_mode'], []).append(r)
for color, rs in colors.items():
    successes = [r for r in rs if r['success']]
    asr = 100 * len(successes) / len(rs)
    avg_q = sum(r['num_queries'] for r in successes) / len(successes) if successes else 0
    print(f"  {color:8s} ASR={asr:5.1f}% ({len(successes)}/{len(rs)})  avg_queries={avg_q:.1f}")

print()
print("=" * 60)
print("2. NUMBER-OF-SPOTS ABLATION (spots_ablation.json)")
print("=" * 60)
with open(results_dir / "spots_ablation.json") as f:
    data = json.load(f)
spot_counts = {}
for r in data:
    spot_counts.setdefault(r['num_spots'], []).append(r)
for count, rs in sorted(spot_counts.items()):
    successes = [r for r in rs if r['success']]
    asr = 100 * len(successes) / len(rs)
    avg_q = sum(r['num_queries'] for r in successes) / len(successes) if successes else 0
    print(f"  {count:3d} spots  ASR={asr:5.1f}% ({len(successes)}/{len(rs)})  avg_queries={avg_q:.1f}")

print()
print("=" * 60)
print("3. TRANSFERABILITY (transferability.json)")
print("=" * 60)
with open(results_dir / "transferability.json") as f:
    data = json.load(f)
successes = [r for r in data if r['source_attack_success']]
print(f"  Source (ResNet50) ASR: {100*len(successes)/len(data):.1f}% ({len(successes)}/{len(data)})")
for model_name in ['vgg19', 'mobilenet', 'densenet']:
    fooled = sum(1 for r in successes if r['transfer'][model_name])
    rate = 100 * fooled / len(successes) if successes else 0
    print(f"  {model_name:10s} {fooled}/{len(successes)} fooled  ({rate:.1f}% transfer rate)")

print()
print("=" * 60)
print("4. MISCLASSIFICATION ANALYSIS (misclassification.json)")
print("=" * 60)
with open(results_dir / "misclassification.json") as f:
    data = json.load(f)
successes = [r for r in data if r['success']]
print(f"  ASR: {100*len(successes)/len(data):.1f}% ({len(successes)}/{len(data)})")
from collections import Counter
label_counts = Counter(r['misclassified_as'] for r in successes)
print("  Most common wrong labels:")
for label, count in label_counts.most_common(10):
    pct = 100 * count / len(successes)
    print(f"    {label:20s} {count:2d} ({pct:.1f}%)")

print()
print("=" * 60)
print("5. BASELINE ACCURACY (correctly_classified.json)")
print("=" * 60)
with open(results_dir / "correctly_classified.json") as f:
    data = json.load(f)
print(f"  Correctly classified images kept: {len(data)}")