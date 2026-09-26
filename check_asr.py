import json
from collections import defaultdict

with open('../results/comparison_v2.json') as f:
    data = json.load(f)

groups = defaultdict(list)
for r in data:
    groups[r['color_mode']].append(r)

for color, rs in groups.items():
    successes = [r for r in rs if r['success']]
    asr = 100 * len(successes) / len(rs)
    avg_queries = sum(r['num_queries'] for r in successes) / len(successes) if successes else 0
    print(f"{color}: ASR={asr:.1f}% ({len(successes)}/{len(rs)}), avg_queries={avg_queries:.1f}")