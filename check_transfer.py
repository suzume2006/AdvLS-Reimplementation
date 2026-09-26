import json

with open('../results/transferability.json') as f:
    data = json.load(f)

successes = [r for r in data if r['source_attack_success']]
print(f"Source (ResNet50) ASR: {100*len(successes)/len(data):.1f}% ({len(successes)}/{len(data)})")
print()

for model_name in ['vgg19', 'mobilenet', 'densenet']:
    fooled = sum(1 for r in successes if r['transfer'][model_name])
    rate = 100 * fooled / len(successes)
    print(f"{model_name}: {fooled}/{len(successes)} fooled ({rate:.1f}% transfer rate)")