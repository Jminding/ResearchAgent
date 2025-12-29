#!/usr/bin/env python3
import json

with open('/Users/jminding/Desktop/Code/Research Agent/research_platform/media/research_files/session_20251228_212217/files/results/extended_results_table.json', 'r') as f:
    data = json.load(f)

results = data['results']
print(f"Total results: {len(results)}\n")

# Sample different types
samples_by_prefix = {}
for r in results:
    prefix = r['config_name'].split('_')[0]
    if prefix not in samples_by_prefix:
        samples_by_prefix[prefix] = r

print("Sample by prefix:")
for prefix, r in sorted(samples_by_prefix.items()):
    print(f"\n{prefix}:")
    print(f"  Config: {r['config_name']}")
    print(f"  Params: {r['parameters']}")
    print(f"  Metrics keys: {list(r['metrics'].keys())}")

# Count by prefix
from collections import Counter
prefix_counts = Counter(r['config_name'].split('_')[0] for r in results)
print(f"\n\nCounts by prefix:")
for prefix, count in prefix_counts.most_common():
    print(f"  {prefix}: {count}")

# Find d15 with different episode counts
print(f"\n\nSearching for d=15 experiments...")
d15_samples = {}
for r in results:
    if r['parameters'].get('code_distance') == 15:
        episodes = r['parameters'].get('training_episodes')
        key = (r['config_name'].split('_')[0], episodes)
        if key not in d15_samples:
            d15_samples[key] = r

print(f"Found {len(d15_samples)} unique (prefix, episodes) combinations at d=15:")
for (prefix, episodes), r in sorted(d15_samples.items()):
    print(f"  {prefix} @ {episodes} ep: {r['config_name']}")
