#!/usr/bin/env python3
import json
import sys

# Load data
with open('/Users/jminding/Desktop/Code/Research Agent/research_platform/media/research_files/session_20251228_212217/files/results/extended_results_table.json', 'r') as f:
    data = json.load(f)

results = data['results']
print(f"Total experiments: {len(results)}")

# Get unique config name patterns
config_patterns = {}
for r in results:
    config_name = r['config_name']
    # Extract experiment type prefix
    parts = config_name.split('_')
    if len(parts) >= 2:
        exp_type = parts[0] + '_' + parts[1] if parts[0] != 'd' else parts[0] + parts[1]
    else:
        exp_type = parts[0]

    if exp_type not in config_patterns:
        config_patterns[exp_type] = []
    config_patterns[exp_type].append(config_name)

print(f"\nExperiment types found:")
for exp_type, configs in sorted(config_patterns.items()):
    print(f"  {exp_type}: {len(configs)} configs")

# Show first example of each type
print(f"\nFirst example of each type:")
for exp_type, configs in sorted(config_patterns.items()):
    idx = next(i for i, r in enumerate(results) if r['config_name'] == configs[0])
    r = results[idx]
    print(f"\n{exp_type}:")
    print(f"  Config: {r['config_name']}")
    print(f"  Params: {r['parameters']}")
    print(f"  Metrics: {list(r['metrics'].keys())}")

# Save config list for reference
with open('/Users/jminding/Desktop/Code/Research Agent/research_platform/media/research_files/session_20251228_212217/config_summary.json', 'w') as f:
    json.dump({
        'total_experiments': len(results),
        'experiment_types': {k: len(v) for k, v in config_patterns.items()},
        'sample_configs': {k: v[0] for k, v in config_patterns.items()}
    }, f, indent=2)

print("\n\nSaved config_summary.json")
