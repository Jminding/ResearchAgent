#!/usr/bin/env python3
import json

with open('/Users/jminding/Desktop/Code/Research Agent/research_platform/media/research_files/session_20251228_212217/files/results/extended_results_table.json', 'r') as f:
    data = json.load(f)

# Find all d=15 extended training experiments
d15_configs = []
for i, r in enumerate(data['results']):
    if r['parameters'].get('code_distance') == 15:
        d15_configs.append({
            'index': i,
            'config_name': r['config_name'],
            'training_episodes': r['parameters'].get('training_episodes'),
            'seed': r['parameters'].get('seed'),
            'logical_error_rate': r['metrics'].get('logical_error_rate') or r['metrics'].get('logical_error_rate_rl')
        })

print(f"Found {len(d15_configs)} d=15 configurations\n")

# Group by experiment type
from collections import defaultdict
by_type = defaultdict(list)
for cfg in d15_configs:
    if 'comparison' in cfg['config_name']:
        by_type['comparison'].append(cfg)
    elif 'd15' in cfg['config_name']:
        by_type['extended_training'].append(cfg)
    elif 'reward' in cfg['config_name']:
        by_type['reward'].append(cfg)
    elif 'gnn' in cfg['config_name']:
        by_type['gnn'].append(cfg)
    else:
        by_type['other'].append(cfg)

for exp_type, configs in by_type.items():
    print(f"\n{exp_type}: {len(configs)} configs")
    if len(configs) <= 10:
        for cfg in configs[:5]:
            print(f"  {cfg}")
    else:
        print(f"  First 3:")
        for cfg in configs[:3]:
            print(f"    {cfg}")

# Show first few entries of extended_training
if 'extended_training' in by_type:
    print(f"\n\nExtended training d=15 breakdown:")
    by_episodes = defaultdict(list)
    for cfg in by_type['extended_training']:
        by_episodes[cfg['training_episodes']].append(cfg)

    for ep in sorted(by_episodes.keys()):
        print(f"  {ep} episodes: {len(by_episodes[ep])} seeds")
        print(f"    LERs: {[c['logical_error_rate'] for c in by_episodes[ep][:5]]}")
