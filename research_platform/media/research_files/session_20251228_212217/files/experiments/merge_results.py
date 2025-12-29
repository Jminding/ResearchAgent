#!/usr/bin/env python3
"""Merge all batch results into a single extended results file."""
import json
import os
import glob

RESULTS_DIR = "/Users/jminding/Desktop/Code/Research Agent/research_platform/media/research_files/session_20251228_212217/files/results"

def merge_results():
    all_results = []

    # Load all batch JSON files
    batch_files = glob.glob(os.path.join(RESULTS_DIR, "batch*.json"))

    for batch_file in sorted(batch_files):
        print(f"Loading {os.path.basename(batch_file)}...")
        try:
            with open(batch_file, 'r') as f:
                data = json.load(f)
                if "results" in data:
                    all_results.extend(data["results"])
                    print(f"  Added {len(data['results'])} results")
        except Exception as e:
            print(f"  Error: {e}")

    # Check for other result files
    other_files = [
        "extended_training_results.json",
        "extended_results_part1.json"
    ]

    for fname in other_files:
        fpath = os.path.join(RESULTS_DIR, fname)
        if os.path.exists(fpath):
            print(f"Loading {fname}...")
            try:
                with open(fpath, 'r') as f:
                    data = json.load(f)
                    if "results" in data:
                        all_results.extend(data["results"])
                        print(f"  Added {len(data['results'])} results")
            except Exception as e:
                print(f"  Error: {e}")

    # Remove duplicates based on config_name
    seen = set()
    unique_results = []
    for r in all_results:
        key = r.get("config_name", str(r))
        if key not in seen:
            seen.add(key)
            unique_results.append(r)

    print(f"\nTotal unique results: {len(unique_results)}")

    # Save merged results
    output = {
        "project_name": "QEC_RL_Scaling_Revision",
        "results": unique_results
    }

    json_path = os.path.join(RESULTS_DIR, "extended_results_table.json")
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Saved to {json_path}")

    # Also save as CSV
    if unique_results:
        csv_path = os.path.join(RESULTS_DIR, "extended_results_table.csv")

        # Collect all keys
        all_params = set()
        all_metrics = set()
        for r in unique_results:
            all_params.update(r.get("parameters", {}).keys())
            all_metrics.update(r.get("metrics", {}).keys())

        headers = ["config_name", "ablation", "error"]
        headers.extend(sorted(all_params))
        headers.extend(sorted(all_metrics))

        lines = [",".join(headers)]
        for r in unique_results:
            row = [
                r.get("config_name", ""),
                str(r.get("ablation", "")),
                str(r.get("error", ""))
            ]
            for p in sorted(all_params):
                row.append(str(r.get("parameters", {}).get(p, "")))
            for m in sorted(all_metrics):
                row.append(str(r.get("metrics", {}).get(m, "")))
            lines.append(",".join(row))

        with open(csv_path, 'w') as f:
            f.write("\n".join(lines))
        print(f"Saved to {csv_path}")

    return unique_results


if __name__ == "__main__":
    results = merge_results()

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    from collections import Counter
    config_types = Counter()
    for r in results:
        name = r.get("config_name", "")
        if "extended_" in name:
            config_types["Extended Training"] += 1
        elif "comparison_" in name:
            config_types["RL vs MWPM"] += 1
        elif "reward_" in name:
            config_types["Reward Ablation"] += 1
        elif "gnn_" in name:
            config_types["GNN Ablation"] += 1
        elif "zeroshot_" in name:
            config_types["Zero-Shot"] += 1
        elif "mwpm_validation" in name:
            config_types["MWPM Validation"] += 1
        elif "learning_curve" in name:
            config_types["Learning Curves"] += 1
        else:
            config_types["Other"] += 1

    for exp_type, count in sorted(config_types.items()):
        print(f"  {exp_type}: {count}")

    error_count = sum(1 for r in results if r.get("error"))
    print(f"\n  Total errors: {error_count}")
