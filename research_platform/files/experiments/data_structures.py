"""
Data structures for experiment management and results tracking.
"""
import json
import csv
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from datetime import datetime


@dataclass
class ExperimentResult:
    """Single experiment result."""
    config_name: str
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    ablation: Optional[str] = None
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ResultsTable:
    """Collection of experiment results."""
    project_name: str
    results: List[ExperimentResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_result(self, result: ExperimentResult):
        self.results.append(result)

    def to_dict(self) -> Dict:
        return {
            "project_name": self.project_name,
            "metadata": self.metadata,
            "results": [r.to_dict() for r in self.results]
        }

    def to_json(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def to_csv(self, filepath: str):
        if not self.results:
            return

        # Flatten results for CSV
        rows = []
        for r in self.results:
            row = {
                'config_name': r.config_name,
                'ablation': r.ablation or '',
                'error': r.error or '',
                'timestamp': r.timestamp
            }
            # Add parameters
            for k, v in r.parameters.items():
                row[f'param_{k}'] = v
            # Add metrics
            for k, v in r.metrics.items():
                row[f'metric_{k}'] = v
            rows.append(row)

        # Get all keys
        all_keys = set()
        for row in rows:
            all_keys.update(row.keys())
        all_keys = sorted(all_keys)

        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            writer.writerows(rows)

    @classmethod
    def from_json(cls, filepath: str) -> 'ResultsTable':
        with open(filepath, 'r') as f:
            data = json.load(f)

        table = cls(project_name=data.get('project_name', 'Unknown'))
        table.metadata = data.get('metadata', {})

        for r in data.get('results', []):
            result = ExperimentResult(
                config_name=r['config_name'],
                parameters=r['parameters'],
                metrics=r['metrics'],
                ablation=r.get('ablation'),
                error=r.get('error'),
                timestamp=r.get('timestamp', '')
            )
            table.add_result(result)

        return table


@dataclass
class ExperimentPlan:
    """Experiment plan specification."""
    project_name: str
    experiments: List[Dict]
    robustness_checklist: Dict

    @classmethod
    def from_json(cls, filepath: str) -> 'ExperimentPlan':
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(
            project_name=data.get('project_name', 'Unknown'),
            experiments=data.get('experiments', []),
            robustness_checklist=data.get('robustness_checklist', {})
        )
