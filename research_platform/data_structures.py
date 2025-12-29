"""
Data structures for experiment tracking and results storage.
"""
import json
import csv
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from datetime import datetime


@dataclass
class ExperimentResult:
    """Single experiment result with configuration and metrics."""
    config_name: str
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    ablation: Optional[str] = None
    error: Optional[str] = None
    seed: Optional[int] = None
    training_time_seconds: Optional[float] = None
    inference_time_ms: Optional[float] = None
    memory_usage_gb: Optional[float] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        return asdict(self)

    def to_flat_dict(self) -> Dict:
        """Flatten for CSV export."""
        flat = {
            'config_name': self.config_name,
            'ablation': self.ablation,
            'error': self.error,
            'seed': self.seed,
            'training_time_seconds': self.training_time_seconds,
            'inference_time_ms': self.inference_time_ms,
            'memory_usage_gb': self.memory_usage_gb,
            'timestamp': self.timestamp,
        }
        # Flatten parameters
        for k, v in self.parameters.items():
            flat[f'param_{k}'] = v
        # Flatten metrics
        for k, v in self.metrics.items():
            flat[f'metric_{k}'] = v
        return flat


@dataclass
class ResultsTable:
    """Collection of experiment results with I/O methods."""
    project_name: str
    results: List[ExperimentResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_result(self, result: ExperimentResult):
        self.results.append(result)

    def to_json(self, filepath: str):
        """Save results to JSON file."""
        data = {
            'project_name': self.project_name,
            'metadata': self.metadata,
            'total_results': len(self.results),
            'results': [r.to_dict() for r in self.results]
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def to_csv(self, filepath: str):
        """Save results to CSV file."""
        if not self.results:
            return

        # Get all unique keys
        all_keys = set()
        for r in self.results:
            all_keys.update(r.to_flat_dict().keys())

        fieldnames = sorted(list(all_keys))

        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self.results:
                writer.writerow(r.to_flat_dict())

    @classmethod
    def from_json(cls, filepath: str) -> 'ResultsTable':
        """Load results from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        table = cls(project_name=data['project_name'])
        table.metadata = data.get('metadata', {})

        for r_data in data.get('results', []):
            result = ExperimentResult(
                config_name=r_data['config_name'],
                parameters=r_data['parameters'],
                metrics=r_data['metrics'],
                ablation=r_data.get('ablation'),
                error=r_data.get('error'),
                seed=r_data.get('seed'),
                training_time_seconds=r_data.get('training_time_seconds'),
                inference_time_ms=r_data.get('inference_time_ms'),
                memory_usage_gb=r_data.get('memory_usage_gb'),
                timestamp=r_data.get('timestamp', '')
            )
            table.add_result(result)

        return table


@dataclass
class ExperimentPlan:
    """Experiment plan loader."""
    project_name: str
    experiments: List[Dict]
    robustness_checklist: Dict
    data_guidelines: Dict
    hypotheses: List[Dict]

    @classmethod
    def from_json(cls, filepath: str) -> 'ExperimentPlan':
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(
            project_name=data['project_name'],
            experiments=data['experiments'],
            robustness_checklist=data['robustness_checklist'],
            data_guidelines=data['data_guidelines'],
            hypotheses=data['hypotheses']
        )
