"""
Data structures for the enhanced multi-agent research system.

These structured data classes enable type-safe communication between agents
and ensure consistent data formats throughout the research pipeline.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import json
from pathlib import Path


class ResearchDomain(Enum):
    """Research domain for tailored validation and robustness checks."""
    FINANCE = "finance"
    PDE = "pde"
    ML = "ml"
    QUANTUM = "quantum"
    GENERAL = "general"


class ExperimentStatus(Enum):
    """Status of an experimental configuration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Reference:
    """A literature reference with key findings."""
    shortname: str
    year: int
    finding: str
    doi: Optional[str] = None
    url: Optional[str] = None


@dataclass
class EvidenceSheet:
    """
    Structured evidence from literature review.

    This sheet captures quantitative facts, typical ranges, and known pitfalls
    from prior work to inform hypothesis setting and result validation.
    """
    metric_ranges: Dict[str, List[float]] = field(default_factory=dict)
    # e.g., {"large_cap_momentum_sharpe": [0.35, 0.42]}

    typical_sample_sizes: Dict[str, str] = field(default_factory=dict)
    # e.g., {"momentum_universe_size": "> 1000"}

    known_pitfalls: List[str] = field(default_factory=list)
    # e.g., ["survivorship_bias", "small_sample_instability"]

    key_references: List[Reference] = field(default_factory=list)

    domain: ResearchDomain = ResearchDomain.GENERAL

    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'metric_ranges': self.metric_ranges,
            'typical_sample_sizes': self.typical_sample_sizes,
            'known_pitfalls': self.known_pitfalls,
            'key_references': [
                {
                    'shortname': ref.shortname,
                    'year': ref.year,
                    'finding': ref.finding,
                    'doi': ref.doi,
                    'url': ref.url
                }
                for ref in self.key_references
            ],
            'domain': self.domain.value,
            'notes': self.notes
        }

    def to_json(self, filepath: Path) -> None:
        """Save to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvidenceSheet':
        """Load from dictionary."""
        references = [
            Reference(
                shortname=ref['shortname'],
                year=ref['year'],
                finding=ref['finding'],
                doi=ref.get('doi'),
                url=ref.get('url')
            )
            for ref in data.get('key_references', [])
        ]

        return cls(
            metric_ranges=data.get('metric_ranges', {}),
            typical_sample_sizes=data.get('typical_sample_sizes', {}),
            known_pitfalls=data.get('known_pitfalls', []),
            key_references=references,
            domain=ResearchDomain(data.get('domain', 'general')),
            notes=data.get('notes', '')
        )

    @classmethod
    def from_json(cls, filepath: Path) -> 'EvidenceSheet':
        """Load from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class ParameterGrid:
    """A parameter and its values to sweep over."""
    name: str
    values: List[Any]


@dataclass
class ExperimentConfig:
    """
    A single experiment configuration with parameter grid.

    Supports ablation studies and parameter sweeps.
    """
    name: str
    description: str
    parameters: Dict[str, List[Any]] = field(default_factory=dict)
    # e.g., {"frequency": ["weekly", "monthly"], "transaction_cost_bps": [5, 10]}

    ablations: List[str] = field(default_factory=list)
    # e.g., ["full_model", "no_constraints", "no_microstructure"]

    status: ExperimentStatus = ExperimentStatus.PENDING

    def get_grid_size(self) -> int:
        """Calculate total number of configurations in grid."""
        size = 1
        for values in self.parameters.values():
            size *= len(values)
        return size * max(1, len(self.ablations))


@dataclass
class RobustnessChecklist:
    """
    Robustness checks to perform for each experiment.

    Domain-specific requirements ensure results generalize.
    """
    hyperparameter_perturbations: List[str] = field(default_factory=list)
    # e.g., ["learning_rate_±25%", "batch_size_±50%"]

    additional_datasets: List[str] = field(default_factory=list)
    # Finance: ["small_caps", "intl_markets"]
    # PDE: ["different_ic", "different_parameters"]

    parameter_regimes: List[str] = field(default_factory=list)
    # PDE: ["low_volatility", "high_volatility"]

    required_checks: int = 3  # Minimum number of robustness checks

    notes: str = ""


@dataclass
class DataSelectionGuidelines:
    """
    Guidelines for data selection: real vs synthetic.

    Enforces preference for real data and transparency about synthetic data.
    """
    prefer_real_data: bool = True

    real_data_sources: List[str] = field(default_factory=list)
    # e.g., ["CRSP", "Compustat", "LOB dataset"]

    synthetic_data_justification: str = ""
    # Required if using synthetic data

    synthetic_data_generation_method: str = ""
    # How synthetic data was created

    known_synthetic_biases: List[str] = field(default_factory=list)
    # Documented limitations of synthetic data

    data_labeling: Dict[str, str] = field(default_factory=dict)
    # e.g., {"prices": "real", "fundamentals": "synthetic"}


@dataclass
class ExperimentPlan:
    """
    Complete experimental plan with configs, ablations, and robustness checks.

    This is the blueprint generated by the experimental design agent.
    """
    project_name: str
    experiments: List[ExperimentConfig] = field(default_factory=list)
    robustness_checklist: RobustnessChecklist = field(default_factory=RobustnessChecklist)
    data_guidelines: DataSelectionGuidelines = field(default_factory=DataSelectionGuidelines)

    hypotheses: List[str] = field(default_factory=list)
    # Falsifiable hypotheses to test

    expected_outcomes: Dict[str, Any] = field(default_factory=dict)
    # Based on evidence sheet, e.g., {"sharpe_range": [0.3, 0.5]}

    mode: str = "discovery"  # "discovery" or "demo"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'project_name': self.project_name,
            'experiments': [
                {
                    'name': exp.name,
                    'description': exp.description,
                    'parameters': exp.parameters,
                    'ablations': exp.ablations,
                    'status': exp.status.value
                }
                for exp in self.experiments
            ],
            'robustness_checklist': asdict(self.robustness_checklist),
            'data_guidelines': asdict(self.data_guidelines),
            'hypotheses': self.hypotheses,
            'expected_outcomes': self.expected_outcomes,
            'mode': self.mode
        }

    def to_json(self, filepath: Path) -> None:
        """Save to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentPlan':
        """Load from dictionary."""
        experiments = [
            ExperimentConfig(
                name=exp['name'],
                description=exp['description'],
                parameters=exp['parameters'],
                ablations=exp.get('ablations', []),
                status=ExperimentStatus(exp.get('status', 'pending'))
            )
            for exp in data.get('experiments', [])
        ]

        robustness = RobustnessChecklist(**data.get('robustness_checklist', {}))
        data_guidelines = DataSelectionGuidelines(**data.get('data_guidelines', {}))

        return cls(
            project_name=data['project_name'],
            experiments=experiments,
            robustness_checklist=robustness,
            data_guidelines=data_guidelines,
            hypotheses=data.get('hypotheses', []),
            expected_outcomes=data.get('expected_outcomes', {}),
            mode=data.get('mode', 'discovery')
        )

    @classmethod
    def from_json(cls, filepath: Path) -> 'ExperimentPlan':
        """Load from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class ExperimentResult:
    """A single experimental result."""
    config_name: str
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    # e.g., {"sharpe": 0.42, "max_drawdown": -0.15}

    ablation: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'config_name': self.config_name,
            'parameters': self.parameters,
            'metrics': self.metrics,
            'ablation': self.ablation,
            'error': self.error
        }


@dataclass
class ResultsTable:
    """
    Structured table of all experimental results.

    Enables systematic comparison and statistical analysis.
    """
    project_name: str
    results: List[ExperimentResult] = field(default_factory=list)

    def add_result(self, result: ExperimentResult) -> None:
        """Add a result to the table."""
        self.results.append(result)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'project_name': self.project_name,
            'results': [r.to_dict() for r in self.results]
        }

    def to_json(self, filepath: Path) -> None:
        """Save to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)

    def to_csv(self, filepath: Path) -> None:
        """Save to CSV file."""
        import csv

        if not self.results:
            return

        # Flatten results for CSV
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            # Get all parameter and metric keys
            param_keys = set()
            metric_keys = set()
            for r in self.results:
                param_keys.update(r.parameters.keys())
                metric_keys.update(r.metrics.keys())

            param_keys = sorted(param_keys)
            metric_keys = sorted(metric_keys)

            fieldnames = ['config_name', 'ablation'] + param_keys + metric_keys + ['error']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for r in self.results:
                row = {
                    'config_name': r.config_name,
                    'ablation': r.ablation or '',
                    'error': r.error or ''
                }
                row.update(r.parameters)
                row.update(r.metrics)
                writer.writerow(row)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ResultsTable':
        """Load from dictionary."""
        results = [
            ExperimentResult(
                config_name=r['config_name'],
                parameters=r['parameters'],
                metrics=r['metrics'],
                ablation=r.get('ablation'),
                error=r.get('error')
            )
            for r in data.get('results', [])
        ]

        return cls(
            project_name=data['project_name'],
            results=results
        )

    @classmethod
    def from_json(cls, filepath: Path) -> 'ResultsTable':
        """Load from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class AnalysisSummary:
    """
    Statistical analysis summary for a specific comparison.

    Provides rigorous statistical backing for claims in the final report.
    """
    comparison: str
    # e.g., "quarterly_vs_weekly" or "hybrid_vs_lstm"

    metric: str
    # e.g., "Sharpe", "RMSE", "AUC"

    estimate_diff: float
    # Point estimate of difference

    ci_95: Tuple[float, float]
    # 95% confidence interval

    p_value: Optional[float] = None
    # Statistical significance test p-value

    test_statistic: Optional[float] = None
    # e.g., t-statistic, DM statistic

    test_method: str = ""
    # e.g., "bootstrap", "diebold_mariano", "t_test"

    conclusion: str = ""
    # Plain language interpretation

    sample_size: Optional[int] = None

    additional_metrics: Dict[str, Any] = field(default_factory=dict)
    # Any other relevant statistics

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'comparison': self.comparison,
            'metric': self.metric,
            'estimate_diff': self.estimate_diff,
            'ci_95': list(self.ci_95),
            'p_value': self.p_value,
            'test_statistic': self.test_statistic,
            'test_method': self.test_method,
            'conclusion': self.conclusion,
            'sample_size': self.sample_size,
            'additional_metrics': self.additional_metrics
        }

    def to_json(self, filepath: Path) -> None:
        """Save to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnalysisSummary':
        """Load from dictionary."""
        return cls(
            comparison=data['comparison'],
            metric=data['metric'],
            estimate_diff=data['estimate_diff'],
            ci_95=tuple(data['ci_95']),
            p_value=data.get('p_value'),
            test_statistic=data.get('test_statistic'),
            test_method=data.get('test_method', ''),
            conclusion=data.get('conclusion', ''),
            sample_size=data.get('sample_size'),
            additional_metrics=data.get('additional_metrics', {})
        )

    @classmethod
    def from_json(cls, filepath: Path) -> 'AnalysisSummary':
        """Load from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class FollowUpHypothesis:
    """
    A diagnostic hypothesis to test when primary hypothesis fails.

    Enables systematic investigation of unexpected results.
    """
    hypothesis: str
    # e.g., "Constraints too strong"

    diagnostic_experiment: str
    # What to test, e.g., "Relax constraint strength by 50%"

    expected_outcome: str
    # What we'd see if hypothesis is correct

    priority: int = 1  # 1 = high, 2 = medium, 3 = low


@dataclass
class FollowUpPlan:
    """
    Plan for follow-up experiments when hypotheses fail.

    Generated by analysis agent when results contradict expectations.
    """
    trigger: str
    # What caused the need for follow-up, e.g., "Hybrid worse than LSTM"

    hypotheses: List[FollowUpHypothesis] = field(default_factory=list)

    selected_followup: Optional[str] = None
    # Which follow-up to run (in discovery mode)

    mode: str = "demo"  # "discovery" or "demo"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'trigger': self.trigger,
            'hypotheses': [
                {
                    'hypothesis': h.hypothesis,
                    'diagnostic_experiment': h.diagnostic_experiment,
                    'expected_outcome': h.expected_outcome,
                    'priority': h.priority
                }
                for h in self.hypotheses
            ],
            'selected_followup': self.selected_followup,
            'mode': self.mode
        }

    def to_json(self, filepath: Path) -> None:
        """Save to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FollowUpPlan':
        """Load from dictionary."""
        hypotheses = [
            FollowUpHypothesis(
                hypothesis=h['hypothesis'],
                diagnostic_experiment=h['diagnostic_experiment'],
                expected_outcome=h['expected_outcome'],
                priority=h.get('priority', 1)
            )
            for h in data.get('hypotheses', [])
        ]

        return cls(
            trigger=data['trigger'],
            hypotheses=hypotheses,
            selected_followup=data.get('selected_followup'),
            mode=data.get('mode', 'demo')
        )

    @classmethod
    def from_json(cls, filepath: Path) -> 'FollowUpPlan':
        """Load from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)
