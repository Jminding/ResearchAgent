# Multi-Agent Research System

A comprehensive multi-agent research system that conducts autonomous scientific research from literature review through publication-quality PDF generation. The system coordinates 8 specialized subagents to execute a complete research pipeline: literature review, theory formalization, experimental design, data collection, experimentation, statistical analysis, and report writing.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Make Django migrations
cd research_platform
python manage.py makemigrations
python manage.py migrate

# Run the server
python manage.py runserver
```

## How It Works

The system executes a complete 11-step scientific research pipeline:

1. **Lead Agent Orchestration** - Decomposes research query into 2-4 distinct subtopics
2. **Parallel Literature Review** - Spawns 2-4 literature-reviewer subagents simultaneously; each creates `evidence_sheet.json` with quantitative metrics
3. **Wait & Verify** - Lead agent confirms all literature reviews complete and evidence sheet exists
4. **Theory Formalization** - Theorist subagent formalizes mathematical/conceptual framework and hypothesis
5. **Experimental Design** - Experimental-designer creates `experiment_plan.json` specifying parameter grids, ablations, robustness checks
6. **Data Collection** - Data-collector identifies real-world datasets or justifies synthetic data
7. **Experimentation** - Experimentalist implements and executes all configurations → `results_table.json`
8. **Statistical Analysis** - Analyst performs hypothesis tests with 95% CIs and p-values → `comparison_*.json`
9. **Follow-up Experiments** - If primary hypothesis fails (discovery mode), automatically proposes and executes diagnostic experiments
10. **Report Writing** - Report-writer synthesizes all outputs into publication-ready LaTeX manuscript
11. **PDF Compilation** - LaTeX-compiler generates final PDF with error handling

## Agents

The system uses **Anthropic's Claude Agent SDK** to define 8 specialized subagents, each with specific models, tools, and outputs:

| Agent | Model | Tools | Purpose | Outputs |
|-------|-------|-------|---------|---------|
| **lead-agent** | Haiku | Task | Orchestrates entire pipeline; spawns subagents sequentially | Session logs |
| **literature-reviewer** | Haiku | WebSearch, Write | Surveys academic literature; creates quantitative evidence sheet | `lit_review_*.md`, `evidence_sheet.json` |
| **theorist** | Opus | Write | Formalizes mathematical/conceptual framework; writes pseudocode blueprint | `theory_*.md` |
| **experimental-designer** | Sonnet | Read, Write | Designs experiment configurations with parameter grids and ablations | `experiment_plan.json` |
| **data-collector** | Sonnet | WebSearch, Write | Identifies real-world datasets; justifies synthetic data if needed | `dataset_*.md` |
| **experimentalist** | Opus | Read, Write, Bash | Implements and executes all experiment configurations | `results_table.json`, `results_table.csv`, experiment code |
| **analyst** | Sonnet | Read, Write, Bash | Performs statistical analysis; tests hypotheses; proposes follow-ups | `comparison_*.json`, `analysis_summary.json`, `followup_plan.json` |
| **report-writer** | Sonnet | Glob, Read, Write | Synthesizes all outputs into publication-ready LaTeX manuscript | `*_paper.tex` |
| **latex-compiler** | Sonnet | Read, Write, Bash | Compiles .tex to PDF; fixes compilation errors | Final PDF report |

### Agent Coordination

**Sequential Dependency Chain:**
```
Literature Review → Theorist (reads evidence_sheet.json)
  → Experimental Designer (reads evidence_sheet + theory)
    → Data Collector (reads experiment_plan)
      → Experimentalist (reads experiment_plan + data docs)
        → Analyst (reads results_table + experiment_plan + evidence_sheet)
          → Report Writer (reads ALL outputs)
            → LaTeX Compiler (compiles .tex)
```

**Parallel Execution:**
- Lead agent spawns 2-4 literature reviewers simultaneously for different subtopics
- All must complete before theorist stage begins

**Mixed Model Strategy:**
- **Opus**: Complex reasoning tasks (theory formalization, experimentation)
- **Sonnet**: Intermediate tasks (experimental design, analysis, report writing)
- **Haiku**: Orchestration and literature review (cost-effective for coordination)

## Data Structures

The system uses type-safe data classes for structured communication between agents. These enable explicit handoffs and prevent misunderstandings:

**Core Classes** (from `research_agent/data_structures.py`):

- **EvidenceSheet**: Quantitative findings from literature
  - Metric ranges, sample sizes, known pitfalls, academic references
  - Provides baseline for hypothesis testing

- **ExperimentPlan**: Specifies all configurations to test
  - Parameter grids (e.g., `learning_rate: [0.001, 0.01, 0.1]`)
  - Ablations (e.g., remove dropout, change activation function)
  - Robustness checklists (domain-specific requirements)
  - Data collection guidelines

- **ExperimentConfig**: Single experiment specification
  - Parameter sweep definitions
  - Expected runtime estimates

- **ResultsTable**: Structured output from experimentalist
  - Config name, parameters, metrics, standard errors
  - Enables programmatic analysis

- **AnalysisSummary**: Statistical comparison results
  - Metric name, 95% confidence intervals, p-values
  - Conclusions backed by statistical tests

- **FollowUpPlan**: Diagnostic hypotheses
  - Generated when primary hypothesis fails
  - Proposes targeted experiments to identify root causes

- **RobustnessChecklist**: Domain-specific robustness requirements
  - E.g., for ML: convergence analysis, sensitivity to hyperparameters

All classes support JSON serialization/deserialization for file-based agent communication.

## Key Features

- **Parallel Research**: Multiple subagents research different subtopics simultaneously for faster literature coverage
- **Statistical Rigor**: Bootstrap confidence intervals, Diebold-Mariano tests, hypothesis tests with p-values
- **Structured Communication**: Type-safe data classes prevent inter-agent misunderstandings
- **Adaptive Inquiry**: Automatically proposes follow-up diagnostic experiments if primary hypothesis fails
- **Reproducibility**: All code, configurations, data, and analysis saved; full audit trail in session logs
- **Mixed Model Strategy**: Optimizes cost/performance by using Opus for complex reasoning, Sonnet for intermediate tasks, Haiku for orchestration
- **Web Integration Ready**: Programmatic API (`agent_api.py`) enables integration with web applications

## Example Queries

**Scientific Research:**
- "Research quantum error correction codes and compare stabilizer vs. surface codes"
- "Investigate transformer attention mechanisms and test scaled dot-product vs. alternative variants"
- "Analyze renewable energy storage solutions and benchmark lithium-ion vs. flow batteries"

**Machine Learning:**
- "Compare gradient descent optimizers (SGD, Adam, RMSprop) on image classification tasks"
- "Evaluate regularization techniques (dropout, L2, early stopping) for preventing overfitting"

**Algorithm Analysis:**
- "Benchmark sorting algorithms (quicksort, mergesort, heapsort) across different data distributions"

## Output Structure

Research outputs are organized in two directories:

```
files/
├── research_notes/     # Literature review outputs
│   ├── lit_review_*.md
│   └── evidence_sheet.json
├── theory/             # Theory formalization documents
│   └── theory_*.md
├── data/               # Dataset documentation
│   └── dataset_*.md
├── experiments/        # Experiment code and configurations
│   ├── experiment_*.py
│   └── experiment_plan.json
├── results/            # Experiment results
│   ├── results_table.json
│   ├── results_table.csv
│   ├── comparison_*.json
│   ├── analysis_summary.json
│   └── followup_plan.json (if needed)
├── charts/             # PNG visualizations (referenced in paper)
│   └── *.png
└── reports/            # Final LaTeX manuscript and PDF
    ├── *_paper.tex
    └── *_paper.pdf

logs/
└── session_YYYYMMDD_HHMMSS/
    ├── transcript.txt      # Human-readable conversation
    ├── tool_calls.jsonl    # Structured tool usage log
    └── agent_prompts.txt   # Full system prompts for debugging
```

## Project Structure

```
Research Agent/
│
├── research_agent/              # Core multi-agent research system
│   ├── agent.py                 # CLI entry point (interactive mode)
│   ├── agent_api.py             # Programmatic API (for web integration)
│   ├── data_structures.py       # Type-safe data classes for inter-agent communication
│   ├── statistics.py            # Statistical analysis tools (bootstrap CIs, hypothesis tests)
│   ├── prompts/                 # Agent prompt templates (12 specialized prompts)
│   │   ├── lead_agent.txt       # Pipeline orchestration logic
│   │   ├── researcher.txt       # Literature review strategy
│   │   ├── theory.txt           # Theory formalization guidelines
│   │   ├── experimental_design.txt
│   │   ├── data_collector.txt
│   │   ├── experimentalist.txt
│   │   ├── analyst.txt
│   │   ├── report_writer.txt
│   │   └── latex_compiler.txt
│   └── utils/
│       ├── subagent_tracker.py  # Tracks tool calls via SDK hooks
│       ├── transcript.py        # Session logging
│       └── message_handler.py   # Processes assistant responses
│
├── research_platform/           # Django web application
│   ├── agents/                  # Main Django app
│   │   ├── models.py            # Database models (UserProfile, ResearchSession, etc.)
│   │   ├── views.py             # Web views (dashboard, session detail, downloads)
│   │   ├── services.py          # ResearchAgentService (bridge to research_agent/)
│   │   └── encryption.py        # API key encryption with Fernet
│   ├── research_platform/       # Django settings
│   ├── templates/               # HTML templates
│   ├── static/                  # CSS, JavaScript
│   └── manage.py                # Django management commands
│
├── backend/                     # FastAPI REST + WebSocket server
│   ├── main.py                  # FastAPI app initialization
│   ├── api/                     # REST endpoints
│   │   ├── research.py          # Research submission
│   │   ├── sessions.py          # Session management
│   │   └── websocket.py         # Real-time updates
│   └── services/
│       ├── session_manager.py   # Session discovery and parsing
│       └── file_watcher.py      # Monitors tool_calls.jsonl for updates
│
├── frontend/                    # React + TypeScript UI
│   └── src/
│       ├── pages/               # Dashboard, NewResearch, SessionDetail
│       ├── components/          # PipelinePhaseIndicator, SubagentCard, ToolCallTimeline
│       ├── contexts/            # SessionContext (state management)
│       └── services/            # API client (Axios)
│
└── files/                       # Research outputs (generated at runtime)
```

### Component Roles

**research_agent/** - Core Multi-Agent Research System
- Standalone CLI tool for running research
- Can be used directly via `python research_agent/agent.py`
- Generates research papers through multi-agent coordination
- Uses Anthropic's Claude Agent SDK
- Entry points:
  - `agent.py` - Interactive CLI mode
  - `agent_api.py` - Programmatic API (used by web integration)

**research_platform/** - Django Web Application
- User authentication and profile management
- Encrypted API key storage (Fernet symmetric encryption)
- Session persistence in relational database
- File management for research outputs
- Peer review feedback mechanism for iterative improvements
- Admin dashboard

**backend/** - FastAPI REST + WebSocket Server
- REST API for research submission and session management
- WebSocket streaming for real-time progress updates
- File watcher monitors `tool_calls.jsonl` for new events
- Broadcasts tool calls and subagent spawns to connected clients

**frontend/** - React + TypeScript UI
- Modern web interface for research management
- Dashboard with session overview and status tracking
- Live progress visualization (pipeline phases, subagent activity, tool calls)
- Real-time updates via WebSocket connection

## Architecture Overview

The system has two operational modes:

### 1. Standalone CLI Mode

```
User (Terminal)
  → research_agent/agent.py
  → Claude API (multi-agent execution)
  → files/ (research outputs)
```

Use this for direct research execution without the web interface.

### 2. Web Application Mode

```
User (Browser)
  → React Frontend (UI)
  → FastAPI Backend (REST + WebSocket)
  → Django Platform (auth, persistence, file management)
  → research_agent/agent_api.py (programmatic API)
  → Claude API (multi-agent execution)
  → files/ (research outputs)
```

The web application provides:
- User authentication and API key encryption
- Session history and management
- Real-time progress tracking with visual pipeline indicators
- File downloads (PDFs, CSVs, logs)
- Peer review feedback for iterative improvements

**Integration Points:**
- Django's `ResearchAgentService` calls `research_agent.agent_api.run_research_query()`
- FastAPI's `FileWatcher` monitors `logs/session_*/tool_calls.jsonl` for real-time updates
- React components subscribe to WebSocket for live progress display

## Subagent Tracking with Hooks

The system tracks all tool calls using **SDK hooks** to enable debugging, logging, and real-time progress visualization in the web UI.

### What Gets Tracked

- **Who**: Which agent (LITERATURE-REVIEWER-1, EXPERIMENTALIST-1, etc.)
- **What**: Tool name (WebSearch, Write, Bash, etc.)
- **When**: Timestamp of invocation
- **Input/Output**: Parameters passed and results returned

### How It Works

Hooks intercept every tool call before and after execution:

```python
from anthropic_agent.hooks import Hooks

hooks = Hooks(
    pre_tool_use=[tracker.pre_tool_use_hook],
    post_tool_use=[tracker.post_tool_use_hook]
)
```

The `parent_tool_use_id` links tool calls to their subagent:
- Lead Agent spawns a Researcher via `Task` tool → gets ID "task_123"
- All tool calls from that Researcher include `parent_tool_use_id = "task_123"`
- Hooks use this ID to identify which subagent made the call

### Log Output

**transcript.txt** - Human-readable conversation:
```
You: Research quantum error correction codes...

Agent: [Spawning LITERATURE-REVIEWER-1: stabilizer codes]
[LITERATURE-REVIEWER-1] → WebSearch (query='stabilizer codes quantum error correction')
[LITERATURE-REVIEWER-1] → Write (file='files/research_notes/lit_review_stabilizer_codes.md')

[Spawning EXPERIMENTALIST-1: implement experiments]
[EXPERIMENTALIST-1] → Read (file='files/theory/experiment_plan.json')
[EXPERIMENTALIST-1] → Bash (command='python experiments/run_qec_simulation.py')
```

**tool_calls.jsonl** - Structured JSON (enables web UI real-time updates):
```json
{"event":"tool_call_start","agent_id":"LITERATURE-REVIEWER-1","tool_name":"WebSearch","timestamp":"2025-01-15T10:23:45Z","query":"stabilizer codes"}
{"event":"tool_call_complete","agent_id":"LITERATURE-REVIEWER-1","success":true,"output_size":15234}
{"event":"subagent_spawn","agent_id":"EXPERIMENTALIST-1","parent":"lead-agent","timestamp":"2025-01-15T10:25:12Z"}
```

### Web UI Integration

The FastAPI backend's `FileWatcher` monitors `tool_calls.jsonl`:
- Polls every 500ms for new entries
- Parses JSON events
- Broadcasts via WebSocket to connected React clients
- React components update in real-time:
  - Pipeline phase indicators advance
  - Subagent cards display active agents
  - Tool call timeline shows chronological activity

This enables users to watch research progress live in the browser without refreshing.

## Statistical Analysis

The system includes comprehensive statistical tools in `research_agent/statistics.py`:

**Bootstrap Confidence Intervals:**
- Non-parametric resampling for metric uncertainty quantification
- Configurable confidence levels (default: 95%)
- Handles small sample sizes robustly

**Hypothesis Testing:**
- Diebold-Mariano test for comparing predictive accuracy
- Paired t-tests for metric comparisons
- Multiple testing correction (Bonferroni, Holm-Bonferroni)

**Risk-Adjusted Metrics:**
- Sharpe ratio calculations
- Drawdown analysis
- Custom risk metrics per domain

All statistical claims in generated papers are backed by these rigorous tests, with p-values and confidence intervals reported transparently.

## Research Modes

The system supports two research modes:

**Discovery Mode** (default):
- If primary hypothesis fails statistical tests, automatically generates `followup_plan.json`
- Proposes diagnostic experiments to identify root causes
- Executes highest-priority follow-up automatically
- Iterates until hypothesis supported or conclusive negative result

**Demo Mode** (`mode=demo`):
- Single-pass execution without follow-ups
- Faster execution for demonstrations
- Still includes full statistical analysis

Specify mode in initial query or via command-line argument.

## Memory Management

The system automatically detects available system RAM and applies memory limits:

**Default Behavior:**
- Limits research agent to 25% of system RAM
- Prevents runaway processes during experimentation
- Configurable via `RESEARCH_AGENT_MEMORY_LIMIT` environment variable

**Production Recommendation:**
- Set explicit limits based on workload
- Monitor memory usage during large-scale experiments
- Consider containerization (Docker) with resource constraints

## Security

**API Key Encryption:**
- User API keys encrypted with Fernet (symmetric encryption)
- Master key stored in `ENCRYPTION_KEY` environment variable
- Keys only decrypted in memory during research execution
- Never logged or exposed in plaintext

**Authentication:**
- Django user authentication required for all operations
- Session-based auth for web interface
- CORS configured for localhost development

**File Access:**
- Users can only access their own research sessions
- File downloads require authentication
- Session directories isolated per user

## Reproducibility

Every research session is fully reproducible:

**Saved Artifacts:**
- All experiment code with parameter configurations
- Complete datasets or dataset documentation
- Statistical analysis scripts
- Raw results (CSV, JSON)
- Session logs with full conversation history
- Agent prompts used for each subagent

**Audit Trail:**
- `transcript.txt` provides human-readable execution flow
- `tool_calls.jsonl` provides machine-readable structured log
- `agent_prompts.txt` shows exact prompts given to each agent

To reproduce a session:
1. Navigate to `logs/session_YYYYMMDD_HHMMSS/`
2. Review `transcript.txt` for research context
3. Check `files/experiments/` for code and configurations
4. Rerun experiments with same parameters
5. Compare results against `files/results/results_table.csv`

# Credits
This project is based on the research agent from the Anthropic Team's Claude Agent SDK docs. The original research agent was a search and summarization agent that searched for information regarding a specified topic and returned a report of what it found. This project significantly expands upon that agent by enabling it to conduct scientific research and simulations and giving it a more easily accessible user interface.