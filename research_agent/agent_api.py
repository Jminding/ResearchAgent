"""API interface for agent.py - allows programmatic execution."""

import asyncio
import os
from pathlib import Path
from typing import Optional, Callable, Dict, Any
from dotenv import load_dotenv
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions, AgentDefinition, HookMatcher
from anthropic import Anthropic

from .utils.subagent_tracker import SubagentTracker
from .utils.transcript import TranscriptWriter
from .utils.message_handler import process_assistant_message

# Load environment variables
load_dotenv()

# Paths to prompt files
PROMPTS_DIR = Path(__file__).parent / "prompts"


def load_prompt(filename: str) -> str:
    """Load a prompt from the prompts directory."""
    prompt_path = PROMPTS_DIR / filename
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read().strip()


async def generate_project_title(query: str) -> str:
    """
    Generate a concise title for the research project using Claude.

    Args:
        query: The full research query

    Returns:
        A concise title (5-10 words)
    """
    try:
        client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

        response = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=100,
            messages=[{
                "role": "user",
                "content": f"""Generate a concise, professional title (5-10 words) for this research project.
Only output the title, nothing else.

Research query: {query}

Title:"""
            }]
        )

        # Extract title from response
        title = response.content[0].text.strip()

        # Clean up - remove quotes if present
        title = title.strip('"\'')

        # Fallback: if title is too long, truncate
        if len(title) > 100:
            title = title[:97] + "..."

        return title

    except Exception as e:
        # Fallback: use first 50 chars of query
        return query[:50] + ("..." if len(query) > 50 else "")


async def run_research_query(
    query: str,
    session_dir: Path,
    on_progress: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Run a single research query programmatically.

    This function wraps the agent.py logic to enable programmatic execution
    without CLI interaction. It reuses all the same agent definitions, hooks,
    and tracking mechanisms.

    Args:
        query: Research query string
        session_dir: Directory to save session logs
        on_progress: Optional async callback for progress updates

    Returns:
        Dict with session results:
        {
            'status': 'completed'|'failed',
            'session_dir': str,
            'transcript_path': str,
            'tool_log_path': str,
            'error': str (if failed)
        }

    Raises:
        ValueError: If ANTHROPIC_API_KEY not found
        Exception: If agent execution fails
    """
    # Check API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise ValueError("ANTHROPIC_API_KEY not found in environment")

    # Generate project title
    project_title = await generate_project_title(query)
    title_file = session_dir / "title.txt"
    with open(title_file, "w", encoding="utf-8") as f:
        f.write(project_title)

    # Setup transcript
    transcript_file = session_dir / "transcript.txt"
    transcript = TranscriptWriter(transcript_file)

    # Load prompts (same as agent.py)
    lead_agent_prompt = load_prompt("lead_agent.txt")
    literature_review_prompt = load_prompt("researcher.txt")
    theorist_prompt = load_prompt("theory.txt")
    experimental_design_prompt = load_prompt("experimental_design.txt")
    data_collector_prompt = load_prompt("data-collector.txt")
    experimentalist_prompt = load_prompt("experimentalist.txt")
    analyst_prompt = load_prompt("analyst.txt")
    report_writer_prompt = load_prompt("report_writer.txt")
    latex_compiler_prompt = load_prompt("latex_compiler.txt")

    # Initialize tracker
    tracker = SubagentTracker(transcript_writer=transcript, session_dir=session_dir)

    # Define agents (same as agent.py)
    agents = {
        "literature-reviewer": AgentDefinition(
            description=(
                "Use this agent at the very beginning of a research project to survey prior work. "
                "The literature-reviewer performs deep academic research using web search, focusing on "
                "peer-reviewed papers, preprints, and technical reports. It extracts methodologies, "
                "assumptions, datasets, and quantitative results from prior studies and writes structured, "
                "citation-ready notes to files/research_notes/. These notes are later reused directly in "
                "the literature review section of the final paper."
            ),
            tools=["WebSearch", "Write"],
            prompt=literature_review_prompt,
            model="haiku"
        ),
        "theorist": AgentDefinition(
            description=(
                "Use this agent after the literature review to formalize the research problem. "
                "The theorist constructs a mathematical or conceptual framework, defines variables and "
                "assumptions, proposes a falsifiable hypothesis if applicable to the topic at hand, and designs a "
                "step-by-step experimental plan in pseudocode. Outputs are written to files/theory/ "
                "and serve as the authoritative blueprint for the experimentalist."
            ),
            tools=["Write"],
            prompt=theorist_prompt,
            model="opus"
        ),
        "experimental-designer": AgentDefinition(
            description=(
                "Use this agent after the theorist to design comprehensive experiment plans with parameter grids, "
                "ablations, and robustness checks. The experimental-designer reads the EvidenceSheet from literature "
                "review and the theory framework to create an ExperimentPlan JSON specifying all configurations to test, "
                "systematic ablation studies, domain-appropriate robustness checks, and data selection guidelines. "
                "This agent ensures experiments are rigorous, grounded in literature evidence, and test multiple "
                "configurations rather than single points. Output is saved to files/theory/experiment_plan.json."
            ),
            tools=["Read", "Write"],
            prompt=experimental_design_prompt,
            model="sonnet"
        ),
        "data-collector": AgentDefinition(
            description=(
                "Use this agent when empirical data is required to test the theory. "
                "The data-collector searches for, evaluates, and documents real-world datasets "
                "from sources such as Kaggle, UCI, institutional repositories, or paper supplements. "
                "It records dataset properties, limitations, and access instructions in files/data/. "
                "If no suitable dataset exists, it justifies the need for synthetic data."
            ),
            tools=["WebSearch", "Write"],
            prompt=data_collector_prompt,
            model="sonnet"
        ),
        "experimentalist": AgentDefinition(
            description=(
                "Use this agent to implement and run experiments derived from the theorist's framework. "
                "The experimentalist translates pseudocode into executable code, runs simulations or "
                "models, uses datasets provided by the data-collector when applicable, and iteratively "
                "refines the implementation to improve results. Saves code to files/experiments/ "
                "and outputs results to files/results/."
            ),
            tools=["Read", "Write", "Bash"],
            prompt=experimentalist_prompt,
            model="opus"
        ),
        "analyst": AgentDefinition(
            description=(
                "Use this agent after experiments have completed to interpret results. "
                "The analyst reads outputs from files/results/, evaluates performance metrics, "
                "compares results to the original hypothesis, identifies trends and anomalies, "
                "and summarizes implications without running new experiments. Writes structured "
                "analysis to files/results/ for use in the final paper."
            ),
            tools=["Read", "Write"],
            prompt=analyst_prompt,
            model="sonnet"
        ),
        "report-writer": AgentDefinition(
            description=(
                "Use this agent at the end of the research pipeline to produce a publication-ready paper. "
                "The report-writer reads outputs from files/research_notes/, files/theory/, files/data/, "
                "files/results/, and files/charts/, then synthesizes them into a complete LaTeX research "
                "paper with abstract, methodology, results, discussion, and bibliography. "
                "It does NOT conduct new research or analysis and saves the final manuscript to "
                "files/reports/."
            ),
            tools=["Glob", "Read", "Write"],
            prompt=report_writer_prompt,
            model="sonnet"
        ),
        "latex-compiler": AgentDefinition(
            description=(
                "Use this agent after the report-writer to compile LaTeX manuscripts into PDFs. "
                "The latex-compiler finds .tex files in files/reports/, runs pdflatex and bibtex "
                "to generate PDFs, handles compilation errors by reading log files and fixing issues, "
                "and ensures all references, citations, and cross-references are properly resolved. "
                "This is the final step that produces the publication-ready PDF document."
            ),
            tools=["Read", "Write", "Bash"],
            prompt=latex_compiler_prompt,
            model="sonnet"
        ),
    }

    # Setup hooks
    hooks = {
        'PreToolUse': [
            HookMatcher(
                matcher=None,
                hooks=[tracker.pre_tool_use_hook]
            )
        ],
        'PostToolUse': [
            HookMatcher(
                matcher=None,
                hooks=[tracker.post_tool_use_hook]
            )
        ]
    }

    options = ClaudeAgentOptions(
        permission_mode="bypassPermissions",
        setting_sources=["project"],
        system_prompt=lead_agent_prompt,
        allowed_tools=["Task"],
        agents=agents,
        hooks=hooks,
        model="haiku"
    )

    try:
        async with ClaudeSDKClient(options=options) as client:
            # Write query to transcript
            transcript.write_to_file(f"You: {query}\n")

            # Extract session_id from session_dir (e.g., "session_20251222_182508")
            session_id = session_dir.name

            # Include session_id in query so agents can use it for file naming
            query_with_session = f"{query}\n\n[SESSION_ID: {session_id}]"

            # Send query
            await client.query(prompt=query_with_session)

            transcript.write("Agent: ", end="")

            # Process response
            async for msg in client.receive_response():
                if type(msg).__name__ == 'AssistantMessage':
                    process_assistant_message(msg, tracker, transcript)

                    # Optional progress callback
                    if on_progress:
                        try:
                            await on_progress({
                                'type': 'message',
                                'content': str(msg)
                            })
                        except Exception as e:
                            print(f"Progress callback error: {e}")

            transcript.write("\n")

        # Success
        transcript.write("\nResearch complete.\n")
        return {
            'status': 'completed',
            'session_dir': str(session_dir),
            'transcript_path': str(transcript_file),
            'tool_log_path': str(session_dir / 'tool_calls.jsonl')
        }

    except Exception as e:
        # Error handling
        error_msg = f"\nError: {str(e)}\n"
        transcript.write(error_msg)
        return {
            'status': 'failed',
            'session_dir': str(session_dir),
            'transcript_path': str(transcript_file),
            'tool_log_path': str(session_dir / 'tool_calls.jsonl'),
            'error': str(e)
        }

    finally:
        transcript.close()
        tracker.close()
