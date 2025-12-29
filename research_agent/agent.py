"""Entry point for research agent using AgentDefinition for subagents."""

import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions, AgentDefinition, HookMatcher

from .utils.subagent_tracker import SubagentTracker
from .utils.transcript import setup_session, TranscriptWriter
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


async def chat():
    """Start interactive chat with the research agent."""

    # Check API key first, before creating any files
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("\nError: ANTHROPIC_API_KEY not found.")
        print("Set it in a .env file or export it in your shell.")
        print("Get your key at: https://console.anthropic.com/settings/keys\n")
        return

    # Setup session directory and transcript
    transcript_file, session_dir = setup_session()

    # Create transcript writer
    transcript = TranscriptWriter(transcript_file)

    # Load prompts
    lead_agent_prompt = load_prompt("lead_agent.txt")
    literature_review_prompt = load_prompt("researcher.txt")
    theorist_prompt = load_prompt("theory.txt")
    experimental_design_prompt = load_prompt("experimental_design.txt")
    data_collector_prompt = load_prompt("data-collector.txt")
    experimentalist_prompt = load_prompt("experimentalist.txt")
    analyst_prompt = load_prompt("analyst.txt")
    report_writer_prompt = load_prompt("report_writer.txt")
    peer_reviewer_prompt = load_prompt("peer-reviewer.txt")

    # Initialize subagent tracker with transcript writer and session directory
    tracker = SubagentTracker(transcript_writer=transcript, session_dir=session_dir)

    # Define specialized subagents
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
                "Use this agent to implement and run experiments derived from the theorist’s framework. "
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
                "Use this agent after experiments have completed to interpret results with rigorous statistical analysis. "
                "The analyst reads outputs from files/results/, evaluates performance metrics using bootstrap CIs and "
                "statistical tests, compares results to the original hypothesis with p-values, generates AnalysisSummary "
                "JSON files for key comparisons, and proposes follow-up hypotheses when primary hypotheses fail. "
                "It uses the statistics module for rigorous testing and writes structured analysis, comparison JSONs, "
                "and FollowUpPlan to files/results/."
            ),
            tools=["Read", "Write", "Bash"],
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

        # "peer-reviewer": AgentDefinition(
        #     description=(
        #         "Use this agent to critically evaluate the research paper before submission. "
        #         "The peer-reviewer performs a rigorous, multi-pass review of the LaTeX manuscript, "
        #         "checking for soundness of methodology, validity of results, clarity of presentation, "
        #         "and adherence to academic standards. It provides detailed feedback, requests revisions "
        #         "if necessary, and ensures the paper meets the criteria for acceptance at a top-tier venue."
        #     ),
        #     tools=["Glob", "Read", "Write"],
        #     prompt=peer_reviewer_prompt,
        #     model="opus"
        # ),

    }


    # Set up hooks for tracking
    hooks = {
        'PreToolUse': [
            HookMatcher(
                matcher=None,  # Match all tools
                hooks=[tracker.pre_tool_use_hook]
            )
        ],
        'PostToolUse': [
            HookMatcher(
                matcher=None,  # Match all tools
                hooks=[tracker.post_tool_use_hook]
            )
        ]
    }

    options = ClaudeAgentOptions(
        permission_mode="bypassPermissions",
        setting_sources=["project"],  # Load skills from project .claude directory
        system_prompt=lead_agent_prompt,
        allowed_tools=["Task"],
        agents=agents,
        hooks=hooks,
        model="haiku"
    )

    print("\n" + "=" * 50)
    print("  Research Agent")
    print("=" * 50)
    print("\nResearch any topic.")
    print("\nType 'exit' to quit.\n")

    try:
        async with ClaudeSDKClient(options=options) as client:
            while True:
                # Get input
                try:
                    user_input = input("\nYou: ").strip()
                except (EOFError, KeyboardInterrupt):
                    break

                if not user_input or user_input.lower() in ["exit", "quit", "q"]:
                    break

                # Write user input to transcript (file only, not console)
                transcript.write_to_file(f"\nYou: {user_input}\n")

                # Send to agent
                await client.query(prompt=user_input)

                transcript.write("\nAgent: ", end="")

                # Stream and process response
                async for msg in client.receive_response():
                    if type(msg).__name__ == 'AssistantMessage':
                        process_assistant_message(msg, tracker, transcript)

                transcript.write("\n")
    finally:
        transcript.write("\n\nGoodbye!\n")
        transcript.close()
        tracker.close()
        print(f"\nSession logs saved to: {session_dir}")
        print(f"  - Transcript: {transcript_file}")
        print(f"  - Tool calls: {session_dir / 'tool_calls.jsonl'}")


if __name__ == "__main__":
    asyncio.run(chat())
