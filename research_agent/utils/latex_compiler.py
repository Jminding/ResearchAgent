"""Utility for compiling LaTeX files to PDF."""

import subprocess
import shutil
from pathlib import Path
from typing import Tuple, Optional


def compile_latex_to_pdf(tex_file: Path, output_dir: Optional[Path] = None) -> Tuple[bool, str]:
    """
    Compile a LaTeX file to PDF using pdflatex.

    Args:
        tex_file: Path to the .tex file
        output_dir: Optional directory for output (defaults to same as tex_file)

    Returns:
        Tuple of (success: bool, message: str)
    """
    # Check if pdflatex is available
    if not shutil.which("pdflatex"):
        return False, "pdflatex not found. Please install a LaTeX distribution (e.g., TeX Live, MiKTeX)"

    # Validate input
    if not tex_file.exists():
        return False, f"LaTeX file not found: {tex_file}"

    if not tex_file.suffix == '.tex':
        return False, f"File is not a .tex file: {tex_file}"

    # Set output directory
    if output_dir is None:
        output_dir = tex_file.parent

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Standard LaTeX compilation sequence for proper references, citations, and TOC:
        # 1. pdflatex - generates .aux file
        # 2. bibtex - processes bibliography (if .bib files exist)
        # 3. pdflatex - incorporates bibliography
        # 4. pdflatex - resolves all cross-references and TOC

        # First pass - generate aux file
        result = subprocess.run(
            [
                'pdflatex',
                '-interaction=nonstopmode',
                '-output-directory', str(output_dir),
                str(tex_file)
            ],
            capture_output=True,
            text=True,
            cwd=tex_file.parent,
            timeout=60
        )

        if result.returncode != 0:
            error_log = output_dir / f"{tex_file.stem}.log"
            error_msg = f"pdflatex failed on first pass. Return code: {result.returncode}\n"

            if error_log.exists():
                with open(error_log, 'r', encoding='utf-8', errors='ignore') as f:
                    log_content = f.read()
                    error_lines = [line for line in log_content.split('\n')
                                 if '!' in line or 'Error' in line or 'error' in line]
                    if error_lines:
                        error_msg += "Errors:\n" + "\n".join(error_lines[:10])

            return False, error_msg

        # Check if bibtex is needed (if .aux has citations)
        aux_file = output_dir / f"{tex_file.stem}.aux"
        run_bibtex = False
        if aux_file.exists():
            with open(aux_file, 'r', encoding='utf-8', errors='ignore') as f:
                aux_content = f.read()
                if '\\citation' in aux_content or '\\bibdata' in aux_content:
                    run_bibtex = True

        # Run bibtex if needed
        if run_bibtex and shutil.which("bibtex"):
            bibtex_result = subprocess.run(
                ['bibtex', str(aux_file)],
                capture_output=True,
                text=True,
                cwd=tex_file.parent,
                timeout=30
            )
            # Bibtex errors are not fatal - continue anyway

        # Second pass - incorporate bibliography and update references
        for run in [2, 3]:
            result = subprocess.run(
                [
                    'pdflatex',
                    '-interaction=nonstopmode',
                    '-output-directory', str(output_dir),
                    str(tex_file)
                ],
                capture_output=True,
                text=True,
                cwd=tex_file.parent,
                timeout=60
            )

            # Only fail on critical errors in final pass
            if result.returncode != 0 and run == 3:
                error_log = output_dir / f"{tex_file.stem}.log"
                error_msg = f"pdflatex failed on pass {run}. Return code: {result.returncode}\n"

                if error_log.exists():
                    with open(error_log, 'r', encoding='utf-8', errors='ignore') as f:
                        log_content = f.read()
                        error_lines = [line for line in log_content.split('\n')
                                     if '!' in line or 'Error' in line or 'error' in line]
                        if error_lines:
                            error_msg += "Errors:\n" + "\n".join(error_lines[:10])

                return False, error_msg

        # Check if PDF was created
        pdf_file = output_dir / f"{tex_file.stem}.pdf"
        if not pdf_file.exists():
            return False, "PDF file was not created despite successful pdflatex execution"

        # Clean up auxiliary files
        for ext in ['.aux', '.log', '.out', '.toc', '.lof', '.lot']:
            aux_file = output_dir / f"{tex_file.stem}{ext}"
            if aux_file.exists():
                try:
                    aux_file.unlink()
                except Exception:
                    pass  # Ignore cleanup errors

        return True, f"Successfully compiled {tex_file.name} to {pdf_file}"

    except subprocess.TimeoutExpired:
        return False, "pdflatex compilation timed out after 60 seconds"

    except Exception as e:
        return False, f"Compilation error: {str(e)}"


def find_and_compile_reports(session_dir: Path) -> list[Tuple[Path, bool, str]]:
    """
    Find all .tex files in the global reports directory and compile them.

    Note: Agents write to a global files/reports/ directory (relative to project root),
    not to the session directory. This function looks in the correct location.

    Args:
        session_dir: Path to session directory (used to determine project root)

    Returns:
        List of tuples: (tex_file_path, success, message)
    """
    # Agents write to files/reports/ relative to project root
    # Session dir is typically: <project_root>/research_agent/logs/session_YYYYMMDD_HHMMSS
    # So we need to go up to project root, then into files/reports/

    # Find project root (parent of research_agent directory)
    project_root = session_dir
    while project_root.name != 'logs' and project_root.parent != project_root:
        project_root = project_root.parent

    # Go up two more levels: logs -> research_agent -> project_root
    if project_root.name == 'logs':
        project_root = project_root.parent.parent

    reports_dir = project_root / 'files' / 'reports'

    if not reports_dir.exists():
        return []

    results = []

    # Find all .tex files
    tex_files = list(reports_dir.glob('*.tex'))

    for tex_file in tex_files:
        success, message = compile_latex_to_pdf(tex_file)
        results.append((tex_file, success, message))

    return results
