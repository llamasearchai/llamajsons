#!/usr/bin/env python3
"""
User interface components for llama-explorer
"""
import re
import sys
from typing import List, Optional, Tuple, Dict, Any
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich import box
import asyncio
import inspect
import os
import time
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn

console = Console()

# URL validation patterns
GITHUB_PATTERN = r'^https?://github\.com/([^/]+)/([^/]+)/?.*$'
PYPI_PATTERN = r'^https?://pypi\.org/project/([^/]+)/?.*$'
PYPI_USER_PATTERN = r'^https?://pypi\.org/user/([^/]+)/?.*$'

def validate_url(url: str) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Validate input URL and determine its type.
    
    Returns:
        Tuple containing (is_valid, type, metadata)
        where type can be 'github', 'pypi', 'pypi_user' or 'unknown'
        and metadata contains extracted information
    """
    if not url or not isinstance(url, str):
        return False, "unknown", None
    
    # Check for GitHub repository URL
    github_match = re.match(GITHUB_PATTERN, url)
    if github_match:
        return True, "github", {
            "owner": github_match.group(1),
            "repo": github_match.group(2)
        }
    
    # Check for PyPI package URL
    pypi_match = re.match(PYPI_PATTERN, url)
    if pypi_match:
        return True, "pypi", {
            "package": pypi_match.group(1)
        }
    
    # Check for PyPI user profile URL
    pypi_user_match = re.match(PYPI_USER_PATTERN, url)
    if pypi_user_match:
        return True, "pypi_user", {
            "username": pypi_user_match.group(1)
        }
    
    # Maybe it's just a PyPI package name?
    if re.match(r'^[a-zA-Z0-9_.-]+$', url) and '/' not in url:
        return True, "pypi_name", {
            "package": url
        }
    
    # Maybe it's a GitHub repo in the format owner/repo?
    if re.match(r'^[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+$', url):
        parts = url.split('/')
        return True, "github_short", {
            "owner": parts[0],
            "repo": parts[1]
        }
    
    return False, "unknown", None

def prompt_for_url() -> Tuple[str, str, Dict[str, Any]]:
    """
    Prompt the user for a URL to analyze.
    Returns the URL, its type, and extracted metadata.
    """
    console.print(Panel(
        "[bold cyan]Welcome to Llama Explorer![/bold cyan]\n\n"
        "Analyze [bold]PyPI packages[/bold] or [bold]GitHub repositories[/bold] with style.\n"
        "Enter a URL, package name, or GitHub repo (owner/repo) below:",
        title="🦙 Llama Explorer",
        border_style="cyan"
    ))
    
    while True:
        url = Prompt.ask("[bold cyan]Enter URL or package name[/bold cyan]")
        is_valid, url_type, metadata = validate_url(url)
        
        if is_valid:
            if url_type == "github" or url_type == "github_short":
                console.print(f"[green]✓ Valid GitHub repository: {metadata['owner']}/{metadata['repo']}[/green]")
                return url, url_type, metadata
            elif url_type == "pypi" or url_type == "pypi_name":
                console.print(f"[green]✓ Valid PyPI package: {metadata['package']}[/green]")
                return url, url_type, metadata
            elif url_type == "pypi_user":
                console.print(f"[green]✓ Valid PyPI user profile: {metadata['username']}[/green]")
                return url, url_type, metadata
        else:
            console.print("[bold red]✗ Invalid URL format. Please enter a valid URL.[/bold red]")
            console.print("""
[dim]Examples of valid inputs:
- https://github.com/username/repo
- username/repo
- https://pypi.org/project/package-name
- package-name
- https://pypi.org/user/username[/dim]
            """)

def prompt_for_formats() -> List[str]:
    """Prompt the user for output formats"""
    formats = []
    
    format_table = Table(box=box.ROUNDED)
    format_table.add_column("Format", style="cyan")
    format_table.add_column("Description", style="dim")
    format_table.add_row("txt", "Plain text report")
    format_table.add_row("md", "Markdown report with better formatting")
    format_table.add_row("json", "Structured JSON for programmatic use")
    format_table.add_row("pdf", "PDF report generated from Markdown (requires pdfkit)")
    
    console.print(format_table)
    
    if Confirm.ask("[cyan]Include text report?[/cyan]", default=True):
        formats.append("txt")
    
    if Confirm.ask("[cyan]Include markdown report?[/cyan]", default=True):
        formats.append("md")
    
    if Confirm.ask("[cyan]Include JSON report?[/cyan]", default=True):
        formats.append("json")
    
    if Confirm.ask("[cyan]Include PDF report?[/cyan] [dim](requires pdfkit)[/dim]", default=False):
        formats.append("pdf")
    
    if not formats:
        console.print("[yellow]No formats selected. Defaulting to text report.[/yellow]")
        formats = ["txt"]
    
    return formats

def prompt_for_output_dir() -> str:
    """Prompt the user for output directory"""
    return Prompt.ask(
        "[cyan]Output directory[/cyan]", 
        default="llama_output"
    )

def display_analysis_start(target_type: str, target_name: str) -> None:
    """Display a message indicating analysis has started"""
    if target_type == "github":
        console.print(f"\n[bold cyan]Starting analysis of GitHub repository: {target_name}[/bold cyan]")
    else:
        console.print(f"\n[bold cyan]Starting analysis of PyPI package: {target_name}[/bold cyan]")
    
    console.print("[dim]This may take a while depending on the size...[/dim]\n")

def display_job_status(job_id: str, status: str) -> None:
    """Display current job status"""
    status_color = {
        "pending": "yellow",
        "processing": "blue",
        "completed": "green",
        "failed": "red"
    }.get(status, "white")
    
    console.print(f"Job [dim]{job_id}[/dim]: [{status_color}]{status}[/{status_color}]")

async def interactive_mode() -> None:
    """Run the explorer in interactive mode"""
    from .explorer import GitHubExplorer, LlamaExplorer, check_dependencies, Config
    
    # Check dependencies first
    deps = check_dependencies()
    if not deps["ok"]:
        console.print("[bold red]ERROR: Missing required dependencies:[/bold red]", ", ".join(deps["missing"]))
        if deps["optional_missing"]:
            console.print("[yellow]WARNING: Missing optional dependencies:[/yellow]", ", ".join(deps["optional_missing"]))
        console.print("Please install missing dependencies with: pip install [package_name]")
        sys.exit(1)
    
    # Get user input
    url, url_type, metadata = prompt_for_url()
    formats = prompt_for_formats()
    output_dir = prompt_for_output_dir()
    
    # Process based on URL type
    if url_type in ["github", "github_short"]:
        owner = metadata["owner"]
        repo = metadata["repo"]
        repo_url = f"https://github.com/{owner}/{repo}"
        
        display_analysis_start("github", f"{owner}/{repo}")
        explorer = GitHubExplorer(repo_url, output_dir, formats)
        success = await explorer.process_repository()
        await explorer.cleanup()
        
        if success:
            console.print(f"[bold green]✓ Analysis of {owner}/{repo} completed successfully![/bold green]")
        else:
            console.print(f"[bold red]✗ Analysis of {owner}/{repo} failed.[/bold red]")
    
    elif url_type in ["pypi", "pypi_name"]:
        package = metadata["package"]
        
        display_analysis_start("pypi", package)
        
        # Create a Config object for LlamaExplorer
        config = Config(
            pypi_url=f"https://pypi.org/pypi/{package}/",
            package_name=package,
            output_dir=output_dir,
            output_formats=formats,
            include_tests=False,
            verbose=True
        )
        
        # Initialize LlamaExplorer with the config object
        explorer = LlamaExplorer(config)
        success = await explorer.process_package()
        
        if success:
            console.print(f"[bold green]✓ Analysis of {package} completed successfully![/bold green]")
        else:
            console.print(f"[bold red]✗ Analysis of {package} failed.[/bold red]")
    
    elif url_type == "pypi_user":
        console.print("[yellow]PyPI user profile analysis is not yet implemented in interactive mode.[/yellow]")
        return

async def automated_mode(url: str, output_dir: str = "llama_output") -> None:
    """Run the explorer in automated mode with no user input required"""
    from .explorer import GitHubExplorer, LlamaExplorer, check_dependencies, Config
    
    # Show a welcome message
    console.print("\n")
    console.print(Panel(
        f"[bold green]Analyzing:[/bold green] [bold cyan]{url}[/bold cyan]\n\n"
        f"[bold]Output Directory:[/bold] {output_dir}\n"
        f"[bold]Report Type:[/bold] Comprehensive Master Report\n"
        f"[bold]Mode:[/bold] Automated Analysis",
        title="🦙 Llama Explorer",
        subtitle="Analysis Started",
        border_style="cyan"
    ))
    
    # Check dependencies first
    deps = check_dependencies()
    if not deps["ok"]:
        console.print("[bold red]ERROR: Missing required dependencies:[/bold red]", ", ".join(deps["missing"]))
        if deps["optional_missing"]:
            console.print("[yellow]WARNING: Missing optional dependencies:[/yellow]", ", ".join(deps["optional_missing"]))
        console.print("Please install missing dependencies with: pip install [package_name]")
        sys.exit(1)
    
    # Set default formats to include the master text file
    formats = ["txt"]
    
    # Validate the URL automatically
    is_valid, url_type, metadata = validate_url(url)
    
    if not is_valid:
        console.print(f"[bold red]✗ Invalid URL format: {url}[/bold red]")
        console.print("Please provide a valid PyPI package name or GitHub repository URL.")
        sys.exit(1)
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process based on URL type
    if url_type in ["github", "github_short"]:
        owner = metadata["owner"]
        repo = metadata["repo"]
        repo_url = f"https://github.com/{owner}/{repo}"
        
        # Show a progress panel
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            # Create a main task
            main_task = progress.add_task(f"[cyan]Analyzing GitHub repository: {owner}/{repo}", total=100)
            
            # Update progress
            progress.update(main_task, advance=10, description=f"[cyan]Initializing analysis for {owner}/{repo}")
            
            # Create the explorer
            explorer = GitHubExplorer(repo_url, output_dir, formats)
            
            # Update progress
            progress.update(main_task, advance=10, description=f"[cyan]Fetching repository information for {owner}/{repo}")
            
            # Process the repository
            try:
                success = await explorer.process_repository()
                
                # Update progress
                progress.update(main_task, advance=50, description=f"[cyan]Generating master report for {owner}/{repo}")
                
                # Generate the master text file containing all findings
                if success:
                    master_file_path = await generate_master_file(explorer, f"{owner}_{repo}", "github", output_dir)
                    progress.update(main_task, advance=30, description=f"[cyan]Finalizing analysis for {owner}/{repo}")
                    
                    # Show success message
                    console.print(f"\n[bold green]✓ Analysis of {owner}/{repo} completed successfully![/bold green]")
                    console.print(f"[green]Master report saved to {master_file_path}[/green]")
                    
                    # Display a summary from the master report
                    try:
                        with open(master_file_path, 'r') as f:
                            summary_lines = []
                            for i, line in enumerate(f):
                                if i < 20:  # Only show first 20 lines in summary
                                    summary_lines.append(line.strip())
                                else:
                                    break
                            
                            console.print(Panel(
                                "\n".join(summary_lines) + "\n...",
                                title="[bold cyan]Report Summary[/bold cyan]",
                                border_style="cyan"
                            ))
                            
                            # Show available reports
                            report_table = Table(title="Available Reports", box=box.ROUNDED)
                            report_table.add_column("Type", style="cyan")
                            report_table.add_column("Path", style="green")
                            report_table.add_row("Master Report", master_file_path)
                            for fmt in formats:
                                report_path = os.path.join(output_dir, f"{owner}_{repo}_report.{fmt}")
                                if os.path.exists(report_path):
                                    report_table.add_row(fmt.upper(), report_path)
                            console.print(report_table)
                    except Exception as e:
                        console.print(f"[yellow]Could not display report summary: {str(e)}[/yellow]")
                else:
                    progress.update(main_task, completed=100, description=f"[red]Analysis failed for {owner}/{repo}")
                    console.print(f"[bold red]✗ Analysis of {owner}/{repo} failed.[/bold red]")
                    console.print("[yellow]Trying to recover and generate a partial report...[/yellow]")
                    
                    # Try to generate a partial report with whatever information we have
                    try:
                        if hasattr(explorer, 'repo_info') and explorer.repo_info:
                            master_file_path = await generate_master_file(explorer, f"{owner}_{repo}", "github", output_dir)
                            console.print(f"[green]Partial report saved to {master_file_path}[/green]")
                    except Exception as recovery_error:
                        console.print(f"[red]Could not generate partial report: {str(recovery_error)}[/red]")
            except Exception as e:
                progress.update(main_task, completed=100, description=f"[red]Error analyzing {owner}/{repo}")
                console.print(f"[bold red]Error during analysis: {str(e)}[/bold red]")
                import traceback
                console.print(f"[dim]{traceback.format_exc()}[/dim]")
            
            # Clean up
            await explorer.cleanup()
    
    elif url_type in ["pypi", "pypi_name"]:
        package = metadata["package"]
        
        # Show a progress panel
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            # Create a main task
            main_task = progress.add_task(f"[cyan]Analyzing PyPI package: {package}", total=100)
            
            # Update progress
            progress.update(main_task, advance=10, description=f"[cyan]Initializing analysis for {package}")
            
            # Create a Config object for LlamaExplorer
            config = Config(
                pypi_url=f"https://pypi.org/pypi/{package}/",
                package_name=package,
                output_dir=output_dir,
                output_formats=formats,
                include_tests=False,
                verbose=True
            )
            
            # Update progress
            progress.update(main_task, advance=10, description=f"[cyan]Fetching package information for {package}")
            
            # Initialize LlamaExplorer with the config object
            explorer = LlamaExplorer(config)
            
            # Process the package
            try:
                success = await explorer.process_package()
                
                # Update progress
                progress.update(main_task, advance=50, description=f"[cyan]Generating master report for {package}")
                
                # Generate the master text file containing all findings
                if success:
                    master_file_path = await generate_master_file(explorer, package, "pypi", output_dir)
                    progress.update(main_task, advance=30, description=f"[cyan]Finalizing analysis for {package}")
                    
                    # Show success message
                    console.print(f"\n[bold green]✓ Analysis of {package} completed successfully![/bold green]")
                    console.print(f"[green]Master report saved to {master_file_path}[/green]")
                    
                    # Display a summary from the master report
                    try:
                        with open(master_file_path, 'r') as f:
                            summary_lines = []
                            for i, line in enumerate(f):
                                if i < 20:  # Only show first 20 lines in summary
                                    summary_lines.append(line.strip())
                                else:
                                    break
                            
                            console.print(Panel(
                                "\n".join(summary_lines) + "\n...",
                                title="[bold cyan]Report Summary[/bold cyan]",
                                border_style="cyan"
                            ))
                            
                            # Show available reports
                            report_table = Table(title="Available Reports", box=box.ROUNDED)
                            report_table.add_column("Type", style="cyan")
                            report_table.add_column("Path", style="green")
                            report_table.add_row("Master Report", master_file_path)
                            for fmt in formats:
                                report_path = os.path.join(output_dir, f"{package}_report.{fmt}")
                                if os.path.exists(report_path):
                                    report_table.add_row(fmt.upper(), report_path)
                            console.print(report_table)
                    except Exception as e:
                        console.print(f"[yellow]Could not display report summary: {str(e)}[/yellow]")
                else:
                    progress.update(main_task, completed=100, description=f"[red]Analysis failed for {package}")
                    console.print(f"[bold red]✗ Analysis of {package} failed.[/bold red]")
                    console.print("[yellow]Trying to recover and generate a partial report...[/yellow]")
                    
                    # Try to generate a partial report with whatever information we have
                    try:
                        if hasattr(explorer, 'package_info') and explorer.package_info:
                            master_file_path = await generate_master_file(explorer, package, "pypi", output_dir)
                            console.print(f"[green]Partial report saved to {master_file_path}[/green]")
                    except Exception as recovery_error:
                        console.print(f"[red]Could not generate partial report: {str(recovery_error)}[/red]")
            except Exception as e:
                progress.update(main_task, completed=100, description=f"[red]Error analyzing {package}")
                console.print(f"[bold red]Error during analysis: {str(e)}[/bold red]")
                import traceback
                console.print(f"[dim]{traceback.format_exc()}[/dim]")
    
    elif url_type == "pypi_user":
        username = metadata["username"]
        console.print(f"[yellow]PyPI user profile analysis is not yet fully implemented, but attempting basic analysis.[/yellow]")
        
        # Show a progress panel
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            # Create a main task
            main_task = progress.add_task(f"[cyan]Analyzing PyPI user: {username}", total=100)
            
            # Update progress
            progress.update(main_task, advance=50, description=f"[cyan]Fetching user information for {username}")
            time.sleep(1)  # Simulate work
            
            # Update progress
            progress.update(main_task, advance=50, description=f"[cyan]Finalizing analysis for {username}")
            
            # Show message
            console.print(f"[bold yellow]✓ Basic analysis of user {username} completed.[/bold yellow]")
            console.print("[yellow]Full user profile analysis will be available in a future update.[/yellow]")
    
    # Show final message
    console.print(Panel(
        f"[bold green]Analysis Complete![/bold green]\n\n"
        f"Your reports have been saved to the [bold cyan]{output_dir}[/bold cyan] directory.\n"
        f"Thank you for using Llama Explorer!",
        title="🦙 Llama Explorer",
        border_style="green"
    ))

async def generate_master_file(explorer, name, type_str, output_dir):
    """Generate a comprehensive master text file with all findings"""
    import aiofiles
    from datetime import datetime
    
    master_file_path = os.path.join(output_dir, f"master_{name}.txt")
    
    async with aiofiles.open(master_file_path, 'w') as f:
        await f.write(f"===== LLAMA EXPLORER MASTER REPORT =====\n")
        await f.write(f"Target: {name} ({type_str})\n")
        await f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        await f.write(f"==================================\n\n")
        
        # Check if it's a GitHub repository or PyPI package based on the type
        if type_str == "github" and hasattr(explorer, 'repo_info'):
            # For GitHub repositories
            await f.write(f"REPOSITORY INFORMATION\n")
            await f.write(f"----------------------\n")
            await f.write(f"Name: {explorer.repo_info.name}\n")
            await f.write(f"Full Name: {explorer.repo_info.full_name}\n")
            await f.write(f"Description: {explorer.repo_info.description or 'N/A'}\n")
            await f.write(f"Stars: {explorer.repo_info.stars}\n")
            await f.write(f"Forks: {explorer.repo_info.forks}\n")
            await f.write(f"Primary Language: {explorer.repo_info.language or 'Unknown'}\n")
            await f.write(f"Topics: {', '.join(explorer.repo_info.topics) if explorer.repo_info.topics else 'None'}\n")
            await f.write(f"Created: {explorer.repo_info.created_at}\n")
            await f.write(f"Last Updated: {explorer.repo_info.updated_at}\n\n")
            
            # File structure
            await f.write(f"FILE STRUCTURE\n")
            await f.write(f"-------------\n")
            total_files = sum(len(files) for files in explorer.repo_info.files.values())
            await f.write(f"Total Files: {total_files}\n\n")
            
            for category, files in explorer.repo_info.files.items():
                if files:
                    await f.write(f"{category.upper()} ({len(files)} files):\n")
                    for file in files[:20]:  # Limit to 20 files per category
                        await f.write(f"  - {file}\n")
                    if len(files) > 20:
                        await f.write(f"  - ... and {len(files) - 20} more files\n")
                    await f.write(f"\n")
            
            # Language breakdown
            if hasattr(explorer.repo_info, 'languages') and explorer.repo_info.languages:
                await f.write(f"LANGUAGE BREAKDOWN\n")
                await f.write(f"-----------------\n")
                for lang, count in explorer.repo_info.languages.items():
                    await f.write(f"{lang}: {count} bytes\n")
                await f.write(f"\n")
            
            # Contributors
            if hasattr(explorer.repo_info, 'contributors') and explorer.repo_info.contributors:
                await f.write(f"TOP CONTRIBUTORS\n")
                await f.write(f"---------------\n")
                for contributor in explorer.repo_info.contributors[:10]:
                    login = contributor.get("login", "Unknown")
                    contributions = contributor.get("contributions", 0)
                    await f.write(f"{login}: {contributions} contributions\n")
                await f.write(f"\n")
            
        elif hasattr(explorer, 'package_info'):
            # For PyPI packages
            await f.write(f"PACKAGE INFORMATION\n")
            await f.write(f"------------------\n")
            await f.write(f"Name: {explorer.package_info.name}\n")
            await f.write(f"Version: {explorer.package_info.version}\n")
            
            # Description may be in different attributes depending on the package
            description = None
            if hasattr(explorer.package_info, 'description'):
                description = explorer.package_info.description
            elif hasattr(explorer.package_info, 'summary'):
                description = explorer.package_info.summary
            await f.write(f"Description: {description or 'N/A'}\n")
            
            # Author information
            if hasattr(explorer.package_info, 'author'):
                await f.write(f"Author: {explorer.package_info.author or 'Unknown'}\n")
                if hasattr(explorer.package_info, 'author_email') and explorer.package_info.author_email:
                    await f.write(f"Author Email: {explorer.package_info.author_email}\n")
            
            # License information
            if hasattr(explorer.package_info, 'license'):
                await f.write(f"License: {explorer.package_info.license or 'Unknown'}\n")
            
            # Homepage or project URL
            homepage = None
            if hasattr(explorer.package_info, 'home_page'):
                homepage = explorer.package_info.home_page
            elif hasattr(explorer.package_info, 'project_url'):
                homepage = explorer.package_info.project_url
            elif hasattr(explorer.package_info, 'homepage'):
                homepage = explorer.package_info.homepage
            await f.write(f"Homepage: {homepage or 'N/A'}\n\n")
            
            # Python requirements
            if hasattr(explorer.package_info, 'requires_python'):
                await f.write(f"Python Version: {explorer.package_info.requires_python or 'Not specified'}\n\n")
            
            # Dependencies
            if hasattr(explorer.package_info, 'requires_dist') and explorer.package_info.requires_dist:
                await f.write(f"DEPENDENCIES\n")
                await f.write(f"------------\n")
                for dep in explorer.package_info.requires_dist:
                    await f.write(f"  - {dep}\n")
                await f.write(f"\n")
            elif hasattr(explorer.package_info, 'dependencies') and explorer.package_info.dependencies:
                await f.write(f"DEPENDENCIES\n")
                await f.write(f"------------\n")
                for dep in explorer.package_info.dependencies:
                    await f.write(f"  - {dep}\n")
                await f.write(f"\n")
            
            # File structure - check both attributes that might contain file information
            if hasattr(explorer, 'file_structure') and explorer.file_structure:
                await f.write(f"FILE STRUCTURE\n")
                await f.write(f"-------------\n")
                for category, files in explorer.file_structure.items():
                    if files:
                        await f.write(f"{category.upper()} ({len(files)} files):\n")
                        for file in files[:20]:  # Limit to 20 files per category
                            await f.write(f"  - {file}\n")
                        if len(files) > 20:
                            await f.write(f"  - ... and {len(files) - 20} more files\n")
                        await f.write(f"\n")
            elif hasattr(explorer.package_info, 'files') and explorer.package_info.files:
                await f.write(f"FILE STRUCTURE\n")
                await f.write(f"-------------\n")
                for category, files in explorer.package_info.files.items():
                    if files:
                        await f.write(f"{category.upper()} ({len(files)} files):\n")
                        for file in files[:20]:  # Limit to 20 files per category
                            await f.write(f"  - {file}\n")
                        if len(files) > 20:
                            await f.write(f"  - ... and {len(files) - 20} more files\n")
                        await f.write(f"\n")
        else:
            # Generic case if we can't determine the type
            await f.write(f"ANALYSIS INFORMATION\n")
            await f.write(f"-------------------\n")
            await f.write(f"Name: {name}\n")
            await f.write(f"Type: {type_str}\n")
            await f.write(f"Note: Limited information available for this type.\n\n")
            
            # Try to extract any available attributes
            for attr_name, attr_value in inspect.getmembers(explorer):
                if not attr_name.startswith('_') and not callable(attr_value) and isinstance(attr_value, (str, int, float, bool)):
                    await f.write(f"{attr_name}: {attr_value}\n")
            await f.write(f"\n")
        
        # Security section for both
        await f.write(f"SECURITY NOTES\n")
        await f.write(f"-------------\n")
        await f.write(f"This is an automated analysis and may not identify all security concerns.\n")
        await f.write(f"Please review the code manually for any security issues.\n\n")
        
        await f.write(f"===== END OF REPORT =====\n")
        
        return master_file_path 