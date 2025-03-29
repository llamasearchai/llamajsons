#!/usr/bin/env python3
"""
Example script demonstrating how to use the llama-explorer API programmatically
"""

import asyncio
import os
import sys
from rich.console import Console
from rich.panel import Panel

# Add the parent directory to the path so we can import from src
sys.path.insert(0, os.path.abspath('..'))
from src.llama_explorer import LlamaExplorer, GitHubExplorer, Config

console = Console()

async def analyze_pypi_package():
    """Analyze a PyPI package manually"""
    console.print(Panel.fit(
        "[bold green]Analyzing PyPI package[/bold green]",
        title="Example 1"
    ))
    
    # Configure the explorer
    config = Config(
        pypi_url="https://pypi.org/pypi/requests/",
        package_name="requests",
        output_dir="./example_output",
        output_formats=["txt", "md", "json"],
        include_tests=True,
        verbose=True
    )
    
    # Create and run the explorer
    explorer = LlamaExplorer(config)
    success = await explorer.process_package()
    
    if success:
        console.print("[bold green]✓[/bold green] Successfully analyzed PyPI package")
    else:
        console.print("[bold red]✗[/bold red] Failed to analyze PyPI package")

async def analyze_github_repo():
    """Analyze a GitHub repository manually"""
    console.print(Panel.fit(
        "[bold blue]Analyzing GitHub repository[/bold blue]",
        title="Example 2"
    ))
    
    # Create explorer with GitHub repository URL
    explorer = GitHubExplorer(
        repo_url="https://github.com/psf/requests",
        base_output_dir="./example_output",
        formats=["txt", "md", "json"]
    )
    
    # Process the repository
    success = await explorer.process_repository()
    
    if success:
        console.print("[bold green]✓[/bold green] Successfully analyzed GitHub repository")
    else:
        console.print("[bold red]✗[/bold red] Failed to analyze GitHub repository")

async def main():
    """Main entry point"""
    console.print(Panel.fit(
        "[bold magenta]Llama Explorer - API Usage Examples[/bold magenta]",
        border_style="magenta"
    ))
    
    # Create output directory if it doesn't exist
    os.makedirs("example_output", exist_ok=True)
    
    # Run examples
    try:
        # Example 1: Analyze a PyPI package
        await analyze_pypi_package()
        
        console.print("\n" + "-" * 50 + "\n")
        
        # Example 2: Analyze a GitHub repository
        await analyze_github_repo()
        
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        import traceback
        console.print(traceback.format_exc())
    
    console.print("\n[bold green]Examples completed![/bold green]")

if __name__ == "__main__":
    asyncio.run(main()) 