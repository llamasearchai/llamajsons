#!/usr/bin/env python3
"""
Simple demo script for llama-explorer Python API
"""

import asyncio
import sys
from rich.console import Console
from rich.panel import Panel

# Import llama-explorer components
from llama_explorer import automated_mode, LlamaExplorer, GitHubExplorer
from llama_explorer import Config
console = Console()

async def demo_automated():
    """Demonstrate the automated mode (recommended approach)"""
    console.print(Panel.fit(
        "[bold green]Automated Mode Demo[/bold green]\n"
        "This is the simplest way to use llama-explorer",
        title="Example 1"
    ))
    
    # Choose a package to analyze
    package = "requests"
    console.print(f"[yellow]Analyzing package:[/yellow] {package}")
    
    # Run automated analysis
    await automated_mode(package, output_dir="./demo_output")
    
    console.print("[bold green]✓[/bold green] Automated analysis complete! Check the ./demo_output directory")

async def demo_manual_pypi():
    """Demonstrate manual PyPI package analysis"""
    console.print(Panel.fit(
        "[bold blue]Manual PyPI Analysis Demo[/bold blue]\n"
        "This shows how to use the LlamaExplorer class directly",
        title="Example 2"
    ))
    
    # Configure the explorer
    config = Config(
        pypi_url="https://pypi.org/pypi/fastapi/",
        package_name="fastapi",
        output_dir="./demo_output",
        output_formats=["txt", "md", "json"],
        include_tests=True
    )
    
    # Create and run the explorer
    explorer = LlamaExplorer(config)
    console.print("[yellow]Analyzing PyPI package:[/yellow] fastapi")
    
    success = await explorer.process_package()
    
    if success:
        console.print("[bold green]✓[/bold green] Package analysis complete! Check the ./demo_output directory")
    else:
        console.print("[bold red]✗[/bold red] Package analysis failed")

async def demo_github():
    """Demonstrate GitHub repository analysis"""
    console.print(Panel.fit(
        "[bold magenta]GitHub Repository Analysis Demo[/bold magenta]\n"
        "This shows how to use the GitHubExplorer class directly",
        title="Example 3"
    ))
    
    # Configure the explorer
    config = Config(
        github_url="https://github.com/tiangolo/fastapi",
        output_dir="./demo_output",
        output_formats=["txt", "md", "json"],
    )
    
    # Create and run the explorer
    explorer = GitHubExplorer(config)
    console.print("[yellow]Analyzing GitHub repository:[/yellow] tiangolo/fastapi")
    
    success = await explorer.process_repository()
    
    if success:
        console.print("[bold green]✓[/bold green] Repository analysis complete! Check the ./demo_output directory")
    else:
        console.print("[bold red]✗[/bold red] Repository analysis failed")

async def main():
    """Run all demos"""
    console.print("[bold]Llama Explorer API Demo[/bold]", style="blue on white", justify="center")
    console.print("This script demonstrates different ways to use the llama-explorer Python API\n")
    
    try:
        # Run the automated demo (recommended approach)
        await demo_automated()
        
        console.print("\n" + "-" * 50 + "\n")
        
        # Run the manual PyPI demo
        await demo_manual_pypi()
        
        console.print("\n" + "-" * 50 + "\n")
        
        # Run the GitHub demo
        await demo_github()
        
    except KeyboardInterrupt:
        console.print("\n[bold red]Demo interrupted by user[/bold red]")
        return 1
    except Exception as e:
        console.print(f"\n[bold red]Error during demo:[/bold red] {str(e)}")
        return 1
    
    console.print("\n[bold green]All demos completed successfully![/bold green]")
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main())) 