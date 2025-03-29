#!/usr/bin/env python3
"""
Llama Explorer - Main script to run the program with different examples

This script demonstrates various ways to use llama-explorer to analyze PyPI packages 
and GitHub repositories, including:
1. Analyzing a PyPI package (automatic URL detection)
2. Analyzing a GitHub repository (automatic URL detection)
3. Generating comprehensive reports in different formats
"""

import asyncio
import os
import argparse
import sys
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

# Import from the src directory
sys.path.insert(0, os.path.abspath('.'))
from src.llama_explorer import automated_mode, interactive_mode
from src.llama_explorer.explorer import LlamaExplorer, GitHubExplorer, Config

console = Console()

async def analyze_package_or_repo(target, output_dir="llama_output", formats=None):
    """Automatically detect and analyze a package or repository"""
    try:
        console.print(Panel(
            f"[bold green]Starting automatic analysis of:[/bold green] [cyan]{target}[/cyan]\n"
            f"Output directory: [yellow]{output_dir}[/yellow]",
            title="🦙 Llama Explorer",
            border_style="green"
        ))
        
        await automated_mode(target, output_dir=output_dir, formats=formats or ["txt", "md", "json"])
        
        console.print(Panel(
            f"[bold green]Analysis complete![/bold green]\n"
            f"Reports have been saved to: [yellow]{output_dir}[/yellow]",
            title="🦙 Llama Explorer",
            border_style="green"
        ))
        
    except Exception as e:
        console.print(Panel(
            f"[bold red]Error during analysis:[/bold red] {str(e)}",
            title="🦙 Llama Explorer",
            border_style="red"
        ))
        if "--debug" in sys.argv:
            import traceback
            console.print(traceback.format_exc())

async def run_interactive():
    """Run the interactive mode"""
    try:
        console.print(Panel(
            "[bold green]Starting interactive mode[/bold green]\n"
            "You will be prompted for input to configure the analysis.",
            title="🦙 Llama Explorer",
            border_style="blue"
        ))
        
        await interactive_mode()
        
    except Exception as e:
        console.print(Panel(
            f"[bold red]Error during interactive mode:[/bold red] {str(e)}",
            title="🦙 Llama Explorer",
            border_style="red"
        ))
        if "--debug" in sys.argv:
            import traceback
            console.print(traceback.format_exc())

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Llama Explorer - Analyze packages and repositories")
    
    # Main commands
    parser.add_argument("target", nargs="?", help="Package name, GitHub repo, or URL to analyze")
    parser.add_argument("-i", "--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("-o", "--output-dir", default="llama_output", help="Output directory")
    parser.add_argument("-f", "--formats", default="txt,md,json", help="Comma-separated list of output formats")
    parser.add_argument("--debug", action="store_true", help="Show debug information on errors")
    
    return parser.parse_args()

async def main():
    """Main entry point"""
    args = parse_args()
    
    # Draw fancy header
    console.print("\n")
    console.print(Panel.fit(
        "[bold magenta]🦙 LLAMA EXPLORER[/bold magenta]\n"
        "[italic]Extract, analyze and explore PyPI packages and GitHub repositories[/italic]",
        border_style="magenta"
    ))
    console.print("\n")
    
    if args.interactive:
        # Run in interactive mode
        await run_interactive()
    elif args.target:
        # Run with the provided target
        formats = args.formats.split(",") if args.formats else ["txt", "md", "json"]
        await analyze_package_or_repo(args.target, args.output_dir, formats)
    else:
        # No arguments provided, ask for input
        try:
            choice = Prompt.ask(
                "What would you like to do?",
                choices=["interactive", "analyze", "quit"],
                default="interactive"
            )
            
            if choice == "interactive":
                await run_interactive()
            elif choice == "analyze":
                target = Prompt.ask("Enter a package name, GitHub repo, or URL to analyze")
                if target:
                    await analyze_package_or_repo(target, args.output_dir)
            else:
                console.print("[yellow]Exiting program[/yellow]")
        except KeyboardInterrupt:
            console.print("\n[yellow]Program interrupted by user[/yellow]")
    
    console.print("\n[bold green]Thank you for using Llama Explorer![/bold green]")

if __name__ == "__main__":
    asyncio.run(main()) 