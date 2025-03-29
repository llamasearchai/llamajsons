#!/usr/bin/env python3
"""
Command-line entry point for llama-explorer
"""
import asyncio
import sys
import argparse
import threading
import time
import os
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich import box
from .explorer import main, check_dependencies, Config, LlamaExplorer, GitHubExplorer
from .ui import automated_mode, validate_url

# Initialize console
console = Console()

# ASCII Art for the banner
LLAMA_ASCII = r"""
[bright_magenta]                 ,
               ,~
              (  )
             ( |  )
            (__|__)
           /| |/\| |\\
          / | |  | | \\
         /__|_|__|_|__\\
        (    (    )    )
        |    |    |    |
        |    |    |    |
        |____|____|____|
        |____|____|____|[/bright_magenta]
"""

def show_banner():
    """Display the application banner"""
    console.print("\n")
    console.print(LLAMA_ASCII, justify="center")
    console.print("[bold magenta]Llama Explorer[/bold magenta]", justify="center")
    console.print("[dim]Extract and analyze packages & repositories with style[/dim]", justify="center")
    console.print("\n")

def spinner(stop_event):
    """Display a spinner while processing"""
    spinner_chars = ['⣾', '⣽', '⣻', '⢿', '⡿', '⣟', '⣯', '⣷']
    i = 0
    try:
        while not stop_event.is_set():
            sys.stdout.write('\r' + spinner_chars[i % len(spinner_chars)] + ' Processing...')
            sys.stdout.flush()
            time.sleep(0.1)
            i += 1
    except:
        pass
    finally:
        sys.stdout.write('\r                      \r')
        sys.stdout.flush()

def prompt_for_url():
    """Prompt the user for a URL to analyze"""
    show_banner()
    
    console.print(Panel(
        "Enter a [bold cyan]package name[/bold cyan], [bold green]GitHub repository[/bold green], or [bold yellow]URL[/bold yellow] to analyze.\n"
        "Examples:\n"
        "  • [bold cyan]requests[/bold cyan] (PyPI package)\n"
        "  • [bold green]pandas-dev/pandas[/bold green] (GitHub repository)\n"
        "  • [bold yellow]https://github.com/psf/requests[/bold yellow] (Full URL)",
        title="🦙 Enter what you want to analyze",
        border_style="cyan"
    ))
    
    url = Prompt.ask("[bold]>[/bold]")
    return url

def main_cli():
    """CLI entry point that handles async execution"""
    # Create argument parser
    parser = argparse.ArgumentParser(
        description="Extract and analyze PyPI packages or GitHub repositories with style"
    )
    parser.add_argument(
        "package", 
        nargs="?", 
        help="PyPI package name, URL, or GitHub repository (optional)"
    )
    parser.add_argument(
        "-o", "--output-dir", 
        default="llama_output",
        help="Output directory for reports (default: llama_output)"
    )
    parser.add_argument(
        "-f", "--formats", 
        default="txt,md,json",
        help="Comma-separated list of output formats (default: txt,md,json)"
    )
    parser.add_argument(
        "--include-tests", 
        action="store_true",
        help="Include test files in the report"
    )
    parser.add_argument(
        "--api",
        action="store_true",
        help="Run as an API server"
    )
    parser.add_argument(
        "-p", "--port",
        type=int,
        default=8001,
        help="Port to run the API server on (default: 8001)"
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Run in interactive mode with prompts for input"
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Run in automatic mode with no user interaction (generates a master report)"
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Run in legacy mode (without generating master report)"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Run in quiet mode with minimal output"
    )
    parser.add_argument(
        "-v", "--version",
        action="store_true",
        help="Show version information and exit"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Show version and exit if requested
    if args.version:
        from . import __version__
        console.print(f"[bold magenta]Llama Explorer[/bold magenta] v{__version__}")
        console.print("[dim]Copyright (c) 2023-2025 Llama Explorer Team[/dim]")
        sys.exit(0)
    
    # First check that dependencies are available
    deps = check_dependencies()
    if not deps["ok"]:
        console.print("[bold red]ERROR: Missing required dependencies:[/bold red]", ", ".join(deps["missing"]))
        if deps["optional_missing"]:
            console.print("[yellow]WARNING: Missing optional dependencies:[/yellow]", ", ".join(deps["optional_missing"]))
        console.print("Please install missing dependencies with: pip install [package_name]")
        sys.exit(1)
    
    # Run as API server if requested
    if args.api:
        console.print("[bold green]Starting API server...[/bold green]")
        asyncio.run(main(
            package_input=None,
            output_dir=args.output_dir,
            formats=[],
            include_tests=args.include_tests,
            api_mode=True,
            port=args.port
        ))
        return
    
    # Run in interactive mode if requested
    if args.interactive:
        from .ui import interactive_mode
        asyncio.run(interactive_mode())
        return
    
    # If no package is provided, prompt for one
    if not args.package:
        package_input = prompt_for_url()
        if not package_input:
            console.print("[bold red]No input provided. Exiting.[/bold red]")
            sys.exit(1)
    else:
        package_input = args.package
    
    # Validate the URL/package name
    is_valid, url_type, metadata = validate_url(package_input)
    if not is_valid:
        console.print(f"[bold red]Invalid input format: {package_input}[/bold red]")
        console.print("Please provide a valid PyPI package name, GitHub repository, or URL.")
        sys.exit(1)
    
    # Run in auto mode (default for all inputs now)
    if not args.legacy:
        # Start a spinner in a separate thread if not in quiet mode
        stop_spinner = threading.Event()
        spinner_thread = None
        
        if not args.quiet:
            spinner_thread = threading.Thread(target=spinner, args=(stop_spinner,))
            spinner_thread.daemon = True
            spinner_thread.start()
        
        try:
            asyncio.run(automated_mode(package_input, args.output_dir))
        finally:
            if spinner_thread:
                stop_spinner.set()
                spinner_thread.join(timeout=1.0)
        
        return
    
    # Run in legacy mode if explicitly requested
    if args.legacy:
        asyncio.run(main(
            package_input=package_input,
            output_dir=args.output_dir,
            formats=args.formats.split(","),
            include_tests=args.include_tests,
            api_mode=False,
            port=args.port
        ))
        return

if __name__ == "__main__":
    try:
        main_cli()
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[bold red]Error: {str(e)}[/bold red]")
        sys.exit(1) 