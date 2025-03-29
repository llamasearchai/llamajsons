# 🦙 Llama Explorer

**Extract, analyze and explore PyPI packages and GitHub repositories with style**

Llama Explorer is a powerful tool that lets you explore and analyze Python packages from PyPI and repositories from GitHub. It generates comprehensive reports about package structure, dependencies, repository organization, and metadata in various formats.

## Features

- 🔍 **Deep Package Analysis**: Analyzes the structure and contents of any PyPI package
- 📁 **GitHub Repository Analysis**: Examines GitHub repositories for structure, languages, and contributors
- 📊 **Comprehensive Reports**: Generates detailed reports in multiple formats (TXT, Markdown, JSON, PDF)
- 🤖 **Automated Mode**: Process everything automatically with a master report and no user input required (default)
- 🌐 **Interactive Mode**: User-friendly interface to guide you through analysis with prompts
- 🛡️ **Enhanced Scraping**: Uses stealth techniques to avoid rate limiting and blocks
- 🚀 **User-Friendly Interface**: Beautiful console interface with progress indicators
- 📦 **Package Information**: Extracts metadata, dependencies, and structure information
- 🧩 **File Categorization**: Automatically categorizes files by type (Python, documentation, tests, etc.)
- 👥 **Contributor Analysis**: For GitHub repositories, shows top contributors and their statistics
- 🔄 **Automatic Error Recovery**: Attempts to generate partial reports even when analysis fails
- 🎯 **Smart URL Detection**: Automatically detects and processes various URL formats

## Installation

```bash
# Install the basic package
pip install llama-explorer

# Install with all optional dependencies
pip install llama-explorer[all]

# Install with specific optional dependencies
pip install llama-explorer[pdf]  # For PDF report generation
pip install llama-explorer[scraper]  # For enhanced web scraping
```

## Usage

### Super Simple Usage

```bash
# Just run the command and it will prompt you for what to analyze
llama-explorer

# Or provide a package name or repository directly
llama-explorer requests
llama-explorer pandas-dev/pandas
```

### Command Line Options

```bash
# Fully automated mode with master report (DEFAULT)
llama-explorer requests
llama-explorer https://github.com/username/repo
llama-explorer username/repo

# Run in quiet mode with minimal output
llama-explorer requests -q

# Display version information
llama-explorer -v

# Legacy mode (no master report)
llama-explorer requests --legacy

# Interactive mode (guided experience with prompts)
llama-explorer -i

# Analyze with custom output directory
llama-explorer requests -o ./my_reports

# Run as API server
llama-explorer --api --port 8001
```

### Command Line Reference

| Option | Description |
|--------|-------------|
| `PACKAGE` | Package name, URL or GitHub repository to analyze |
| `-i, --interactive` | Run in interactive mode with user prompts |
| `--legacy` | Run in legacy mode without master report |
| `-o, --output-dir DIR` | Output directory for reports (default: llama_output) |
| `-f, --formats FORMATS` | Comma-separated list of output formats (default: txt,md,json) |
| `--include-tests` | Include test files in analysis (legacy mode) |
| `--api` | Run as API server |
| `-p, --port PORT` | Port to run API server on (default: 8001) |
| `-q, --quiet` | Run in quiet mode with minimal output |
| `-v, --version` | Show version information and exit |
| `-h, --help` | Show help message and exit |

### Automated Mode (DEFAULT)

The automated mode processes packages or repositories with minimal user input, and generates a comprehensive master report. This is now the default behavior when you provide a package name or repository URL:

```bash
# These commands all use automated mode by default:
llama-explorer requests
llama-explorer pandas
llama-explorer https://github.com/username/repo
llama-explorer username/repo

# Output will include:
# - llama_output/master_requests.txt (comprehensive master report)
# - llama_output/requests_report.txt (standard report)
```

The master report contains:
- Full package/repository information
- Dependency analysis
- File structure categorization
- Language breakdown
- Security notes
- And more!

### Interactive Mode

The interactive mode provides a guided experience with prompts for:
- URL or package name (supports PyPI packages and GitHub repositories)
- Output formats selection
- Output directory

To use interactive mode:
```bash
llama-explorer -i
```

### Python API

```python
import asyncio
from llama_explorer import LlamaExplorer, GitHubExplorer, automated_mode

# Fully automated analysis (recommended)
async def auto_analyze():
    # Simply pass the package or repo URL
    await automated_mode("requests")  # or "username/repo" or "https://github.com/username/repo"

# Analyze a PyPI package (legacy approach)
async def analyze_package():
    # For more control, you can use the explorer classes directly
    config = Config(
        pypi_url="https://pypi.org/pypi/requests/",
        package_name="requests",
        output_dir="./reports",
        output_formats=["txt", "md", "json"],
        include_tests=True
    )
    explorer = LlamaExplorer(config)
    success = await explorer.process_package()
    
    if success:
        print("Package analysis complete!")

# Run the async functions
asyncio.run(auto_analyze())  # Recommended approach
```

## Optional Dependencies

- **PDF Reports**: Install `pdfkit` for PDF report generation
- **Enhanced Scraping**: Install `cloudscraper` for better web scraping capabilities
- **GitHub Repository**: Install `gitpython` for cloning and analyzing GitHub repositories

## License

MIT License - Copyright (c) 2023-2025 Llama Explorer Team 