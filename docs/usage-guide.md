# 🦙 Llama Explorer - Comprehensive Usage Guide

This guide provides detailed information on using Llama Explorer to analyze PyPI packages and GitHub repositories.

## Table of Contents

- [Installation](#installation)
- [Command Line Interface](#command-line-interface)
- [Analyzing PyPI Packages](#analyzing-pypi-packages)
- [Analyzing GitHub Repositories](#analyzing-github-repositories)
- [Output Formats and Reports](#output-formats-and-reports)
- [Interactive Mode](#interactive-mode)
- [API Server Mode](#api-server-mode)
- [Python API](#python-api)
- [Advanced Configuration](#advanced-configuration)
- [Troubleshooting](#troubleshooting)

## Installation

### Basic Installation

```bash
pip install llama-explorer
```

### Installing Optional Dependencies

Llama Explorer supports several optional features that require additional dependencies:

```bash
# For PDF report generation
pip install llama-explorer[pdf]

# For enhanced web scraping capabilities
pip install llama-explorer[scraper]

# For all optional dependencies
pip install llama-explorer[all]
```

### Development Installation

For development purposes, you can install Llama Explorer in editable mode:

```bash
git clone https://github.com/your-username/llama-explorer.git
cd llama-explorer
pip install -e .
```

## Command Line Interface

Llama Explorer provides a rich command-line interface with various options and modes.

### Basic Command Structure

```bash
llama-explorer [PACKAGE_OR_REPO] [OPTIONS]
```

### Common Options

| Option | Description |
|--------|-------------|
| `-o, --output-dir DIR` | Set output directory (default: llama_output) |
| `-f, --formats FORMATS` | Comma-separated list of output formats (txt,md,json,pdf) |
| `-i, --interactive` | Run in interactive mode with prompts |
| `--legacy` | Run in legacy mode (no master report) |
| `-q, --quiet` | Run with minimal console output |
| `-v, --version` | Show version information |
| `-h, --help` | Show help message |

### Examples

```bash
# Analyze a PyPI package with default settings
llama-explorer requests

# Analyze a GitHub repository
llama-explorer pandas-dev/pandas

# Specify output formats and directory
llama-explorer flask -f txt,md,json -o ./flask_analysis

# Use interactive mode
llama-explorer -i

# Run in legacy mode
llama-explorer numpy --legacy
```

## Analyzing PyPI Packages

Llama Explorer can analyze any package available on PyPI, providing insights into its structure, dependencies, and metadata.

### Basic Package Analysis

```bash
llama-explorer package_name
```

### Analysis Process

When analyzing a PyPI package, Llama Explorer:

1. Fetches package metadata from PyPI
2. Downloads and extracts the package files
3. Analyzes the file structure and categorizes files
4. Identifies dependencies and their relationships
5. Generates comprehensive reports in the requested formats

### Example: Analyzing the Flask Package

```bash
llama-explorer flask -o ./flask_analysis
```

This will:
- Download and analyze the Flask package
- Create a master report with comprehensive information
- Save all reports to the ./flask_analysis directory

## Analyzing GitHub Repositories

Llama Explorer can analyze GitHub repositories to provide insights into their structure, language usage, and contributor statistics.

### Repository Analysis Syntax

```bash
# Using the short format
llama-explorer owner/repo

# Using the full URL
llama-explorer https://github.com/owner/repo
```

### What Gets Analyzed

During GitHub repository analysis, Llama Explorer:

1. Clones the repository locally
2. Analyzes the file structure and categorizes files
3. Detects programming languages and calculates usage statistics
4. Extracts contributor information (top contributors)
5. Analyzes repository metadata (stars, forks, issues)
6. Generates comprehensive reports

### Example: Analyzing the Django Repository

```bash
llama-explorer django/django -o ./django_analysis
```

## Output Formats and Reports

Llama Explorer supports multiple output formats, each with its own advantages.

### Available Formats

- **TXT**: Plain text reports with detailed information
- **MD**: Markdown reports with better formatting and structure
- **JSON**: Machine-readable JSON format for programmatic use
- **PDF**: Professional PDF reports generated from markdown (requires pdfkit)

### Master Report

In the default mode, Llama Explorer generates a comprehensive master report (`master_<name>.txt`) that contains:

- Complete package/repository information
- Detailed dependency analysis
- Comprehensive file structure categorization
- Language breakdown
- Security notes and observations

### Report Location

Reports are saved to the specified output directory (default: `llama_output`), with the following structure:

```
llama_output/
├── master_<name>.txt         # Master report
├── <name>_report.txt         # Text report
├── <name>_report.md          # Markdown report
├── <name>_report.json        # JSON report
└── <name>_report.pdf         # PDF report (if enabled)
```

## Interactive Mode

The interactive mode provides a guided experience with prompts to configure the analysis.

### Starting Interactive Mode

```bash
llama-explorer -i
```

### Interactive Mode Steps

1. **URL/Package Selection**: Enter a PyPI package name, GitHub repository, or URL
2. **Format Selection**: Choose which output formats to generate
3. **Output Directory**: Specify where to save the reports
4. **Processing**: The analysis runs with detailed progress indicators
5. **Results**: Summary of the generated reports

### Example Interactive Session

```
$ llama-explorer -i

🦙 Llama Explorer

Welcome to Llama Explorer!
Analyze PyPI packages or GitHub repositories with style.
Enter a URL, package name, or GitHub repo (owner/repo) below:

Enter URL or package name: requests

✓ Valid PyPI package: requests

[Format selection prompts...]
[Output directory prompts...]

Starting analysis of PyPI package: requests
...
```

## API Server Mode

Llama Explorer can run as an API server, allowing you to integrate it with other applications or services.

### Starting the API Server

```bash
llama-explorer --api --port 8001
```

### API Endpoints

The API server provides several endpoints:

- `POST /analyze/package/{package_name}`: Analyze a PyPI package
- `POST /analyze/github/{owner}/{repo}`: Analyze a GitHub repository
- `GET /jobs/{job_id}`: Get status of a specific analysis job
- `GET /reports/{name}`: List available reports for a package/repository
- `GET /download/report/{name}/{version}/{filename}`: Download a specific report

### API Usage Example (curl)

```bash
# Start an analysis job for a package
curl -X POST "http://localhost:8001/analyze/package/flask" -H "Content-Type: application/json" -d '{"formats": ["txt", "md", "json"]}'

# Start an analysis job for a GitHub repository
curl -X POST "http://localhost:8001/analyze/github/django/django" -H "Content-Type: application/json" -d '{"formats": ["txt", "md", "json"]}'
```

## Python API

Llama Explorer provides a Python API that allows you to integrate its functionality into your own Python applications.

### Automated Mode (Recommended)

```python
import asyncio
from llama_explorer import automated_mode

async def analyze():
    # Analyze a PyPI package
    await automated_mode("requests", output_dir="./output")
    
    # Analyze a GitHub repository
    await automated_mode("django/django", output_dir="./github_output")

asyncio.run(analyze())
```

### Direct Class Usage

```python
import asyncio
from llama_explorer import LlamaExplorer, GitHubExplorer
from llama_explorer.config import Config

async def analyze_with_classes():
    # Analyze PyPI package
    config = Config(
        pypi_url="https://pypi.org/pypi/requests/",
        package_name="requests",
        output_dir="./output",
        output_formats=["txt", "md", "json"],
        include_tests=True
    )
    explorer = LlamaExplorer(config)
    success = await explorer.process_package()
    
    # Analyze GitHub repository
    github_explorer = GitHubExplorer(
        "https://github.com/django/django",
        output_dir="./github_output",
        formats=["txt", "md", "json"]
    )
    success = await github_explorer.process_repository()

asyncio.run(analyze_with_classes())
```

### Processing Reports Programmatically

```python
import json
import asyncio
from llama_explorer import automated_mode

async def analyze_and_process():
    # Run the analysis
    await automated_mode("flask", output_dir="./output")
    
    # Load and process the JSON report
    with open("./output/flask_report.json", "r") as f:
        data = json.load(f)
    
    # Process the data
    print(f"Package name: {data['name']}")
    print(f"Version: {data['version']}")
    print(f"Dependencies: {', '.join(data['dependencies'])}")
    
    # Count files by type
    for category, files in data['files'].items():
        print(f"{category}: {len(files)} files")

asyncio.run(analyze_and_process())
```

## Advanced Configuration

### Environment Variables

- `GITHUB_TOKEN`: GitHub personal access token for higher API rate limits
- `LLAMA_CACHE_DIR`: Custom cache directory location
- `LLAMA_TEMP_DIR`: Custom temporary directory location
- `LLAMA_LOG_LEVEL`: Set logging level (DEBUG, INFO, WARNING, ERROR)

### Custom Scraping Configuration

For enhanced scraping capabilities (with the `scraper` extra dependency), you can configure:

```python
from llama_explorer import Config

config = Config(
    pypi_url="https://pypi.org/pypi/package/",
    package_name="package",
    # Scraper settings
    scraper_settings={
        "browser_type": "chrome",
        "stealth_mode": True,
        "timeout": 60,
        "retry_attempts": 3
    }
)
```

## Troubleshooting

### Missing Dependencies

If you encounter errors about missing modules:

```bash
pip install llama-explorer[all]
```

### GitHub API Rate Limiting

If you encounter GitHub API rate limit errors:

1. Create a personal access token at https://github.com/settings/tokens
2. Set the token as an environment variable:
   ```bash
   export GITHUB_TOKEN=your_token_here
   ```

### PDF Generation Issues

If PDF generation fails:

1. Ensure wkhtmltopdf is installed on your system
2. Install the pdfkit dependency: `pip install pdfkit`
3. Check that your system has the necessary fonts and libraries

### Timeout Errors

For large repositories or packages, you might encounter timeout errors. Try:

```bash
llama-explorer large-repo/name --timeout 300
```

### Cache Clearing

If you encounter issues that might be related to cached data:

```bash
llama-explorer --clear-cache
```

### Verbose Logging

For detailed logging information to help diagnose issues:

```bash
export LLAMA_LOG_LEVEL=DEBUG
llama-explorer package_name
```

---

For more information, please visit the [GitHub repository](https://github.com/your-username/llama-explorer) or open an issue if you encounter any problems. 