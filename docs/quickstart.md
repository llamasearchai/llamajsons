# 🦙 Llama Explorer - Quickstart Guide

This guide will help you get started with Llama Explorer quickly and efficiently.

## Installation

First, install Llama Explorer from PyPI:

```bash
pip install llama-explorer
```

For full functionality, install with all optional dependencies:

```bash
pip install llama-explorer[all]
```

## Basic Usage

### Analyze a Package

To analyze a Python package, simply run:

```bash
llama-explorer requests
```

This will:
1. Download and analyze the "requests" package
2. Create a comprehensive master report
3. Save all outputs to the "llama_output" directory

### Analyze a GitHub Repository

To analyze a GitHub repository, use either the full URL or the short format:

```bash
llama-explorer pandas-dev/pandas
# OR
llama-explorer https://github.com/pandas-dev/pandas
```

### Interactive Mode

For a guided experience with prompts, use interactive mode:

```bash
llama-explorer -i
```

This mode will guide you through:
- Selecting what to analyze (PyPI package or GitHub repository)
- Choosing output formats
- Setting the output directory

## Understanding the Output

After running Llama Explorer, you'll find these files in your output directory:

1. **Master Report** (`master_<name>.txt`): A comprehensive overview containing all information
2. **Standard Reports**: Individual reports in various formats (.txt, .md, .json, etc.)

The master report contains:
- Package/repository information
- Dependency analysis
- File structure categorization
- Language breakdown
- Security notes

## Advanced Usage

### Custom Output Directory

```bash
llama-explorer requests -o ./my_reports
```

### Quiet Mode (Less Console Output)

```bash
llama-explorer requests -q
```

### Legacy Mode (No Master Report)

```bash
llama-explorer requests --legacy
```

### Running as API Server

```bash
llama-explorer --api --port 8001
```

## Using the Python API

```python
import asyncio
from llama_explorer import automated_mode

async def analyze():
    # Analyze a package or repository
    await automated_mode("requests")
    
    # Specify an output directory
    await automated_mode("pandas-dev/pandas", output_dir="./my_reports")

# Run the async function
asyncio.run(analyze())
```

## Common Issues and Troubleshooting

### Missing Dependencies

If you see an error about missing dependencies, install the needed packages:

```bash
pip install llama-explorer[all]
```

### Permission Errors

If you encounter permission errors when creating output directories, try:
- Using a different output directory with `-o`
- Running with appropriate permissions

### Connection Issues

If you experience connection issues when analyzing GitHub repositories:
- Check your internet connection
- Ensure you're not being rate-limited by GitHub
- Consider using a personal access token (set GITHUB_TOKEN environment variable)

### Large Repositories

For very large repositories, the analysis might take some time. Use quiet mode (`-q`) to reduce console output and be patient.

## Next Steps

- Check out the full documentation for more detailed information
- Try analyzing your own projects
- Explore the Python API for integration in your own tools

Happy exploring with Llama Explorer! 🦙 