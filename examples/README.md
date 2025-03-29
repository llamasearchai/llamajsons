# Llama Explorer - Usage Examples

This directory contains examples demonstrating how to use Llama Explorer in different ways.

## Available Examples

### 1. API Usage (`api_usage.py`)

This example shows how to use the Llama Explorer programmatically through its Python API.
It demonstrates how to analyze both a PyPI package and a GitHub repository using the `LlamaExplorer` and `GitHubExplorer` classes.

To run this example:
```bash
cd examples
python api_usage.py
```

### 2. Main Script (`../run_explorer.py`)

The main script in the root directory provides a complete command-line interface for Llama Explorer.
It supports various usage modes, including automatic analysis, interactive mode, and direct command-line arguments.

To run the main script:
```bash
# Run in interactive mode
python ../run_explorer.py -i

# Analyze a package
python ../run_explorer.py requests

# Analyze a GitHub repository
python ../run_explorer.py psf/requests
```

## Creating Your Own Scripts

You can create your own scripts using the Llama Explorer API by following these steps:

1. Import the necessary classes:
```python
from llama_explorer import LlamaExplorer, GitHubExplorer, Config
```

2. Configure and create an explorer:
```python
# For PyPI packages
config = Config(
    pypi_url="https://pypi.org/pypi/package-name/",
    package_name="package-name",
    output_dir="./output",
    output_formats=["txt", "md", "json"]
)
explorer = LlamaExplorer(config)

# For GitHub repositories
explorer = GitHubExplorer(
    repo_url="https://github.com/owner/repo",
    base_output_dir="./output",
    formats=["txt", "md", "json"]
)
```

3. Process the package or repository:
```python
# For PyPI packages
success = await explorer.process_package()

# For GitHub repositories
success = await explorer.process_repository()
```

For more detailed information, refer to the documentation in the `docs/` directory. 