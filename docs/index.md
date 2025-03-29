# 🦙 Llama Explorer Documentation

Welcome to the Llama Explorer documentation. Llama Explorer is a powerful tool that lets you explore and analyze Python packages from PyPI and repositories from GitHub.

## Documentation Pages

- [Quick Start Guide](quickstart.md) - Get started quickly with Llama Explorer
- [Comprehensive Usage Guide](usage-guide.md) - Detailed information on using all features
- [API Reference](api-reference.md) - Complete reference for Python and REST API

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
```

## Basic Usage Examples

### Analyze a Package

```bash
llama-explorer requests
```

### Analyze a GitHub Repository

```bash
llama-explorer pandas-dev/pandas
```

### Interactive Mode

```bash
llama-explorer -i
```

## Contributing

We welcome contributions to Llama Explorer! Please see our [Contributing Guide](contributing.md) for details on how to get involved.

## Support

If you encounter any issues or have questions, please:

1. Check the [Troubleshooting](usage-guide.md#troubleshooting) section
2. Search for existing issues in our [GitHub repository](https://github.com/your-username/llama-explorer/issues)
3. Open a new issue if needed

## License

Llama Explorer is released under the [MIT License](https://opensource.org/licenses/MIT).

---

Happy exploring! 🦙 