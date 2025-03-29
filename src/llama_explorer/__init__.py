"""
Llama Explorer - Extract, analyze and explore PyPI packages and GitHub repositories
"""

__version__ = "1.0.0"

from .explorer import (
    LlamaExplorer, 
    GitHubExplorer, 
    PyPIProfileExplorer,
    check_dependencies,
    Config,
    GitHubInfo,
    PackageInfo,
    automated_mode,
    interactive_mode,
)

# For backward compatibility
from .explorer import main, run_api

__all__ = [
    "main",
    "LlamaExplorer",
    "GitHubExplorer", 
    "check_dependencies",
    "interactive_mode",
    "automated_mode"
] 
