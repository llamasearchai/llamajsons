# Contributing to Llama Explorer

Thank you for your interest in contributing to Llama Explorer! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Environment](#development-environment)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Requests](#pull-requests)
- [Issue Reporting](#issue-reporting)
- [Feature Requests](#feature-requests)

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct, which promotes a welcoming and inclusive environment for all contributors.

## Getting Started

1. **Fork the Repository**: Start by forking the [Llama Explorer repository](https://github.com/your-username/llama-explorer).

2. **Clone Your Fork**:
   ```bash
   git clone https://github.com/YOUR-USERNAME/llama-explorer.git
   cd llama-explorer
   ```

3. **Set Up Upstream Remote**:
   ```bash
   git remote add upstream https://github.com/your-username/llama-explorer.git
   ```

4. **Create a Branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Environment

1. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install Development Dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```

3. **Install Pre-commit Hooks** (optional but recommended):
   ```bash
   pip install pre-commit
   pre-commit install
   ```

## Making Changes

1. **Coding Style**: We follow the [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide for Python code. Using tools like Black, isort, and flake8 (included in the pre-commit hooks) can help maintain consistent style.

2. **Commit Messages**: Write clear, concise commit messages that explain the changes made. Use the present tense ("Add feature" not "Added feature") and include the relevant issue number if applicable.

3. **Keep Changes Focused**: Each pull request should address one specific issue or feature. This makes the review process easier and increases the chance of your contribution being merged quickly.

## Testing

1. **Writing Tests**: Add tests for any new features or bug fixes. We use pytest for testing.

2. **Running Tests**:
   ```bash
   pytest
   ```

3. **Test Coverage**:
   ```bash
   pytest --cov=llama_explorer
   ```

## Documentation

1. **Code Documentation**: Document your code using docstrings following the [Google style guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).

2. **User Documentation**: Update the relevant documentation in the `docs/` directory for any user-facing changes.

3. **README Updates**: Keep the README.md up-to-date with any significant changes, especially those affecting installation or usage.

## Pull Requests

1. **Create a Pull Request**: When you're ready to submit your changes, push your branch to your fork and create a pull request against the main repository.

2. **PR Description**: Include a detailed description of the changes, the problem they solve, and any relevant information that might help reviewers.

3. **Continuous Integration**: Make sure all CI checks pass. This includes tests, code style checks, and type checking.

4. **Review Process**: The maintainers will review your PR and might request changes. Be responsive to feedback and make necessary adjustments.

5. **Merging**: Once your PR is approved, a maintainer will merge it into the main branch.

## Issue Reporting

1. **Search First**: Before reporting an issue, check if it already exists.

2. **Be Specific**: When opening a new issue, provide as much information as possible, including:
   - Steps to reproduce the issue
   - Expected behavior
   - Actual behavior
   - Environment information (OS, Python version, etc.)
   - Screenshots or code snippets, if applicable

3. **Issue Templates**: Use the provided issue templates when available.

## Feature Requests

1. **Feature Discussion**: For significant features, it's best to open an issue first to discuss the idea before implementing it.

2. **Implementation Plan**: When proposing a new feature, include a clear implementation plan and explain how it benefits the project.

3. **Backward Compatibility**: Consider backward compatibility when designing new features.

---

Thank you for contributing to Llama Explorer! Your help is essential for making this tool better for everyone. 