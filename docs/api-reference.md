# 🦙 Llama Explorer - API Reference

This document provides detailed information about the Python API and REST API endpoints available in Llama Explorer.

## Python API Reference

### Core Functions

#### `automated_mode`

```python
async def automated_mode(url: str, output_dir: str = "llama_output") -> None
```

Analyzes a package or repository automatically with sensible defaults.

**Parameters:**
- `url` (str): The package name, GitHub repository, or URL to analyze
- `output_dir` (str, optional): Directory to save reports (default: "llama_output")

**Returns:**
- None

**Example:**
```python
import asyncio
from llama_explorer import automated_mode

async def main():
    await automated_mode("requests")
    
asyncio.run(main())
```

#### `interactive_mode`

```python
async def interactive_mode() -> None
```

Starts an interactive session with prompts for configuration.

**Parameters:**
- None

**Returns:**
- None

**Example:**
```python
import asyncio
from llama_explorer import interactive_mode

asyncio.run(interactive_mode())
```

### Core Classes

#### `Config`

```python
class Config(BaseModel):
    """Configuration for the Llama Explorer"""
    pypi_url: str
    package_name: str
    output_dir: str
    output_formats: List[str] = Field(default_factory=lambda: ["txt", "md", "json"])
    include_tests: bool = False
    include_metadata: bool = True
    include_toc: bool = True
    temp_dir: Optional[str] = None
    parallel_processing: bool = True
    max_retries: int = 3
    timeout: int = 30
    verbose: bool = False
```

Configuration class used to initialize explorers.

**Fields:**
- `pypi_url` (str): URL to the PyPI package
- `package_name` (str): Name of the package to analyze
- `output_dir` (str): Directory to save reports
- `output_formats` (List[str]): Formats to generate (txt, md, json, pdf)
- `include_tests` (bool): Whether to include test files in the analysis
- `include_metadata` (bool): Whether to include metadata in reports
- `include_toc` (bool): Whether to include a table of contents in reports
- `temp_dir` (Optional[str]): Directory for temporary files
- `parallel_processing` (bool): Whether to use parallel processing
- `max_retries` (int): Maximum number of retry attempts for network operations
- `timeout` (int): Timeout for network operations in seconds
- `verbose` (bool): Whether to enable verbose output

#### `LlamaExplorer`

```python
class LlamaExplorer:
    """Class for exploring and analyzing PyPI packages"""
    
    def __init__(self, config: Config): ...
    
    async def fetch_package_info(self): ...
    
    async def download_package(self): ...
    
    async def categorize_files(self, package_dir: Path): ...
    
    async def generate_reports(self): ...
    
    async def generate_text_report(self, report_dir: Path): ...
    
    async def generate_markdown_report(self, report_dir: Path): ...
    
    async def generate_json_report(self, report_dir: Path): ...
    
    async def create_pdf(self, report_dir: Path): ...
    
    async def process_package(self): ...
    
    async def cleanup(self): ...
```

Main class for analyzing PyPI packages.

**Constructor:**
- `config` (Config): Configuration object

**Key Methods:**
- `fetch_package_info()`: Retrieves package information from PyPI
- `download_package()`: Downloads and extracts the package
- `categorize_files()`: Categorizes files by type
- `generate_reports()`: Generates reports in all requested formats
- `process_package()`: Main method to process the entire package
- `cleanup()`: Cleans up temporary files

**Example:**
```python
import asyncio
from llama_explorer import LlamaExplorer, Config

async def main():
    config = Config(
        pypi_url="https://pypi.org/pypi/requests/",
        package_name="requests",
        output_dir="./output"
    )
    explorer = LlamaExplorer(config)
    success = await explorer.process_package()
    print(f"Analysis {'succeeded' if success else 'failed'}")
    
asyncio.run(main())
```

#### `GitHubExplorer`

```python
class GitHubExplorer:
    """Class for exploring and analyzing GitHub repositories"""
    
    def __init__(self, repo_url: str, base_output_dir: str = "llama_output", formats=None): ...
    
    async def fetch_repo_info(self): ...
    
    async def clone_repository(self): ...
    
    async def analyze_repository(self, repo_dir): ...
    
    async def _fetch_contributors(self): ...
    
    async def generate_reports(self): ...
    
    async def generate_text_report(self): ...
    
    async def generate_markdown_report(self): ...
    
    async def generate_json_report(self): ...
    
    async def create_pdf(self): ...
    
    async def cleanup(self): ...
    
    async def process_repository(self): ...
```

Main class for analyzing GitHub repositories.

**Constructor:**
- `repo_url` (str): GitHub repository URL or owner/repo format
- `base_output_dir` (str, optional): Directory to save reports (default: "llama_output")
- `formats` (List[str], optional): Output formats (default: ["txt", "md", "json"])

**Key Methods:**
- `fetch_repo_info()`: Retrieves repository information from GitHub API
- `clone_repository()`: Clones the repository locally
- `analyze_repository()`: Analyzes the repository structure
- `generate_reports()`: Generates reports in all requested formats
- `process_repository()`: Main method to process the entire repository
- `cleanup()`: Cleans up temporary files

**Example:**
```python
import asyncio
from llama_explorer import GitHubExplorer

async def main():
    explorer = GitHubExplorer(
        repo_url="https://github.com/django/django",
        base_output_dir="./output",
        formats=["txt", "md", "json"]
    )
    success = await explorer.process_repository()
    print(f"Analysis {'succeeded' if success else 'failed'}")
    
asyncio.run(main())
```

### Data Models

#### `PackageInfo`

```python
class PackageInfo(BaseModel):
    """Information about a PyPI package"""
    name: str
    version: str
    summary: Optional[str] = None
    description: Optional[str] = None
    author: Optional[str] = None
    author_email: Optional[str] = None
    license: Optional[str] = None
    project_url: Optional[str] = None
    homepage: Optional[str] = None
    requires_python: Optional[str] = None
    dependencies: List[str] = Field(default_factory=list)
    files: Dict[str, List[str]] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
```

Model representing PyPI package information.

#### `GitHubInfo`

```python
class GitHubInfo(BaseModel):
    """Information about a GitHub repository"""
    name: str
    full_name: str
    description: Optional[str] = None
    owner: str
    stars: int = 0
    forks: int = 0
    language: Optional[str] = None
    topics: List[str] = Field(default_factory=list)
    license: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    clone_url: str
    files: Dict[str, List[str]] = Field(default_factory=dict)
    languages: Dict[str, int] = Field(default_factory=dict)
    contributors: List[Dict[str, Any]] = Field(default_factory=list)
    open_issues: int = 0
    readme_content: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
```

Model representing GitHub repository information.

### Utility Functions

#### `check_dependencies()`

```python
def check_dependencies() -> Dict[str, Any]
```

Checks if all required dependencies are installed.

**Returns:**
- Dictionary with `ok` (bool), `missing` (List[str]), and `optional_missing` (List[str])

#### `validate_url(url: str)`

```python
def validate_url(url: str) -> Tuple[bool, str, Optional[Dict[str, Any]]]
```

Validates a URL and determines its type.

**Parameters:**
- `url` (str): The URL to validate

**Returns:**
- Tuple of (is_valid, type, metadata)
  - `is_valid` (bool): Whether the URL is valid
  - `type` (str): URL type ("github", "github_short", "pypi", "pypi_name", "pypi_user", "unknown")
  - `metadata` (Dict[str, Any] or None): Extracted metadata

## REST API Reference

When running in API server mode, Llama Explorer provides the following REST endpoints:

### Package Analysis

#### Analyze a PyPI Package

```
POST /analyze/package/{package_name}
```

**Path Parameters:**
- `package_name` (str): Name of the PyPI package to analyze

**Request Body:**
```json
{
  "formats": ["txt", "md", "json"],
  "include_tests": false,
  "output_dir": "llama_output"
}
```

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "package": "requests",
  "created_at": "2023-01-01T12:00:00Z"
}
```

### GitHub Repository Analysis

#### Analyze a GitHub Repository

```
POST /analyze/github/{owner}/{repo}
```

**Path Parameters:**
- `owner` (str): GitHub repository owner
- `repo` (str): GitHub repository name

**Request Body:**
```json
{
  "formats": ["txt", "md", "json", "pdf"],
  "output_dir": "llama_output"
}
```

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "package": "github/owner/repo",
  "created_at": "2023-01-01T12:00:00Z"
}
```

### Job Management

#### Get Job Status

```
GET /jobs/{job_id}
```

**Path Parameters:**
- `job_id` (str): ID of the job to check

**Response:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "type": "package_analysis",
  "target": "requests",
  "status": "completed",
  "created_at": "2023-01-01T12:00:00Z",
  "updated_at": "2023-01-01T12:05:00Z",
  "result": {
    "package": "requests",
    "reports": ["txt", "md", "json"],
    "output_dir": "/path/to/output"
  },
  "error": null
}
```

#### List All Jobs

```
GET /jobs
```

**Query Parameters:**
- `status` (optional): Filter by job status ("pending", "running", "completed", "failed")

**Response:**
```json
[
  {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "type": "package_analysis",
    "target": "requests",
    "status": "completed",
    "created_at": "2023-01-01T12:00:00Z",
    "updated_at": "2023-01-01T12:05:00Z"
  },
  {
    "id": "550e8400-e29b-41d4-a716-446655440001",
    "type": "github_analysis",
    "target": "owner/repo",
    "status": "running",
    "created_at": "2023-01-01T12:10:00Z",
    "updated_at": "2023-01-01T12:10:30Z"
  }
]
```

#### Cancel a Job

```
DELETE /jobs/{job_id}
```

**Path Parameters:**
- `job_id` (str): ID of the job to cancel

**Response:**
```json
{
  "message": "Job cancelled successfully"
}
```

### Reports

#### List Reports for a Package

```
GET /reports/{package_name}
```

**Path Parameters:**
- `package_name` (str): Name of the package/repository

**Response:**
```json
{
  "package": "requests",
  "reports": [
    {
      "version": "2.28.1",
      "reports": [
        {
          "type": "TXT",
          "path": "/download/report/requests/2.28.1/requests_report.txt"
        },
        {
          "type": "MD",
          "path": "/download/report/requests/2.28.1/requests_report.md"
        },
        {
          "type": "JSON",
          "path": "/download/report/requests/2.28.1/requests_report.json"
        }
      ]
    }
  ]
}
```

#### Download a Report

```
GET /download/report/{package_name}/{version}/{filename}
```

**Path Parameters:**
- `package_name` (str): Name of the package/repository
- `version` (str): Version or timestamp of the report
- `filename` (str): Filename of the report

**Response:**
The requested report file.

### System Management

#### Get Cache Information

```
GET /cache/info
```

**Response:**
```json
{
  "cache_size_bytes": 1048576,
  "cache_files": 25,
  "cache_dir": "/path/to/cache"
}
```

#### Clear Cache

```
POST /cache/clear
```

**Response:**
```json
{
  "message": "Cache cleared successfully"
}
```

#### Get System Statistics

```
GET /stats
```

**Response:**
```json
{
  "total_jobs": 150,
  "active_jobs": 5,
  "completed_jobs": 130,
  "failed_jobs": 15,
  "reports_dir_size": 52428800,
  "cache_dir_size": 1048576,
  "temp_dir_size": 524288
}
```

#### Clean Up Old Reports and Jobs

```
POST /cleanup
```

**Query Parameters:**
- `max_age_hours` (int, optional): Maximum age of jobs to keep in hours (default: 24)
- `max_age_days` (int, optional): Maximum age of reports to keep in days (default: 30)

**Response:**
```json
{
  "message": "Cleanup completed successfully"
}
```

## Error Handling

All endpoints return appropriate HTTP status codes:

- 200: Successful operation
- 400: Bad request (invalid parameters)
- 404: Resource not found
- 500: Internal server error

Error responses have the following format:

```json
{
  "detail": "Error message describing the issue"
}
``` 