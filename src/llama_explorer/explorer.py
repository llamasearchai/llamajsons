#!/usr/bin/env python3
"""
PyPI Llama Explorer - Extract and analyze PyPI packages with style
"""
import asyncio
import json
import logging
import os
import pathlib
from pathlib import Path
import shutil
import tempfile
import re
import zipfile
import tarfile
import sys
import subprocess
import random
import uuid
import aiofiles
import git
import requests
import importlib.util
from typing import List, Dict, Any, Optional, Union
from urllib.parse import urlparse, parse_qs
from datetime import datetime, timezone, timedelta, date
import aiohttp
from bs4 import BeautifulSoup
import markdown2
import pdfkit
import socket
import urllib.parse
import psutil
import inspect

# FastAPI imports
from fastapi import FastAPI, HTTPException, BackgroundTasks, status, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import uvicorn

# Rich UI components
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.align import Align
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from rich.prompt import Prompt, Confirm
from rich.layout import Layout
from rich.live import Live
from rich.markdown import Markdown

# Initialize FastAPI app
app = FastAPI(
    title="PyPI Llama Explorer API",
    description="API for exploring and analyzing PyPI packages",
    version="1.0.0"
)

# Initialize console for rich output
console = Console()

# Define the OutputManager class first, since it needs to be initialized early
class OutputManager:
    """Manages output files, directories, and caching"""
    
    def __init__(self, base_dir: str = "llama_output"):
        self.base_dir = Path(base_dir)
        self.reports_dir = self.base_dir / "reports"
        self.cache_dir = self.base_dir / "cache"
        self.temp_dir = self.base_dir / "temp"
        self.logs_dir = self.base_dir / "logs"
        self.metadata_dir = self.base_dir / "metadata"
        self._setup_directories()
        
        # Configure logging
        self._setup_logging()
        self.logger = logging.getLogger("llama_explorer")
    
    def _setup_directories(self):
        """Create necessary directories if they don't exist"""
        for directory in [self.reports_dir, self.cache_dir, self.temp_dir, 
                         self.logs_dir, self.metadata_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self):
        """Configure logging with both file and console handlers"""
        log_file = self.logs_dir / "llama_explorer.log"
        
        # Create formatters
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_formatter = logging.Formatter(
            '%(levelname)s: %(message)s'
        )
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
    
    def get_report_dir(self, name: str, type_: str = "package") -> Path:
        """Get the report directory for a package or repository"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = self.reports_dir / type_ / name / timestamp
        report_dir.mkdir(parents=True, exist_ok=True)
        return report_dir
    
    def get_cache_path(self, key: str) -> Path:
        """Get the cache file path for a given key"""
        return self.cache_dir / f"{key}.json"
    
    def cache_exists(self, key: str) -> bool:
        """Check if cache exists for a given key"""
        return self.get_cache_path(key).exists()
    
    async def save_to_cache(self, key: str, data: Any):
        """Save data to cache"""
        cache_path = self.get_cache_path(key)
        async with aiofiles.open(cache_path, 'w') as f:
            await f.write(json.dumps(data, default=str))
    
    async def load_from_cache(self, key: str) -> Optional[Any]:
        """Load data from cache"""
        cache_path = self.get_cache_path(key)
        try:
            if cache_path.exists():
                async with aiofiles.open(cache_path) as f:
                    content = await f.read()
                    return json.loads(content)
        except Exception as e:
            self.logger.warning(f"Failed to load cache for {key}: {e}")
        return None
    
    def clear_cache(self):
        """Clear all cached data"""
        shutil.rmtree(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    async def save_metadata(self, name: str, type_: str, metadata: Dict[str, Any]):
        """Save metadata for a package or repository"""
        metadata_file = self.metadata_dir / type_ / f"{name}.json"
        metadata_file.parent.mkdir(parents=True, exist_ok=True)
        
        metadata["last_updated"] = datetime.now(timezone.utc).isoformat()
        async with aiofiles.open(metadata_file, 'w') as f:
            await f.write(json.dumps(metadata, default=str, indent=2))
    
    async def get_metadata(self, name: str, type_: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a package or repository"""
        metadata_file = self.metadata_dir / type_ / f"{name}.json"
        try:
            if metadata_file.exists():
                async with aiofiles.open(metadata_file) as f:
                    content = await f.read()
                    return json.loads(content)
        except Exception as e:
            self.logger.warning(f"Failed to load metadata for {name}: {e}")
        return None
    
    def cleanup_old_reports(self, max_age_days: int = 30):
        """Clean up reports older than specified days"""
        cutoff = datetime.now() - timedelta(days=max_age_days)
        
        for type_dir in self.reports_dir.iterdir():
            if not type_dir.is_dir():
                continue
            
            for name_dir in type_dir.iterdir():
                if not name_dir.is_dir():
                    continue
                
                for timestamp_dir in name_dir.iterdir():
                    if not timestamp_dir.is_dir():
                        continue
                    
                    try:
                        timestamp = datetime.strptime(
                            timestamp_dir.name,
                            "%Y%m%d_%H%M%S"
                        )
                        if timestamp < cutoff:
                            shutil.rmtree(timestamp_dir)
                            self.logger.info(
                                f"Cleaned up old report: {timestamp_dir}"
                            )
                    except ValueError:
                        continue

# Initialize global output manager
output_manager = OutputManager()

# Class UIManager for UI components
class UIManager:
    """Manages user interface components and styling"""
    
    def __init__(self):
        self.console = Console()
        self.logger = logging.getLogger("llama_explorer.ui")
        self._setup_styles()
    
    def _setup_styles(self):
        """Setup custom styles for the UI"""
        self.styles = {
            "title": "bold magenta",
            "subtitle": "dim cyan",
            "success": "bold green",
            "error": "bold red",
            "warning": "bold yellow",
            "info": "bold blue",
            "dim": "dim",
            "highlight": "bold cyan",
            "url": "underline blue",
            "command": "bold yellow",
            "path": "italic cyan",
            "progress": "bold green",
        }
    
    def show_banner(self):
        """Display the application banner"""
        banner = Panel(
            Align.center(LLAMA_ASCII + "\n[bold magenta]PyPI Llama Explorer[/bold magenta]"),
            border_style="bright_magenta",
            padding=(1, 2)
        )
        self.console.print(banner)
        self.console.print(
            "[dim]Extract and analyze PyPI packages with style[/dim]\n",
            justify="center"
        )
    
    def show_help(self):
        """Display help information"""
        help_text = """
[bold magenta]PyPI Llama Explorer - Commands[/bold magenta]

[cyan]Basic Usage:[/cyan]
- Enter a package name: [green]requests[/green]
- Enter a PyPI URL: [green]https://pypi.org/project/requests/[/green]
- Enter a user profile: [green]https://pypi.org/user/username/[/green]
- Enter a GitHub repository: [green]https://github.com/owner/repo[/green]

[cyan]Special Commands:[/cyan]
- [green]help[/green]: Show this help message
- [green]exit[/green]: Exit the program
- [green]clear[/green]: Clear the screen
- [green]config[/green]: Show current configuration
- [green]set format <formats>[/green]: Set output formats (txt,md,json,pdf)
- [green]set output <dir>[/green]: Set output directory
- [green]toggle tests[/green]: Toggle inclusion of tests
- [green]toggle verbose[/green]: Toggle verbose output
- [green]jobs[/green]: List active jobs
- [green]cancel <job_id>[/green]: Cancel a running job
- [green]cleanup[/green]: Clean up old reports and jobs

[cyan]Output Formats:[/cyan]
- TEXT: Detailed information in plain text
- MARKDOWN: Rich formatted documentation
- JSON: Machine-readable data
- PDF: Professional documentation (requires pdfkit)

[cyan]Features:[/cyan]
- Package analysis with dependency tracking
- GitHub repository analysis
- PyPI user profile exploration
- Parallel processing for faster results
- Caching for improved performance
- Comprehensive progress tracking
- Detailed error reporting
- Multiple output formats
- Background job processing
"""
        self.console.print(Panel(help_text, title="Help", border_style="cyan"))
    
    def show_config(self, config):
        """Display current configuration"""
        table = Table(title="Current Configuration", border_style="blue")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")
        
        for field, value in config.__dict__.items():
            if isinstance(value, list):
                value = ", ".join(value)
            table.add_row(field, str(value))
        
        self.console.print(table)
    
    def show_jobs(self, jobs):
        """Display job status information"""
        if not jobs:
            self.console.print("[yellow]No jobs found[/yellow]")
            return
        
        table = Table(title="Jobs", border_style="blue")
        table.add_column("ID", style="dim")
        table.add_column("Type", style="cyan")
        table.add_column("Target", style="blue")
        table.add_column("Status", style="green")
        table.add_column("Progress", style="yellow")
        table.add_column("Created", style="dim")
        
        for job in jobs:
            status_style = {
                "pending": "yellow",
                "running": "blue",
                "completed": "green",
                "failed": "red",
                "cancelled": "red"
            }.get(job["status"], "white")
            
            table.add_row(
                job["id"][:8],
                job["type"],
                job["target"],
                f"[{status_style}]{job['status']}[/{status_style}]",
                f"{job.get('progress', 0):.1f}%",
                job["created_at"].strftime("%Y-%m-%d %H:%M:%S")
            )
        
        self.console.print(table)
    
    def create_progress(self) -> Progress:
        """Create a new progress bar"""
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=self.console
        )
    
    def error(self, message: str):
        """Display an error message"""
        self.console.print(f"[{self.styles['error']}]Error: {message}[/{self.styles['error']}]")
        self.logger.error(message)
    
    def warning(self, message: str):
        """Display a warning message"""
        self.console.print(f"[{self.styles['warning']}]Warning: {message}[/{self.styles['warning']}]")
        self.logger.warning(message)
    
    def success(self, message: str):
        """Display a success message"""
        self.console.print(f"[{self.styles['success']}]✓ {message}[/{self.styles['success']}]")
        self.logger.info(message)
    
    def info(self, message: str):
        """Display an info message"""
        self.console.print(f"[{self.styles['info']}]{message}[/{self.styles['info']}]")
        self.logger.info(message)
    
    def prompt(self, message: str, default: str = "") -> str:
        """Display a prompt and get user input"""
        return Prompt.ask(message, default=default)
    
    def confirm(self, message: str, default: bool = True) -> bool:
        """Display a confirmation prompt"""
        return Confirm.ask(message, default=default)

# Initialize global UI manager
ui_manager = UIManager()

# Define the JobManager class
class JobManager:
    """Manages background jobs and their status"""
    
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.active_jobs: Dict[str, asyncio.Task] = {}
        self.logger = logging.getLogger("llama_explorer.jobs")
    
    def create_job(self, job_type: str, target: str) -> str:
        """Create a new job and return its ID"""
        job_id = str(uuid.uuid4())
        job = {
            "id": job_id,
            "type": job_type,
            "target": target,
            "status": "pending",
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
            "result": None,
            "error": None,
            "progress": 0.0
        }
        self.jobs[job_id] = job
        return job_id
    
    def update_job(self, job_id: str, status: str, result: Optional[Dict] = None, 
                  error: Optional[str] = None, progress: Optional[float] = None):
        """Update job status and details"""
        if job_id not in self.jobs:
            raise ValueError(f"Job {job_id} not found")
        
        job = self.jobs[job_id]
        job["status"] = status
        job["updated_at"] = datetime.now(timezone.utc)
        
        if result is not None:
            job["result"] = result
        if error is not None:
            job["error"] = error
        if progress is not None:
            job["progress"] = progress
        
        self.logger.info(
            f"Job {job_id} ({job['type']}) updated: {status}",
            extra={"job_id": job_id, "status": status}
        )
    
    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job details"""
        return self.jobs.get(job_id)
    
    def list_jobs(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all jobs, optionally filtered by status"""
        jobs = list(self.jobs.values())
        if status:
            jobs = [job for job in jobs if job["status"] == status]
        return sorted(jobs, key=lambda x: x["created_at"], reverse=True)
    
    def cleanup_old_jobs(self, max_age_hours: int = 24):
        """Clean up old completed or failed jobs"""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        to_remove = []
        
        for job_id, job in self.jobs.items():
            if job["status"] in ["completed", "failed"]:
                if job["updated_at"] < cutoff:
                    to_remove.append(job_id)
        
        for job_id in to_remove:
            del self.jobs[job_id]
            self.logger.info(f"Cleaned up old job: {job_id}")
    
    async def cancel_job(self, job_id: str):
        """Cancel a running job"""
        if job_id not in self.jobs:
            raise ValueError(f"Job {job_id} not found")
        
        if job_id in self.active_jobs:
            task = self.active_jobs[job_id]
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            del self.active_jobs[job_id]
        
        self.update_job(job_id, "cancelled", error="Job cancelled by user")
    
    def register_task(self, job_id: str, task: asyncio.Task):
        """Register an active task for a job"""
        self.active_jobs[job_id] = task
        
        def cleanup_callback(task):
            if job_id in self.active_jobs:
                del self.active_jobs[job_id]
        
        task.add_done_callback(cleanup_callback)

# Initialize global job manager
job_manager = JobManager()

# API Models
class AnalysisRequest(BaseModel):
    """Request model for package analysis"""
    formats: List[str] = Field(default=['txt', 'md', 'json'])
    include_tests: bool = Field(default=False)
    output_dir: str = Field(default="llama_output")

class GitHubAnalysisRequest(BaseModel):
    """Request model for GitHub repository analysis"""
    formats: List[str] = Field(default=['txt', 'md', 'json'])
    output_dir: str = Field(default="llama_output")

class AnalysisResponse(BaseModel):
    """Response model for analysis requests"""
    job_id: str
    status: str
    package: str
    created_at: datetime

class JobStatus(BaseModel):
    """Job status response model"""
    id: str
    type: str
    target: str
    status: str
    created_at: datetime
    updated_at: datetime
    result: Optional[Dict[str, Any]]
    error: Optional[str]

def run_api(host="0.0.0.0", port=8001, max_port_attempts=10):
    """Run the FastAPI application with port conflict handling."""
    import uvicorn
    import socket
    
    # Try up to max_port_attempts ports, starting from the specified port
    original_port = port
    for attempt in range(max_port_attempts):
        try:
            # Test if port is available
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((host, port))
            
            # Port is available, run the server
            console.print(f"[green]Starting API server on http://{host}:{port}[/green]")
            config = uvicorn.Config(app, host=host, port=port)
            server = uvicorn.Server(config)
            return server.serve()
        
        except OSError as e:
            if e.errno == 48 or e.errno == 98:  # Address already in use
                console.print(f"[yellow]Port {port} is already in use, trying {port+1}...[/yellow]")
                port += 1
            else:
                console.print(f"[bold red]Error starting API server: {str(e)}[/bold red]")
                raise
    
    console.print(f"[bold red]Failed to find an available port after {max_port_attempts} attempts, starting from {original_port}[/bold red]")
    console.print("[bold yellow]Please specify a different port using --port option[/bold yellow]")
    return None

def check_dependencies():
    """Check if all required dependencies are installed"""
    missing = []
    
    # Check core dependencies
    for package in ["aiohttp", "fastapi", "rich", "pydantic", "bs4", "markdown2"]:
        if importlib.util.find_spec(package) is None:
            missing.append(package)
    
    # Check optional dependencies
    optional = []
    if importlib.util.find_spec("pdfkit") is None:
        optional.append("pdfkit")
    if importlib.util.find_spec("gitpython") is None:
        optional.append("gitpython")
    
    return {
        "missing": missing,
        "optional_missing": optional,
        "ok": len(missing) == 0
    }

def get_random_user_agent():
    """Get a random user agent string"""
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Safari/605.1.15",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
        "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1"
    ]
    return random.choice(user_agents)

# Rich UI components
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.align import Align
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from rich.prompt import Prompt, Confirm
from rich.layout import Layout
from rich.live import Live
from rich.markdown import Markdown
from pydantic import BaseModel, Field

# Llama ASCII Art
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

# Define these variables
PDFKIT_AVAILABLE = False
CLOUDSCRAPER_AVAILABLE = False

# Common user agents for rotation
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/115.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Edge/120.0.0.0",
]

def get_random_user_agent():
    """Return a random user agent from the list"""
    return random.choice(USER_AGENTS)

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
    languages: Dict[str, int] = Field(default_factory=dict)  # Map of language to byte count
    contributors: List[Dict[str, Any]] = Field(default_factory=list)
    open_issues: int = 0
    readme_content: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class LlamaExplorer:
    """Class for exploring and analyzing PyPI packages"""
    
    def __init__(self, config: Config):
        self.config = config
        if self.config.temp_dir is None:
            self.config.temp_dir = tempfile.mkdtemp(prefix="llama_explorer_")
        else:
            os.makedirs(self.config.temp_dir, exist_ok=True)
        
        self.temp_dir = Path(self.config.temp_dir)
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.package_info = None
        self.ui = ui_manager
    
    async def fetch_package_info(self):
        """Fetch package information from PyPI"""
        try:
            self.ui.info(f"Fetching package information for {self.config.package_name}...")
            
            # Use cached data if available
            cache_key = f"package_info_{self.config.package_name}"
            cached_data = await output_manager.load_from_cache(cache_key)
            if cached_data:
                self.ui.success(f"Using cached package information for {self.config.package_name}")
                self.package_info = PackageInfo(**cached_data)
                return self.package_info
            
            async with aiohttp.ClientSession() as session:
                response = await session.get(
                    f"{self.config.pypi_url.rstrip('/')}/json",
                    timeout=self.config.timeout
                )
                
                if response.status != 200:
                    self.ui.error(f"Failed to fetch package information. Status code: {response.status}")
                    return None
                
                data = await response.json()
                
                info = data.get('info', {})
                version = info.get('version', '')
                releases = data.get('releases', {})
                latest_release = releases.get(version, [{}])[0] if version in releases else {}
                
                # Extract dependencies
                requires_dist = info.get('requires_dist', [])
                dependencies = []
                
                if requires_dist:
                    for req in requires_dist:
                        if req and ';' not in req:  # Skip conditional dependencies
                            package = req.split('>=')[0].split('==')[0].split('<')[0].split('[')[0].strip()
                            dependencies.append(package)
                
                # Prepare package info
                self.package_info = PackageInfo(
                    name=self.config.package_name,
                    version=version,
                    summary=info.get('summary'),
                    description=info.get('description'),
                    author=info.get('author'),
                    author_email=info.get('author_email'),
                    license=info.get('license'),
                    project_url=info.get('project_url'),
                    homepage=info.get('home_page'),
                    requires_python=info.get('requires_python'),
                    dependencies=dependencies,
                    metadata=info
                )
                
                # Cache the package info
                await output_manager.save_to_cache(cache_key, self.package_info.dict())
                
                self.ui.success(f"Retrieved package information for {self.config.package_name} v{version}")
                return self.package_info
        
        except asyncio.TimeoutError:
            self.ui.error(f"Timeout fetching package information for {self.config.package_name}")
            return None
        except Exception as e:
            self.ui.error(f"Error fetching package information: {str(e)}")
            return None
    
    async def download_package(self):
        """Download and extract the package"""
        try:
            if not self.package_info:
                self.ui.error("Package information not available")
                return False
            
            package_dir = self.temp_dir / self.package_info.name
            if package_dir.exists():
                shutil.rmtree(package_dir)
            package_dir.mkdir(parents=True, exist_ok=True)
            
            self.ui.info(f"Downloading {self.package_info.name} v{self.package_info.version}...")
            
            async with aiohttp.ClientSession() as session:
                # Find the wheel or sdist URL
                pypi_json_url = f"https://pypi.org/pypi/{self.package_info.name}/{self.package_info.version}/json"
                
                response = await session.get(
                    pypi_json_url,
                    timeout=self.config.timeout
                )
                
                if response.status != 200:
                    self.ui.error(f"Failed to fetch release information. Status code: {response.status}")
                    return False
                
                data = await response.json()
                urls = data.get('urls', [])
                
                # Prefer wheel over sdist
                package_url = None
                for url_info in urls:
                    if url_info.get('packagetype') == 'bdist_wheel':
                        package_url = url_info.get('url')
                        break
                
                # If no wheel found, try sdist
                if not package_url:
                    for url_info in urls:
                        if url_info.get('packagetype') == 'sdist':
                            package_url = url_info.get('url')
                            break
                
                if not package_url:
                    self.ui.error("No downloadable artifacts found")
                    return False
                
                # Download the package
                download_path = package_dir / "package.zip"
                response = await session.get(
                    package_url,
                    timeout=self.config.timeout
                )
                
                if response.status != 200:
                    self.ui.error(f"Failed to download package. Status code: {response.status}")
                    return False
                
                with open(download_path, 'wb') as f:
                    f.write(await response.read())
                
                # Extract the package
                self.ui.info(f"Extracting {self.package_info.name}...")
                
                try:
                    if package_url.endswith('.whl') or package_url.endswith('.zip'):
                        # Extract as zip file
                        with zipfile.ZipFile(download_path, 'r') as zip_ref:
                            zip_ref.extractall(package_dir)
                    elif package_url.endswith('.tar.gz'):
                        # Extract as tar.gz file
                        with tarfile.open(download_path, 'r:gz') as tar_ref:
                            tar_ref.extractall(package_dir)
                    else:
                        self.ui.error(f"Unsupported package format: {package_url}")
                        return False
                    
                    self.ui.success(f"Extracted {self.package_info.name} to {package_dir}")
                    
                    return True
                
                except Exception as e:
                    self.ui.error(f"Error extracting package: {str(e)}")
                    return False
        
        except asyncio.TimeoutError:
            self.ui.error(f"Timeout downloading package {self.package_info.name}")
            return False
        except Exception as e:
            self.ui.error(f"Error downloading package: {str(e)}")
            return False
    
    async def categorize_files(self, package_dir: Path):
        """Categorize files in the package"""
        try:
            if not self.package_info:
                return False
            
            self.ui.info("Categorizing files...")
            
            files = {
                "source": [],
                "tests": [],
                "docs": [],
                "config": [],
                "data": [],
                "other": []
            }
            
            # Walk through all files
            for root, _, filenames in os.walk(package_dir):
                for filename in filenames:
                    file_path = os.path.join(root, filename)
                    rel_path = os.path.relpath(file_path, package_dir)
                    
                    # Skip __pycache__ directories
                    if "__pycache__" in rel_path:
                        continue
                    
                    # Categorize based on path and extension
                    if "test" in rel_path.lower() and self.config.include_tests:
                        files["tests"].append(rel_path)
                    elif any(ext in filename for ext in ['.py', '.pyx', '.pyd', '.so']):
                        if "test" not in rel_path.lower() or self.config.include_tests:
                            files["source"].append(rel_path)
                    elif any(doc in rel_path.lower() for doc in ['docs', 'doc', 'readme', 'changelog', '.md', '.rst']):
                        files["docs"].append(rel_path)
                    elif any(conf in filename.lower() for conf in ['setup.py', 'setup.cfg', 'pyproject.toml', '.ini', '.yaml', '.yml', '.json']):
                        files["config"].append(rel_path)
                    elif any(data_ext in filename for data_ext in ['.csv', '.json', '.xml', '.yaml', '.yml', '.txt', '.dat']):
                        files["data"].append(rel_path)
                    else:
                        files["other"].append(rel_path)
            
            # Sort files for consistent output
            for category in files:
                files[category].sort()
            
            self.package_info.files = files
            self.ui.success(f"Categorized {sum(len(files[cat]) for cat in files)} files")
            
            return True
        
        except Exception as e:
            self.ui.error(f"Error categorizing files: {str(e)}")
            return False
    
    async def generate_reports(self):
        """Generate all requested report formats"""
        try:
            if hasattr(self, 'package_info'):
                package_name = self.package_info.name
                console.print(f"[bold blue]Generating reports for {package_name}...[/bold blue]")
            else:
                console.print(f"[bold blue]Generating reports...[/bold blue]")
            
            # Keep track of successful formats
            successful_formats = []
            failed_formats = []
            
            # Get the report directory first
            report_dir = output_manager.get_report_dir(self.package_info.name, "package")
            
            # Create progress to track report generation
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                console=console
            ) as progress:
                task = progress.add_task("[cyan]Generating reports...", total=len(self.config.output_formats))
                
                # Generate requested formats
                for fmt in self.config.output_formats:
                    progress.update(task, description=f"[cyan]Generating {fmt.upper()} report...")
                    
                    try:
                        if fmt == 'txt':
                            result = await self.generate_text_report(report_dir)
                        elif fmt == 'md':
                            result = await self.generate_markdown_report(report_dir)
                        elif fmt == 'json':
                            result = await self.generate_json_report(report_dir)
                        elif fmt == 'pdf':
                            result = await self.create_pdf(report_dir)
                        else:
                            console.print(f"[yellow]Unknown format: {fmt}[/yellow]")
                            result = False
                        
                        if result:
                            successful_formats.append(fmt)
                        else:
                            failed_formats.append(fmt)
                    
                    except Exception as e:
                        console.print(f"[bold red]Error generating {fmt} report: {str(e)}[/bold red]")
                        failed_formats.append(fmt)
                    
                    progress.advance(task)
            
            # Output success message
            if successful_formats:
                console.print(f"[bold green]Successfully generated reports:[/bold green] {', '.join(fmt.upper() for fmt in successful_formats)}")
                console.print(f"[bold green]Reports saved to:[/bold green] {report_dir}")
            
            # Output failure message if any
            if failed_formats:
                console.print(f"[bold yellow]Failed to generate reports:[/bold yellow] {', '.join(fmt.upper() for fmt in failed_formats)}")
            
            return len(successful_formats) > 0
            
        except Exception as e:
            console.print(f"[bold red]Error generating reports: {str(e)}[/bold red]")
            return False
    
    async def generate_text_report(self, report_dir: Path):
        """Generate plain text report"""
        if not self.package_info:
            return False
        
        self.ui.info("Generating text report...")
        report_path = report_dir / f"{self.package_info.name}_{self.package_info.version}.txt"
        
        async with aiofiles.open(report_path, 'w') as f:
            # Header
            await f.write(f"=== {self.package_info.name} v{self.package_info.version} ===\n\n")
            
            # Basic information
            await f.write("PACKAGE INFORMATION\n")
            await f.write("-" * 80 + "\n")
            await f.write(f"Name: {self.package_info.name}\n")
            await f.write(f"Version: {self.package_info.version}\n")
            
            if self.package_info.summary:
                await f.write(f"Summary: {self.package_info.summary}\n")
            
            if self.package_info.author:
                await f.write(f"Author: {self.package_info.author}")
                if self.package_info.author_email:
                    await f.write(f" <{self.package_info.author_email}>")
                await f.write("\n")
            
            if self.package_info.license:
                await f.write(f"License: {self.package_info.license}\n")
            
            if self.package_info.homepage:
                await f.write(f"Homepage: {self.package_info.homepage}\n")
            
            if self.package_info.requires_python:
                await f.write(f"Python requirement: {self.package_info.requires_python}\n")
            
            # Dependencies
            if self.package_info.dependencies:
                await f.write("\nDEPENDENCIES\n")
                await f.write("-" * 80 + "\n")
                for dep in self.package_info.dependencies:
                    await f.write(f"- {dep}\n")
            
            # File structure
            await f.write("\nFILE STRUCTURE\n")
            await f.write("-" * 80 + "\n")
            
            for category, files in self.package_info.files.items():
                if files:
                    await f.write(f"\n{category.upper()} FILES ({len(files)})\n")
                    for file in files:
                        await f.write(f"- {file}\n")
            
            # Package description
            if self.package_info.description:
                await f.write("\nDESCRIPTION\n")
                await f.write("-" * 80 + "\n")
                # Remove markdown for plain text
                description = self.package_info.description
                description = re.sub(r'#+ ', '', description)
                description = re.sub(r'\*\*(.*?)\*\*', r'\1', description)
                description = re.sub(r'\*(.*?)\*', r'\1', description)
                await f.write(f"{description}\n")
            
            # Metadata
            if self.config.include_metadata and self.package_info.metadata:
                await f.write("\nADDITIONAL METADATA\n")
                await f.write("-" * 80 + "\n")
                for key, value in self.package_info.metadata.items():
                    if key not in ['description', 'summary'] and value:
                        await f.write(f"{key}: {value}\n")
            
            # Source Code Section
            await f.write("\nSOURCE CODE\n")
            await f.write("-" * 80 + "\n")
            await f.write("Complete concatenated source code of the package:\n\n")
            
            # Get the source files (Python files)
            source_files = self.package_info.files.get("source", [])
            if source_files:
                for file_path in sorted(source_files):
                    full_path = os.path.join(self.temp_dir, self.package_info.name, file_path)
                    try:
                        if os.path.exists(full_path) and os.path.isfile(full_path):
                            # Only include Python files or other relevant source files
                            if file_path.endswith(('.py', '.pyx', '.pyd', '.c', '.h', '.cpp', '.js', '.ts')):
                                await f.write(f"\n{'=' * 40}\n")
                                await f.write(f"FILE: {file_path}\n")
                                await f.write(f"{'=' * 40}\n\n")
                                
                                # Read and write the file content
                                try:
                                    async with aiofiles.open(full_path, 'r', encoding='utf-8', errors='ignore') as source_file:
                                        content = await source_file.read()
                                        await f.write(content)
                                        await f.write("\n\n")
                                except Exception as e:
                                    await f.write(f"[Error reading file: {str(e)}]\n\n")
                    except Exception as e:
                        await f.write(f"[Error processing file {file_path}: {str(e)}]\n\n")
            else:
                await f.write("No source files found in the package.\n")
            
            # Footer
            await f.write("\n" + "=" * 80 + "\n")
            await f.write(f"Report generated by PyPI Llama Explorer on {datetime.now()}\n")
        
        self.ui.success(f"Text report saved to {report_path}")
        return True
    
    async def generate_markdown_report(self, report_dir: Path):
        """Generate markdown report"""
        if not self.package_info:
            return False
        
        self.ui.info("Generating markdown report...")
        report_path = report_dir / f"{self.package_info.name}_{self.package_info.version}.md"
        
        async with aiofiles.open(report_path, 'w') as f:
            # Header
            await f.write(f"# {self.package_info.name} v{self.package_info.version}\n\n")
            
            if self.package_info.summary:
                await f.write(f"*{self.package_info.summary}*\n\n")
            
            # Table of contents
            if self.config.include_toc:
                await f.write("## Table of Contents\n\n")
                await f.write("- [Package Information](#package-information)\n")
                
                if self.package_info.dependencies:
                    await f.write("- [Dependencies](#dependencies)\n")
                
                await f.write("- [File Structure](#file-structure)\n")
                
                if self.package_info.description:
                    await f.write("- [Description](#description)\n")
                
                if self.config.include_metadata and self.package_info.metadata:
                    await f.write("- [Additional Metadata](#additional-metadata)\n")
                
                await f.write("\n")
            
            # Basic information
            await f.write("## Package Information\n\n")
            await f.write("| Property | Value |\n")
            await f.write("|----------|-------|\n")
            await f.write(f"| Name | {self.package_info.name} |\n")
            await f.write(f"| Version | {self.package_info.version} |\n")
            
            if self.package_info.author:
                author = self.package_info.author
                if self.package_info.author_email:
                    author += f" &lt;{self.package_info.author_email}&gt;"
                await f.write(f"| Author | {author} |\n")
            
            if self.package_info.license:
                await f.write(f"| License | {self.package_info.license} |\n")
            
            if self.package_info.homepage:
                await f.write(f"| Homepage | [{self.package_info.homepage}]({self.package_info.homepage}) |\n")
            
            if self.package_info.requires_python:
                await f.write(f"| Python requirement | {self.package_info.requires_python} |\n")
            
            await f.write("\n")
            
            # Dependencies
            if self.package_info.dependencies:
                await f.write("## Dependencies\n\n")
                for dep in self.package_info.dependencies:
                    await f.write(f"- `{dep}`\n")
                await f.write("\n")
            
            # File structure
            await f.write("## File Structure\n\n")
            
            for category, files in self.package_info.files.items():
                if files:
                    await f.write(f"### {category.title()} Files ({len(files)})\n\n")
                    await f.write("<details>\n<summary>Click to expand</summary>\n\n")
                    for file in files:
                        await f.write(f"- `{file}`\n")
                    await f.write("\n</details>\n\n")
            
            # Package description
            if self.package_info.description:
                await f.write("## Description\n\n")
                await f.write(f"{self.package_info.description}\n\n")
            
            # Metadata
            if self.config.include_metadata and self.package_info.metadata:
                await f.write("## Additional Metadata\n\n")
                await f.write("| Property | Value |\n")
                await f.write("|----------|-------|\n")
                
                for key, value in self.package_info.metadata.items():
                    if key not in ['description', 'summary'] and value:
                        # Format value for markdown table
                        if isinstance(value, list):
                            value = ", ".join(value)
                        elif isinstance(value, dict):
                            value = json.dumps(value)
                        
                        # Escape pipe characters in markdown tables
                        value = str(value).replace("|", "\\|")
                        
                        # Truncate long values
                        if len(str(value)) > 100:
                            value = str(value)[:100] + "..."
                        
                        await f.write(f"| {key} | {value} |\n")
            
            # Footer
            await f.write("\n---\n")
            await f.write(f"*Report generated by PyPI Llama Explorer on {datetime.now()}*\n")
        
        self.ui.success(f"Markdown report saved to {report_path}")
        return True
    
    async def generate_json_report(self, report_dir: Path):
        """Generate JSON report"""
        if not self.package_info:
            return False
        
        self.ui.info("Generating JSON report...")
        report_path = report_dir / f"{self.package_info.name}_{self.package_info.version}.json"
        
        # Prepare data
        data = {
            "name": self.package_info.name,
            "version": self.package_info.version,
            "summary": self.package_info.summary,
            "description": self.package_info.description,
            "author": self.package_info.author,
            "author_email": self.package_info.author_email,
            "license": self.package_info.license,
            "homepage": self.package_info.homepage,
            "requires_python": self.package_info.requires_python,
            "dependencies": self.package_info.dependencies,
            "files": self.package_info.files,
            "file_counts": {category: len(files) for category, files in self.package_info.files.items()},
            "total_files": sum(len(files) for files in self.package_info.files.values()),
            "generated_at": datetime.now().isoformat(),
            "metadata": self.package_info.metadata if self.config.include_metadata else {}
        }
        
        # Add source code data
        source_files = self.package_info.files.get("source", [])
        data["source_code"] = {}
        
        if source_files:
            for file_path in sorted(source_files):
                full_path = os.path.join(self.temp_dir, self.package_info.name, file_path)
                try:
                    if os.path.exists(full_path) and os.path.isfile(full_path):
                        # Only include Python files or other relevant source files
                        if file_path.endswith(('.py', '.pyx', '.pyd', '.c', '.h', '.cpp', '.js', '.ts')):
                            try:
                                with open(full_path, 'r', encoding='utf-8', errors='ignore') as source_file:
                                    content = source_file.read()
                                    data["source_code"][file_path] = content
                            except Exception as e:
                                data["source_code"][file_path] = f"[Error reading file: {str(e)}]"
                except Exception as e:
                    continue
        
        # Convert datetime objects to ISO format strings
        def json_serializer(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            if isinstance(obj, Enum):
                return obj.value
            raise TypeError(f"Type {type(obj)} not serializable")
        
        # Write to file
        async with aiofiles.open(report_path, 'w') as f:
            json_str = json.dumps(data, indent=2, default=json_serializer)
            await f.write(json_str)
        
        self.ui.success(f"JSON report saved to {report_path}")
        return True
    
    async def create_pdf(self, report_dir: Path):
        """Create PDF report from markdown"""
        if not PDFKIT_AVAILABLE:
            self.ui.warning("pdfkit not available. Skipping PDF generation.")
            return False
        
        if not self.package_info:
            return False
        
        self.ui.info("Generating PDF report...")
        
        md_path = report_dir / f"{self.package_info.name}_{self.package_info.version}.md"
        pdf_path = report_dir / f"{self.package_info.name}_{self.package_info.version}.pdf"
        
        try:
            if not md_path.exists():
                await self.generate_markdown_report(report_dir)
            
            # Read markdown content
            async with aiofiles.open(md_path, 'r') as f:
                md_content = await f.read()
            
            # Convert markdown to HTML
            html = markdown2.markdown(
                md_content,
                extras=["tables", "fenced-code-blocks", "header-ids"]
            )
            
            # Add CSS for better styling
            styled_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <title>{self.package_info.name} v{self.package_info.version}</title>
                <style>
                    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; padding: 20px; max-width: 900px; margin: 0 auto; }}
                    h1, h2, h3 {{ color: #333; }}
                    h1 {{ border-bottom: 1px solid #eee; padding-bottom: 10px; }}
                    h2 {{ margin-top: 30px; }}
                    table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                    th, td {{ text-align: left; padding: 12px; border-bottom: 1px solid #ddd; }}
                    th {{ background-color: #f2f2f2; }}
                    tr:hover {{ background-color: #f5f5f5; }}
                    code {{ background-color: #f5f5f5; padding: 2px 5px; border-radius: 3px; font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace; }}
                    pre {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; overflow-x: auto; }}
                    a {{ color: #0366d6; text-decoration: none; }}
                    a:hover {{ text-decoration: underline; }}
                    .footer {{ margin-top: 30px; font-size: 14px; color: #666; border-top: 1px solid #eee; padding-top: 20px; }}
                    details {{ margin: 10px 0; }}
                    summary {{ cursor: pointer; font-weight: bold; }}
                </style>
            </head>
            <body>
                {html}
                <div class="footer">
                    <p>Report generated by PyPI Llama Explorer on {datetime.now()}</p>
                </div>
            </body>
            </html>
            """
            
            # Convert HTML to PDF
            await asyncio.to_thread(
                pdfkit.from_string,
                styled_html,
                pdf_path
            )
            
            self.ui.success(f"PDF report saved to {pdf_path}")
            return True
        
        except Exception as e:
            self.ui.error(f"Error generating PDF: {str(e)}")
            return False
    
    async def process_package(self):
        """Process the package - download and generate reports"""
        try:
            self.ui.info(f"Starting analysis of {self.config.package_name}")
            
            # Step 1: Get package info
            if not await self.fetch_package_info():
                self.ui.error(f"Failed to fetch package information for {self.config.package_name}")
                return False
            
            # Step 2: Download the package
            if not await self.download_package():
                self.ui.error(f"Failed to download package {self.config.package_name}")
                return False
            
            # Use the package directory from temp_dir
            package_dir = self.temp_dir / self.package_info.name
            
            # Step 3: Analyze files
            if not await self.categorize_files(package_dir):
                self.ui.error(f"Failed to categorize files for {self.config.package_name}")
                return False
            
            # Step 4: Generate reports
            if not await self.generate_reports():
                self.ui.error(f"Failed to generate reports for {self.config.package_name}")
                
                # Try to generate a partial report anyway
                try:
                    self.ui.info("Trying to recover and generate a partial report...")
                    
                    # Create a master file with whatever information we have
                    master_file = os.path.join(self.config.output_dir, f"master_{self.package_info.name}.txt")
                    
                    async with aiofiles.open(master_file, 'w') as f:
                        await f.write(f"=== PARTIAL REPORT FOR {self.package_info.name} v{self.package_info.version} ===\n\n")
                        await f.write("Note: This is a partial report due to errors during processing.\n\n")
                        
                        # Basic information
                        await f.write("PACKAGE INFORMATION\n")
                        await f.write("-" * 80 + "\n")
                        await f.write(f"Name: {self.package_info.name}\n")
                        await f.write(f"Version: {self.package_info.version}\n")
                        
                        if self.package_info.summary:
                            await f.write(f"Summary: {self.package_info.summary}\n")
                        
                        if self.package_info.author:
                            await f.write(f"Author: {self.package_info.author}\n")
                        
                        if self.package_info.license:
                            await f.write(f"License: {self.package_info.license}\n")
                        
                        # Dependencies
                        if self.package_info.dependencies:
                            await f.write("\nDEPENDENCIES\n")
                            await f.write("-" * 80 + "\n")
                            for dep in self.package_info.dependencies:
                                await f.write(f"- {dep}\n")
                        
                        # File structure summary (if available)
                        if self.package_info.files:
                            await f.write("\nFILE STRUCTURE SUMMARY\n")
                            await f.write("-" * 80 + "\n")
                            
                            for category, files in self.package_info.files.items():
                                await f.write(f"{category}: {len(files)} files\n")
                        
                        # Footer
                        await f.write("\n" + "=" * 80 + "\n")
                        await f.write(f"Partial report generated by Llama Explorer on {datetime.now()}\n")
                    
                    self.ui.info(f"Partial report saved to {master_file}")
                except Exception as e:
                    self.ui.error(f"Failed to generate partial report: {str(e)}")
                
                return False
            
            # Step 5: Clean up temporary files
            await self.cleanup()
            
            self.ui.success(f"Successfully processed {self.package_info.name} v{self.package_info.version}")
            return True
            
        except Exception as e:
            self.ui.error(f"Error processing package: {str(e)}")
            
            # For debugging, show full traceback but only in verbose mode
            if self.config.verbose:
                import traceback
                self.ui.error(f"Traceback: {traceback.format_exc()}")
            
            # Try to clean up temporary files even if processing failed
            try:
                await self.cleanup()
            except Exception:
                pass
            
            return False
    
    async def cleanup(self):
        """Clean up temporary files"""
        try:
            if os.path.exists(self.config.temp_dir):
                shutil.rmtree(self.config.temp_dir)
                self.ui.info(f"Cleaned up temporary directory: {self.config.temp_dir}")
        except Exception as e:
            self.ui.warning(f"Error cleaning up: {str(e)}")

class PyPIProfileExplorer:
    """Class for exploring and analyzing PyPI user profiles"""
    
    def __init__(self, profile_url: str, output_dir: str = "llama_output", formats=None):
        self.profile_url = profile_url
        self.output_dir = output_dir
        self.formats = formats or ['txt', 'md', 'json']
        if PDFKIT_AVAILABLE and 'pdf' not in self.formats:
            self.formats.append('pdf')
        self.username = self._extract_username(profile_url)
        self.packages = []
        self.ui = ui_manager
    
    def _extract_username(self, url: str) -> str:
        """Extract username from profile URL"""
        return url.rstrip('/').split('/')[-1]
    
    async def fetch_user_packages(self):
        """Fetch list of packages for a PyPI user"""
        try:
            self.ui.info(f"Fetching packages for user {self.username}...")
            
            # Use cached data if available
            cache_key = f"user_packages_{self.username}"
            cached_data = await output_manager.load_from_cache(cache_key)
            if cached_data:
                self.ui.success(f"Using cached package list for user {self.username}")
                self.packages = cached_data
                return self.packages
            
            async with aiohttp.ClientSession() as session:
                packages = []
                page = 1
                
                while True:
                    url = f"https://pypi.org/user/{self.username}/?page={page}"
                    
                    response = await session.get(url)
                    if response.status != 200:
                        break
                    
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Find package elements
                    package_elements = soup.select(".package-snippet")
                    if not package_elements:
                        break
                    
                    for element in package_elements:
                        name_elem = element.select_one(".package-snippet__name")
                        version_elem = element.select_one(".package-snippet__version")
                        
                        if name_elem:
                            package_name = name_elem.text.strip()
                            package_version = version_elem.text.strip() if version_elem else ""
                            package_url = f"https://pypi.org/project/{package_name}/"
                            
                            packages.append({
                                "name": package_name,
                                "version": package_version,
                                "url": package_url
                            })
                    
                    # Check if there's a next page
                    next_page = soup.select_one("a.next")
                    if not next_page:
                        break
                    
                    page += 1
                
                self.packages = packages
                
                # Cache the package list
                await output_manager.save_to_cache(cache_key, self.packages)
                
                self.ui.success(f"Found {len(packages)} packages for user {self.username}")
                return packages
        
        except Exception as e:
            self.ui.error(f"Error fetching user packages: {str(e)}")
            return []
    
    async def process_all_packages(self):
        """Process all packages for a user"""
        try:
            # Fetch packages
            packages = await self.fetch_user_packages()
            if not packages:
                self.ui.error(f"No packages found for user {self.username}")
                return False
            
            # Create output directory
            profile_dir = os.path.join(self.output_dir, "profiles", self.username)
            os.makedirs(profile_dir, exist_ok=True)
            
            # Create progress bar
            progress = self.ui.create_progress()
            success_count = 0
            failed_count = 0
            
            with progress:
                # Add tasks
                task_id = progress.add_task(
                    f"[cyan]Processing packages for {self.username}...",
                    total=len(packages)
                )
                
                # Process packages with parallel processing
                for i, package in enumerate(packages):
                    package_name = package["name"]
                    progress.update(task_id, description=f"[cyan]Processing {package_name}...")
                    
                    try:
                        # Create config for the package
                        config = Config(
                            pypi_url=package["url"],
                            package_name=package_name,
                            output_dir=self.output_dir,
                            output_formats=self.formats
                        )
                        
                        # Process package
                        explorer = LlamaExplorer(config)
                        success = await explorer.process_package()
                        await explorer.cleanup()
                        
                        if success:
                            success_count += 1
                        else:
                            failed_count += 1
                    
                    except Exception as e:
                        self.ui.error(f"Error processing {package_name}: {str(e)}")
                        failed_count += 1
                    
                    progress.update(task_id, completed=i+1)
            
            # Generate summary report
            await self.generate_summary_report(profile_dir, success_count, failed_count)
            
            self.ui.success(
                f"Profile processing completed: "
                f"{success_count} succeeded, {failed_count} failed"
            )
            return True
        
        except Exception as e:
            self.ui.error(f"Error processing profile: {str(e)}")
            return False
    
    async def generate_summary_report(self, profile_dir: Path, success_count: int, failed_count: int):
        """Generate summary report for the profile"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(profile_dir, f"{self.username}_summary_{timestamp}.md")
        
        async with aiofiles.open(report_path, 'w') as f:
            # Header
            await f.write(f"# PyPI Profile: {self.username}\n\n")
            await f.write(f"*Profile URL: [{self.profile_url}]({self.profile_url})*\n\n")
            
            # Stats
            await f.write("## Statistics\n\n")
            await f.write(f"- Total packages: {len(self.packages)}\n")
            await f.write(f"- Successfully processed: {success_count}\n")
            await f.write(f"- Failed to process: {failed_count}\n\n")
            
            # Package table
            await f.write("## Packages\n\n")
            await f.write("| Package | Version | URL |\n")
            await f.write("|---------|---------|-----|\n")
            
            for package in sorted(self.packages, key=lambda x: x["name"].lower()):
                name = package["name"]
                version = package["version"]
                url = package["url"]
                
                await f.write(f"| {name} | {version} | [{url}]({url}) |\n")
            
            # Footer
            await f.write("\n---\n")
            await f.write(f"*Report generated by PyPI Llama Explorer on {datetime.now()}*\n")
        
        self.ui.success(f"Summary report saved to {report_path}")
        return True

class GitHubExplorer:
    """Class for exploring and analyzing GitHub repositories"""
    
    def __init__(self, repo_url: str, base_output_dir: str = "llama_output", formats=None):
        """Initialize the GitHub Explorer"""
        if formats is None:
            formats = ['txt', 'md', 'json']
            if PDFKIT_AVAILABLE:
                formats.append('pdf')
        
        self.repo_url = repo_url
        self.base_output_dir = base_output_dir
        self.formats = formats
        self.temp_dir = Path(tempfile.mkdtemp(prefix="github_explorer_"))
        self.ui = ui_manager
        self.repo_info = None
        self.api_headers = self._setup_github_api()
        
        # Language detection extensions
        self.language_extensions = {
            'python': ['.py', '.pyx', '.pyi', '.pyd'],
            'javascript': ['.js', '.jsx', '.mjs'],
            'typescript': ['.ts', '.tsx'],
            'java': ['.java', '.class', '.jar'],
            'c': ['.c', '.h'],
            'cpp': ['.cpp', '.cc', '.cxx', '.hpp', '.hxx', '.h'],
            'csharp': ['.cs'],
            'go': ['.go'],
            'rust': ['.rs'],
            'ruby': ['.rb'],
            'php': ['.php'],
            'swift': ['.swift'],
            'kotlin': ['.kt', '.kts'],
            'scala': ['.scala'],
            'html': ['.html', '.htm'],
            'css': ['.css', '.scss', '.sass', '.less'],
            'shell': ['.sh', '.bash', '.zsh'],
            'powershell': ['.ps1'],
            'sql': ['.sql'],
            'r': ['.r', '.R'],
            'dart': ['.dart'],
            'lua': ['.lua'],
            'haskell': ['.hs', '.lhs'],
            'perl': ['.pl', '.pm'],
            'julia': ['.jl'],
            'elixir': ['.ex', '.exs'],
            'clojure': ['.clj', '.cljs'],
            'erlang': ['.erl'],
            'lisp': ['.lisp', '.cl'],
            'fortran': ['.f', '.f90', '.f95'],
            'groovy': ['.groovy'],
            'objective-c': ['.m', '.mm'],
            'matlab': ['.m', '.mat'],
        }
        
        # Configuration files by type
        self.config_files = {
            'package': ['package.json', 'setup.py', 'requirements.txt', 'Cargo.toml',
                       'pom.xml', 'build.gradle', 'project.clj', 'composer.json',
                       'Gemfile', 'pyproject.toml', 'go.mod', 'build.sbt'],
            'ci': ['.travis.yml', '.github/workflows', 'azure-pipelines.yml',
                  '.gitlab-ci.yml', '.circleci', 'Jenkinsfile', '.drone.yml'],
            'docker': ['Dockerfile', 'docker-compose.yml', '.dockerignore'],
            'git': ['.gitignore', '.gitattributes', '.gitmodules'],
            'editor': ['.editorconfig', '.vscode', '.idea', '.project'],
            'lint': ['.eslintrc', '.pylintrc', '.flake8', '.rubocop.yml',
                    'tslint.json', 'stylelint.config.js', '.prettierrc'],
            'env': ['.env', '.env.example', '.env.sample', '.env.template'],
        }
    
    def _setup_github_api(self):
        """Setup GitHub API client"""
        import requests
        self.session = requests.Session()
        self.session.headers.update({
            'Accept': 'application/vnd.github.v3+json',
            'User-Agent': get_random_user_agent()
        })
        # Add GitHub token if available
        github_token = os.environ.get('GITHUB_TOKEN')
        if github_token:
            self.session.headers['Authorization'] = f'token {github_token}'
    
    def _parse_github_url(self) -> tuple:
        """Parse GitHub URL to extract owner and repo name"""
        parsed = urlparse(self.repo_url)
        path_parts = parsed.path.strip('/').split('/')
        if len(path_parts) < 2:
            raise ValueError("Invalid GitHub repository URL")
        return path_parts[0], path_parts[1]
    
    async def fetch_repo_info(self):
        """Fetch repository information from GitHub API"""
        try:
            owner, repo = self._parse_github_url()
            url = f"https://api.github.com/repos/{owner}/{repo}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.api_headers) as response:
                    if response.status != 200:
                        error_msg = f"Error fetching repository info: HTTP {response.status}"
                        try:
                            error_data = await response.json()
                            if "message" in error_data:
                                error_msg += f" - {error_data['message']}"
                        except:
                            pass
                        self.ui.error(error_msg)
                        return None
                    
                    data = await response.json()
            
            # Create GitHubInfo object
            repo_info = GitHubInfo(
                name=data["name"],
                full_name=data["full_name"],
                description=data.get("description"),
                owner=data["owner"]["login"],
                stars=data.get("stargazers_count", 0),
                forks=data.get("forks_count", 0),
                language=data.get("language"),
                topics=data.get("topics", []),
                license=data.get("license", {}).get("name") if data.get("license") else None,
                created_at=datetime.fromisoformat(data["created_at"].replace("Z", "+00:00")) if "created_at" in data else None,
                updated_at=datetime.fromisoformat(data["updated_at"].replace("Z", "+00:00")) if "updated_at" in data else None,
                clone_url=data["clone_url"],
                open_issues=data.get("open_issues_count", 0),
                files={
                    "source": [],
                    "documentation": [],
                    "tests": [],
                    "configuration": [],
                    "data": [],
                    "assets": [],
                    "other": []
                },
                languages={},
                metadata={
                    "default_branch": data.get("default_branch", "main"),
                    "visibility": data.get("visibility", "unknown"),
                    "url": data.get("html_url"),
                    "api_url": data.get("url"),
                    "size": data.get("size", 0)
                }
            )
            
            # Set the repo_info attribute
            self.repo_info = repo_info
            
            self.ui.success(f"Retrieved repository information for {repo_info.full_name}")
            return True
            
        except Exception as e:
            self.ui.error(f"Error fetching repository information: {str(e)}")
            return False
    
    async def clone_repository(self):
        """Clone the repository"""
        try:
            console.print(f"[bold blue]Cloning repository {self.repo_info.full_name}...[/bold blue]")
            
            repo_dir = self.temp_dir / self.repo_info.name
            await asyncio.to_thread(
                git.Repo.clone_from,
                self.repo_info.clone_url,
                repo_dir
            )
            
            console.print(f"[green]✓[/green] Cloned repository to {repo_dir}")
            return repo_dir
        
        except Exception as e:
            console.print(f"[bold red]Error cloning repository: {str(e)}[/bold red]")
            return None
    
    async def analyze_repository(self, repo_dir):
        """Analyze repository files and categorize them"""
        try:
            # Ensure repo_dir exists
            repo_dir = Path(repo_dir)
            if not repo_dir.exists() or not repo_dir.is_dir():
                console.print(f"[bold red]Repository directory not found: {repo_dir}[/bold red]")
                return False
            
            self.repo_dir = repo_dir  # Save for later use
            
            # Apply optimizations for large repositories
            optimizations = await optimize_for_large_repo(repo_dir)
            
            # If this is a massive repository, only analyze the README
            if optimizations["readme_only"]:
                console.print("[yellow]Repository is extremely large. Only analyzing README and basic information.[/yellow]")
                
                # Create minimal file categories
                file_categories = {
                    "source": [],
                    "docs": [],
                    "tests": [],
                    "config": [],
                    "data": [],
                    "assets": [],
                    "other": []
                }
                
                # Just find and read README
                for readme_file in ["README.md", "README.rst", "README.txt", "README"]:
                    readme_path = repo_dir / readme_file
                    if readme_path.exists():
                        try:
                            with open(readme_path, "r", encoding="utf-8", errors="ignore") as f:
                                self.repo_info.readme_content = f.read()
                                # Add to docs category
                                file_categories["docs"].append(str(readme_path.relative_to(repo_dir)))
                            break
                        except:
                            pass
                
                # Update repository information with minimal data
                self.repo_info.files = file_categories
                console.print("[green]✓[/green] Repository README analyzed")
                return True
            
            # File categories
            file_categories = {
                "source": [],    # Source code files
                "docs": [],      # Documentation files
                "tests": [],     # Test files
                "config": [],    # Configuration files
                "data": [],      # Data files
                "assets": [],    # Assets (images, etc.)
                "other": []      # Uncategorized files
            }
            
            # Track processed files to respect the limit
            processed_files = 0
            max_files = optimizations["max_files_to_process"]
            
            # Categorize files
            for root, dirs, files in os.walk(repo_dir):
                # Skip .git directory
                if ".git" in dirs:
                    dirs.remove(".git")
                
                # Skip common build/output directories
                for skip_dir in ["__pycache__", "node_modules", "dist", "build", "site-packages", "venv", ".venv", ".tox"]:
                    if skip_dir in dirs:
                        dirs.remove(skip_dir)
                
                for file in files:
                    # Check if we've reached the file limit
                    if processed_files >= max_files:
                        console.print(f"[yellow]Reached file processing limit ({max_files}). Stopping file scan.[/yellow]")
                        break
                    
                    processed_files += 1
                    
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, repo_dir)
                    
                    # Skip hidden files
                    if file.startswith(".") or "/." in rel_path:
                        continue
                    
                    # Skip large files if configured
                    if optimizations["skip_large_files"]:
                        try:
                            size_mb = os.path.getsize(file_path) / (1024 * 1024)
                            if size_mb > 5:  # Skip files larger than 5MB
                                continue
                        except:
                            pass
                    
                    # Categorize based on extension and path
                    if any(rel_path.endswith(ext) for ext in [".py", ".js", ".ts", ".java", ".c", ".cpp", ".go", ".rs", ".rb", ".php"]):
                        file_categories["source"].append(rel_path)
                    elif any(rel_path.endswith(ext) for ext in [".md", ".rst", ".txt", ".html", ".pdf", ".adoc"]) or any(doc_dir in rel_path.lower() for doc_dir in ["/doc", "/docs/", "documentation"]):
                        file_categories["docs"].append(rel_path)
                    elif "test" in rel_path.lower() or rel_path.endswith(("_test.py", "test_.py", ".spec.js", ".test.js")):
                        file_categories["tests"].append(rel_path)
                    elif any(rel_path.endswith(ext) for ext in [".json", ".yaml", ".yml", ".toml", ".ini", ".xml", ".conf"]) or file in ["Dockerfile", "docker-compose.yml", ".gitignore", ".dockerignore", "requirements.txt", "setup.py", "setup.cfg", "pyproject.toml"]:
                        file_categories["config"].append(rel_path)
                    elif any(rel_path.endswith(ext) for ext in [".csv", ".tsv", ".xlsx", ".sqlite", ".db", ".json", ".xml"]):
                        file_categories["data"].append(rel_path)
                    elif any(rel_path.endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".gif", ".svg", ".ico", ".mp3", ".mp4", ".wav", ".ogg", ".webp"]):
                        file_categories["assets"].append(rel_path)
                    else:
                        file_categories["other"].append(rel_path)
                
                # Check if we've reached the file limit
                if processed_files >= max_files:
                    break
            
            # Apply source file limit if specified
            if optimizations["source_file_limit"] and len(file_categories["source"]) > optimizations["source_file_limit"]:
                console.print(f"[yellow]Limiting source files from {len(file_categories['source'])} to {optimizations['source_file_limit']}[/yellow]")
                # Use the file filtering method to prioritize important source files
                important_files = await self._get_important_files({"source": file_categories["source"]}, max_files=optimizations["source_file_limit"])
                file_categories["source"] = important_files["source"]
            
            # Get language statistics
            language_stats = {}
            for category, files in file_categories.items():
                if category == "source":
                    for file_path in files:
                        ext = os.path.splitext(file_path)[1].lower()
                        language = {
                            ".py": "Python",
                            ".js": "JavaScript",
                            ".ts": "TypeScript",
                            ".java": "Java",
                            ".c": "C",
                            ".cpp": "C++",
                            ".cs": "C#",
                            ".go": "Go",
                            ".rb": "Ruby",
                            ".php": "PHP",
                            ".swift": "Swift",
                            ".kt": "Kotlin",
                            ".scala": "Scala",
                            ".rs": "Rust",
                            ".sh": "Shell",
                            ".sql": "SQL",
                            ".html": "HTML",
                            ".css": "CSS",
                        }.get(ext, "Other")
                        
                        # Get file size
                        try:
                            full_path = os.path.join(repo_dir, file_path)
                            size = os.path.getsize(full_path)
                            language_stats[language] = language_stats.get(language, 0) + size
                        except:
                            pass
            
            # Update repository information
            self.repo_info.files = file_categories
            self.repo_info.languages = language_stats
            
            # Try to find and read README file if not already done
            if not hasattr(self.repo_info, 'readme_content') or not self.repo_info.readme_content:
                for readme_file in ["README.md", "README.rst", "README.txt", "README"]:
                    readme_path = repo_dir / readme_file
                    if readme_path.exists():
                        try:
                            with open(readme_path, "r", encoding="utf-8", errors="ignore") as f:
                                self.repo_info.readme_content = f.read()
                            break
                        except:
                            pass
            
            # Report success
            total_files_analyzed = sum(len(files) for files in file_categories.values())
            console.print(f"[green]✓[/green] Categorized {total_files_analyzed} files")
            return True
            
        except Exception as e:
            console.print(f"[bold red]Error analyzing repository: {str(e)}[/bold red]")
            # Print traceback for debugging
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
            return False
    
    async def _fetch_contributors(self):
        """Fetch repository contributors from GitHub API"""
        try:
            owner, repo = self._parse_github_url()
            url = f"https://api.github.com/repos/{owner}/{repo}/contributors"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.api_headers) as response:
                    if response.status == 200:
                        contributors = await response.json()
                        # Limit to top 10 contributors to avoid too much data
                        self.repo_info.contributors = [
                            {
                                "login": c["login"],
                                "contributions": c["contributions"],
                                "url": c["html_url"]
                            } for c in contributors[:10]
                        ]
                    else:
                        console.print(f"[yellow]Could not fetch contributors: HTTP {response.status}[/yellow]")
        except Exception as e:
            console.print(f"[yellow]Error fetching contributors: {str(e)}[/yellow]")
    
    async def generate_reports(self):
        """Generate all requested report formats"""
        try:
            console.print(f"[bold blue]Generating reports for {self.repo_info.full_name}...[/bold blue]")
            
            # Keep track of successful formats
            successful_formats = []
            failed_formats = []
            
            # Get the report directory first
            report_dir = output_manager.get_report_dir(self.repo_info.full_name, "github")
            
            # Create progress to track report generation
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                console=console
            ) as progress:
                task = progress.add_task("[cyan]Generating reports...", total=len(self.formats))
                
                # Generate requested formats
                for fmt in self.formats:
                    progress.update(task, description=f"[cyan]Generating {fmt.upper()} report...")
                    
                    try:
                        if fmt == 'txt':
                            result = await self.generate_text_report(report_dir)
                        elif fmt == 'md':
                            result = await self.generate_markdown_report(report_dir)
                        elif fmt == 'json':
                            result = await self.generate_json_report(report_dir)
                        elif fmt == 'pdf':
                            result = await self.create_pdf(report_dir)
                        else:
                            console.print(f"[yellow]Unknown format: {fmt}[/yellow]")
                            failed_formats.append(fmt)
                            continue
                        
                        if result:
                            successful_formats.append(fmt)
                        else:
                            failed_formats.append(fmt)
                            
                    except Exception as e:
                        console.print(f"[red]Error generating {fmt} report: {str(e)}[/red]")
                        failed_formats.append(fmt)
                    
                    finally:
                        progress.advance(task)
            
            # Print summary
            if successful_formats:
                console.print(f"[green]✓ Generated reports: {', '.join(successful_formats)}[/green]")
            
            if failed_formats:
                console.print(f"[yellow]Failed to generate reports: {', '.join(failed_formats)}[/yellow]")
                
            return len(successful_formats) > 0
        
        except Exception as e:
            console.print(f"[red]Error generating reports: {str(e)}[/red]")
            return False
    
    async def generate_text_report(self, report_dir=None):
        """Generate a text report about the repository"""
        try:
            console.print(f"[bold blue]Generating text report...[/bold blue]")
            
            # Use provided report_dir or get a new one
            if report_dir is None:
                report_dir = output_manager.get_report_dir(self.repo_info.full_name, "github")
                
            report_file = report_dir / f"{self.repo_info.name}_report.txt"
            
            async with aiofiles.open(report_file, 'w') as f:
                # Title
                await f.write(f"{self.repo_info.full_name} Repository Analysis\n")
                await f.write("=" * len(f"{self.repo_info.full_name} Repository Analysis") + "\n\n")
                
                # Repository information
                await f.write("Repository Information\n")
                await f.write("-" * 22 + "\n\n")
                await f.write(f"Name:        {self.repo_info.name}\n")
                await f.write(f"Full Name:   {self.repo_info.full_name}\n")
                await f.write(f"Description: {self.repo_info.description or 'Not provided'}\n")
                await f.write(f"Owner:       {self.repo_info.owner}\n")
                await f.write(f"Stars:       {self.repo_info.stars}\n")
                await f.write(f"Forks:       {self.repo_info.forks}\n")
                await f.write(f"Language:    {self.repo_info.language or 'Not specified'}\n")
                
                if self.repo_info.license:
                    await f.write(f"License:     {self.repo_info.license}\n")
                
                if self.repo_info.created_at:
                    await f.write(f"Created at:  {self.repo_info.created_at.strftime('%Y-%m-%d %H:%M:%S')}\n")
                
                if self.repo_info.updated_at:
                    await f.write(f"Updated at:  {self.repo_info.updated_at.strftime('%Y-%m-%d %H:%M:%S')}\n")
                
                if self.repo_info.topics:
                    await f.write(f"Topics:      {', '.join(self.repo_info.topics)}\n")
                
                await f.write(f"Clone URL:   {self.repo_info.clone_url}\n")
                
                # Open issues 
                if hasattr(self.repo_info, 'open_issues'):
                    await f.write(f"Open Issues: {self.repo_info.open_issues}\n")
                
                await f.write("\n")
                
                # Language breakdown
                if hasattr(self.repo_info, 'languages') and self.repo_info.languages:
                    await f.write("Language Breakdown\n")
                    await f.write("-" * 18 + "\n\n")
                    
                    # Calculate total bytes
                    total_bytes = sum(self.repo_info.languages.values())
                    
                    # Sort languages by byte count (descending)
                    sorted_languages = sorted(
                        self.repo_info.languages.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )
                    
                    for language, bytes_count in sorted_languages:
                        percentage = (bytes_count / total_bytes) * 100
                        await f.write(f"{language}: {bytes_count} bytes ({percentage:.1f}%)\n")
                    
                    await f.write("\n")
                
                # Contributors
                if hasattr(self.repo_info, 'contributors') and self.repo_info.contributors:
                    await f.write("Top Contributors\n")
                    await f.write("-" * 16 + "\n\n")
                    
                    for contributor in self.repo_info.contributors[:10]:  # Show top 10
                        login = contributor.get('login', 'Unknown')
                        contributions = contributor.get('contributions', 0)
                        await f.write(f"{login}: {contributions} contributions\n")
                    
                    if len(self.repo_info.contributors) > 10:
                        await f.write(f"... and {len(self.repo_info.contributors) - 10} more contributors\n")
                    
                    await f.write("\n")
                
                # File structure
                await f.write("File Structure\n")
                await f.write("-" * 14 + "\n\n")
                
                total_files = sum(len(files) for files in self.repo_info.files.values())
                await f.write(f"Total files: {total_files}\n\n")
                
                for category, files in self.repo_info.files.items():
                    if files:
                        await f.write(f"{category.capitalize()} ({len(files)} files):\n")
                        
                        # Only show up to 20 files per category
                        for file_path in sorted(files)[:20]:
                            await f.write(f"  - {file_path}\n")
                        
                        if len(files) > 20:
                            await f.write(f"  ... and {len(files) - 20} more {category} files\n")
                        
                        await f.write("\n")
                
                # Source Code Section
                await f.write("Source Code\n")
                await f.write("-" * 11 + "\n\n")
                await f.write("Complete concatenated source code of key files in the repository:\n\n")
                
                # Get the source files (prioritizing Python, JS, etc.)
                source_files = self.repo_info.files.get("source", [])
                if source_files:
                    # Sort files by importance/language
                    def file_priority(filename):
                        # Lower number = higher priority
                        if filename.endswith('.py'):
                            return 0
                        elif filename.endswith(('.js', '.ts')):
                            return 1
                        elif filename.endswith(('.c', '.cpp', '.h', '.hpp')):
                            return 2
                        elif filename.endswith(('.java', '.kt')):
                            return 3
                        elif filename.endswith(('.go', '.rs', '.rb')):
                            return 4
                        else:
                            return 10
                    
                    # Sort by priority first, then by name
                    sorted_files = sorted(source_files, key=lambda f: (file_priority(f), f))
                    
                    # Limit to a reasonable number of files for large repos
                    MAX_FILES_TO_INCLUDE = 50
                    files_to_process = sorted_files[:MAX_FILES_TO_INCLUDE]
                    
                    for file_path in files_to_process:
                        full_path = os.path.join(self.repo_dir, file_path)
                        try:
                            if os.path.exists(full_path) and os.path.isfile(full_path):
                                # Only include code files
                                if file_path.endswith(('.py', '.js', '.ts', '.c', '.cpp', '.h', '.java', '.go', '.rs', '.rb', '.kt')):
                                    await f.write(f"\n{'=' * 40}\n")
                                    await f.write(f"FILE: {file_path}\n")
                                    await f.write(f"{'=' * 40}\n\n")
                                    
                                    # Read and write the file content
                                    try:
                                        async with aiofiles.open(full_path, 'r', encoding='utf-8', errors='ignore') as source_file:
                                            content = await source_file.read()
                                            await f.write(content)
                                            await f.write("\n\n")
                                    except Exception as e:
                                        await f.write(f"[Error reading file: {str(e)}]\n\n")
                        except Exception as e:
                            await f.write(f"[Error processing file {file_path}: {str(e)}]\n\n")
                    
                    if len(sorted_files) > MAX_FILES_TO_INCLUDE:
                        await f.write(f"\nNote: Only showing {MAX_FILES_TO_INCLUDE} out of {len(sorted_files)} source files.\n")
                else:
                    await f.write("No source files found in the repository.\n")
                
                # README content if available
                if hasattr(self.repo_info, 'readme_content') and self.repo_info.readme_content:
                    await f.write("\nREADME\n")
                    await f.write("-" * 6 + "\n\n")
                    
                    # Try to convert markdown to plain text
                    readme = self.repo_info.readme_content
                    readme = re.sub(r'#+ ', '', readme)  # Remove headers
                    readme = re.sub(r'\*\*(.*?)\*\*', r'\1', readme)  # Bold text
                    readme = re.sub(r'\*(.*?)\*', r'\1', readme)  # Italic text
                    readme = re.sub(r'```.*?```', '', readme, flags=re.DOTALL)  # Code blocks
                    
                    await f.write(readme)
                    await f.write("\n\n")
                
                # Footer
                await f.write("\n" + "=" * 50 + "\n")
                await f.write(f"Report generated by Llama Explorer on {datetime.now()}\n")
            
            console.print(f"[green]Text report saved to {report_file}[/green]")
            return True
        
        except Exception as e:
            console.print(f"[red]Error generating text report: {str(e)}[/red]")
            return False
    
    async def generate_markdown_report(self, report_dir=None):
        """Generate a markdown report about the repository"""
        try:
            console.print(f"[bold blue]Generating markdown report...[/bold blue]")
            
            # Use provided report_dir or get a new one
            if report_dir is None:
                report_dir = output_manager.get_report_dir(self.repo_info.full_name, "github")
                
            report_file = report_dir / f"{self.repo_info.name}_report.md"
            
            async with aiofiles.open(report_file, 'w') as f:
                # Title and Overview
                await f.write(f"# {self.repo_info.full_name} Repository Analysis\n\n")
                
                # Repository metadata
                await f.write("## Repository Overview\n\n")
                await f.write(f"- **Repository**: [{self.repo_info.full_name}](https://github.com/{self.repo_info.full_name})\n")
                await f.write(f"- **Owner**: [{self.repo_info.owner}](https://github.com/{self.repo_info.owner})\n")
                await f.write(f"- **Stars**: {self.repo_info.stars:,}\n")
                await f.write(f"- **Forks**: {self.repo_info.forks:,}\n")
                await f.write(f"- **Open Issues**: {self.repo_info.open_issues:,}\n")
                if self.repo_info.license:
                    await f.write(f"- **License**: {self.repo_info.license}\n")
                if self.repo_info.created_at:
                    await f.write(f"- **Created**: {self.repo_info.created_at.strftime('%B %d, %Y')}\n")
                if self.repo_info.updated_at:
                    await f.write(f"- **Last Updated**: {self.repo_info.updated_at.strftime('%B %d, %Y')}\n")
                
                # Topics
                if self.repo_info.topics:
                    await f.write("\n### Topics\n\n")
                    for topic in self.repo_info.topics:
                        await f.write(f"`{topic}` ")
                    await f.write("\n")
                
                # Description
                if self.repo_info.description:
                    await f.write("\n## Description\n\n")
                    await f.write(f"{self.repo_info.description}\n")
                
                # Languages
                if self.repo_info.languages:
                    await f.write("\n## Languages\n\n")
                    
                    # Create a table for language distribution
                    await f.write("| Language | Bytes | Percentage |\n")
                    await f.write("|----------|-------|------------|\n")
                    
                    total_bytes = sum(self.repo_info.languages.values())
                    for lang, bytes_count in list(self.repo_info.languages.items())[:10]:  # Top 10 languages
                        percentage = (bytes_count / total_bytes) * 100 if total_bytes > 0 else 0
                        await f.write(f"| {lang.capitalize()} | {bytes_count:,} | {percentage:.1f}% |\n")
                
                # Contributors
                if self.repo_info.contributors:
                    await f.write("\n## Contributors\n\n")
                    await f.write("| Username | Contributions | Profile |\n")
                    await f.write("|----------|--------------|--------|\n")
                    
                    for contributor in self.repo_info.contributors:
                        username = contributor.get("login", "Unknown")
                        contributions = contributor.get("contributions", 0)
                        url = contributor.get("url", f"https://github.com/{username}")
                        await f.write(f"| {username} | {contributions:,} | [Profile]({url}) |\n")
                
                # File Structure
                await f.write("\n## File Structure\n\n")
                
                total_files = sum(len(files) for files in self.repo_info.files.values())
                await f.write(f"Total files analyzed: **{total_files:,}**\n\n")
                
                # File categories with collapsible sections
                for category, files in sorted(self.repo_info.files.items()):
                    if files:
                        category_name = category.capitalize()
                        await f.write(f"<details>\n<summary><strong>{category_name} ({len(files):,} files)</strong></summary>\n\n")
                        
                        # Show up to 30 files for each category
                        display_files = files[:30]
                        if files:
                            await f.write("```\n")
                            for file in display_files:
                                await f.write(f"{file}\n")
                            if len(files) > 30:
                                await f.write(f"... and {len(files) - 30} more files\n")
                            await f.write("```\n")
                        else:
                            await f.write("*No files found in this category*\n")
                        
                        await f.write("</details>\n\n")
                
                # README Content (if available)
                if self.repo_info.readme_content:
                    await f.write("\n## README\n\n")
                    await f.write("<details>\n<summary><strong>Click to expand README</strong></summary>\n\n")
                    
                    # Limit README size in the report if it's very large
                    readme_content = self.repo_info.readme_content
                    if len(readme_content) > 10000:
                        readme_content = readme_content[:10000] + "...\n\n*README content truncated due to size*"
                    
                    await f.write(f"{readme_content}\n")
                    await f.write("</details>\n\n")
                
                # Footer with analysis information
                await f.write("\n---\n\n")
                await f.write(f"*This report was generated by Llama Explorer on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
                
            console.print(f"[green]✓[/green] Markdown report saved to {report_file}")
            return True
            
        except Exception as e:
            console.print(f"[bold red]Error generating Markdown report: {str(e)}[/bold red]")
            return False
    
    async def generate_json_report(self, report_dir=None):
        """Generate a JSON report about the repository"""
        try:
            console.print(f"[bold blue]Generating JSON report...[/bold blue]")
            
            # Use provided report_dir or get a new one
            if report_dir is None:
                report_dir = output_manager.get_report_dir(self.repo_info.full_name, "github")
                
            report_file = report_dir / f"{self.repo_info.name}_report.json"
            
            # Prepare the data dictionary
            data = {
                "name": self.repo_info.name,
                "full_name": self.repo_info.full_name,
                "description": self.repo_info.description,
                "owner": self.repo_info.owner,
                "stars": self.repo_info.stars,
                "forks": self.repo_info.forks,
                "language": self.repo_info.language,
                "topics": self.repo_info.topics,
                "license": self.repo_info.license,
                "created_at": self.repo_info.created_at,
                "updated_at": self.repo_info.updated_at,
                "clone_url": self.repo_info.clone_url,
                "files": self.repo_info.files,
                "file_counts": {category: len(files) for category, files in self.repo_info.files.items()},
                "total_files": sum(len(files) for files in self.repo_info.files.values()),
                "generated_at": datetime.now()
            }
            
            # Add language statistics if available
            if hasattr(self.repo_info, 'languages') and self.repo_info.languages:
                data["languages"] = self.repo_info.languages
                data["language_percentages"] = {}
                
                total_bytes = sum(self.repo_info.languages.values())
                for lang, bytes_count in self.repo_info.languages.items():
                    data["language_percentages"][lang] = round((bytes_count / total_bytes) * 100, 2)
            
            # Add contributors if available
            if hasattr(self.repo_info, 'contributors') and self.repo_info.contributors:
                data["contributors"] = self.repo_info.contributors
            
            # Add open issues if available
            if hasattr(self.repo_info, 'open_issues'):
                data["open_issues"] = self.repo_info.open_issues
            
            # Add readme content if available
            if hasattr(self.repo_info, 'readme_content') and self.repo_info.readme_content:
                data["readme"] = self.repo_info.readme_content
            
            # Add source code data
            source_files = self.repo_info.files.get("source", [])
            data["source_code"] = {}
            
            if source_files:
                # Sort and prioritize files similar to text report
                def file_priority(filename):
                    # Lower number = higher priority
                    if filename.endswith('.py'):
                        return 0
                    elif filename.endswith(('.js', '.ts')):
                        return 1
                    elif filename.endswith(('.c', '.cpp', '.h', '.hpp')):
                        return 2
                    elif filename.endswith(('.java', '.kt')):
                        return 3
                    elif filename.endswith(('.go', '.rs', '.rb')):
                        return 4
                    else:
                        return 10
                
                # Sort by priority first, then by name
                sorted_files = sorted(source_files, key=lambda f: (file_priority(f), f))
                
                # Limit to a reasonable number of files for large repos
                MAX_FILES_TO_INCLUDE = 100
                files_to_process = sorted_files[:MAX_FILES_TO_INCLUDE]
                
                for file_path in files_to_process:
                    full_path = os.path.join(self.repo_dir, file_path)
                    try:
                        if os.path.exists(full_path) and os.path.isfile(full_path):
                            # Only include code files
                            if file_path.endswith(('.py', '.js', '.ts', '.c', '.cpp', '.h', '.java', '.go', '.rs', '.rb', '.kt')):
                                try:
                                    with open(full_path, 'r', encoding='utf-8', errors='ignore') as source_file:
                                        content = source_file.read()
                                        data["source_code"][file_path] = content
                                except Exception as e:
                                    data["source_code"][file_path] = f"[Error reading file: {str(e)}]"
                    except Exception:
                        continue
                
                if len(sorted_files) > MAX_FILES_TO_INCLUDE:
                    data["source_code_note"] = f"Only including {MAX_FILES_TO_INCLUDE} out of {len(sorted_files)} source files."
            
            # Custom JSON serializer for datetime objects
            def json_serialize(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                raise TypeError(f"Type {type(obj)} not serializable")
            
            # Write the report
            async with aiofiles.open(report_file, 'w') as f:
                json_data = json.dumps(data, indent=2, default=json_serialize)
                await f.write(json_data)
            
            console.print(f"[green]JSON report saved to {report_file}[/green]")
            return True
        
        except Exception as e:
            console.print(f"[red]Error generating JSON report: {str(e)}[/red]")
            return False
    
    async def create_pdf(self, report_dir=None):
        """Create PDF report from markdown"""
        try:
            # Check if pdfkit is available
            try:
                import pdfkit
            except ImportError:
                console.print("[bold red]Error:[/bold red] pdfkit not installed. Install with: pip install pdfkit")
                return False
            
            console.print(f"[bold blue]Generating PDF report...[/bold blue]")
            
            # Use provided report_dir or get a new one
            if report_dir is None:
                report_dir = output_manager.get_report_dir(self.repo_info.full_name, "github")
                
            # Get the markdown file path and generate if not exists
            md_file = report_dir / f"{self.repo_info.name}_report.md"
            if not md_file.exists():
                await self.generate_markdown_report(report_dir)
            
            # Read markdown content
            async with aiofiles.open(md_file, 'r') as f:
                md_content = await f.read()
                html = markdown2.markdown(
                    md_content,
                    extras=["tables", "fenced-code-blocks", "code-friendly"]
                )
                
                # Add some CSS styling
                styled_html = f"""
                <html>
                <head>
                    <meta charset="utf-8">
                    <style>
                        body {{ font-family: Arial, sans-serif; line-height: 1.6; max-width: 900px; margin: 0 auto; padding: 20px; }}
                        h1, h2, h3, h4 {{ color: #2c3e50; }}
                        h1 {{ border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
                        h2 {{ border-bottom: 1px solid #bdc3c7; padding-bottom: 5px; }}
                        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                        th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
                        th {{ background-color: #f2f2f2; }}
                        code {{ background-color: #f8f8f8; padding: 2px 4px; border-radius: 3px; font-family: monospace; }}
                        pre {{ background-color: #f8f8f8; padding: 10px; border-radius: 5px; overflow-x: auto; }}
                        details {{ margin-bottom: 10px; }}
                        summary {{ cursor: pointer; font-weight: bold; }}
                        a {{ color: #3498db; text-decoration: none; }}
                        a:hover {{ text-decoration: underline; }}
                        .footer {{ margin-top: 30px; padding-top: 10px; border-top: 1px solid #eee; font-size: 0.9em; color: #7f8c8d; }}
                    </style>
                </head>
                <body>
                    {html}
                    <div class="footer">
                        Generated by Llama Explorer | GitHub Repository Analysis Tool
                    </div>
                </body>
                </html>
                """
                
                # Generate PDF
                options = {
                    'page-size': 'A4',
                    'margin-top': '20mm',
                    'margin-right': '20mm',
                    'margin-bottom': '20mm',
                    'margin-left': '20mm',
                    'encoding': 'UTF-8',
                    'quiet': '',
                }
                
                pdfkit.from_string(styled_html, str(report_dir / f"{self.repo_info.name}_report.pdf"), options=options)
            
            console.print(f"[green]✓[/green] PDF report saved to {str(report_dir / f'{self.repo_info.name}_report.pdf')}")
            return True
            
        except Exception as e:
            console.print(f"[bold red]Error generating PDF report: {str(e)}[/bold red]")
            return False
    
    async def cleanup(self):
        """Clean up temporary files"""
        try:
            await asyncio.to_thread(shutil.rmtree, self.temp_dir)
            console.print(f"[dim]Cleaned up temporary files[/dim]")
        except Exception as e:
            console.print(f"[yellow]Warning: Could not clean up temporary files: {str(e)}[/yellow]")
    
    async def process_repository(self):
        """Process the GitHub repository - fetch info, clone, and generate reports"""
        try:
            console.print(f"[bold blue]Starting analysis of {self.repo_url}...[/bold blue]")
            
            # Step 1: Fetch repository information
            console.print(f"[cyan]Fetching repository information...[/cyan]")
            if not await self.fetch_repo_info():
                console.print(f"[bold red]Error: Failed to fetch repository information[/bold red]")
                return False
            
            # Check memory usage after fetching info
            if not await self._check_memory_usage():
                console.print(f"[bold red]Aborting due to high memory usage[/bold red]")
                return False
            
            # Step 2: Clone the repository
            console.print(f"[cyan]Cloning repository {self.repo_info.full_name}...[/cyan]")
            repo_dir = await self.clone_repository()
            if not repo_dir:
                console.print(f"[bold red]Error: Failed to clone repository[/bold red]")
                return False
            
            # Check memory usage after cloning
            if not await self._check_memory_usage():
                console.print(f"[bold red]Aborting due to high memory usage[/bold red]")
                return False
            
            # Step 3: Analyze the repository
            console.print(f"[cyan]Analyzing repository structure...[/cyan]")
            if not await self.analyze_repository(repo_dir):
                console.print(f"[bold red]Error: Failed to analyze repository structure[/bold red]")
                
                # Try to generate a partial report anyway
                try:
                    console.print(f"[yellow]Trying to recover and generate a partial report...[/yellow]")
                    
                    # Create a master file with whatever information we have
                    master_file = os.path.join(self.base_output_dir, f"master_{self.repo_info.name}.txt")
                    
                    async with aiofiles.open(master_file, 'w') as f:
                        await f.write(f"=== PARTIAL REPORT FOR {self.repo_info.full_name} ===\n\n")
                        await f.write("Note: This is a partial report due to errors during processing.\n\n")
                        
                        # Basic information
                        await f.write("REPOSITORY INFORMATION\n")
                        await f.write("-" * 80 + "\n")
                        await f.write(f"Name: {self.repo_info.name}\n")
                        await f.write(f"Full Name: {self.repo_info.full_name}\n")
                        await f.write(f"Description: {self.repo_info.description or 'N/A'}\n")
                        await f.write(f"Owner: {self.repo_info.owner}\n")
                        await f.write(f"Stars: {self.repo_info.stars}\n")
                        await f.write(f"Forks: {self.repo_info.forks}\n")
                        await f.write(f"Language: {self.repo_info.language or 'Unknown'}\n")
                        
                        if self.repo_info.license:
                            await f.write(f"License: {self.repo_info.license}\n")
                        
                        if hasattr(self.repo_info, 'open_issues'):
                            await f.write(f"Open Issues: {self.repo_info.open_issues}\n")
                        
                        # Limited language info if available
                        if hasattr(self.repo_info, 'languages') and self.repo_info.languages:
                            await f.write("\nLANGUAGE INFORMATION\n")
                            await f.write("-" * 80 + "\n")
                            for lang, bytes_count in self.repo_info.languages.items():
                                await f.write(f"{lang}: {bytes_count} bytes\n")
                        
                        # Footer
                        await f.write("\n" + "=" * 80 + "\n")
                        await f.write(f"Partial report generated by Llama Explorer on {datetime.now()}\n")
                    
                    console.print(f"[green]Partial report saved to {master_file}[/green]")
                except Exception as e:
                    console.print(f"[red]Failed to generate partial report: {str(e)}[/red]")
                
                return False
            
            # Check memory usage after analysis
            if not await self._check_memory_usage():
                console.print(f"[bold red]Aborting due to high memory usage[/bold red]")
                # Try to generate a partial report before exiting
                try:
                    master_file = os.path.join(self.base_output_dir, f"master_{self.repo_info.name}.txt")
                    async with aiofiles.open(master_file, 'w') as f:
                        await f.write(f"=== PARTIAL REPORT FOR {self.repo_info.full_name} ===\n\n")
                        await f.write("Note: This is a partial report due to memory constraints.\n\n")
                        await f.write(f"Repository: {self.repo_info.full_name}\n")
                        await f.write(f"Memory usage exceeded safe limits. Analysis was aborted.\n")
                    console.print(f"[yellow]Saved minimal report to {master_file}[/yellow]")
                except:
                    pass
                return False
            
            # Step 4: Fetch contributors information
            console.print(f"[cyan]Fetching contributors information...[/cyan]")
            await self._fetch_contributors()
            
            # Step 5: Generate all requested reports
            console.print(f"[cyan]Generating reports...[/cyan]")
            success = await self.generate_reports()
            
            # Step 6: Clean up
            console.print(f"[cyan]Cleaning up temporary files...[/cyan]")
            await self.cleanup()
            
            if success:
                console.print(f"[bold green]✓ Analysis of {self.repo_info.full_name} completed successfully![/bold green]")
                
                # Show summary
                console.print(Panel(
                    f"[bold green]✓[/bold green] Successfully processed GitHub repository: [bold]{self.repo_info.full_name}[/bold]\n\n"
                    f"[cyan]Repository Info:[/cyan]\n"
                    f"  • Stars: {self.repo_info.stars:,}\n"
                    f"  • Forks: {self.repo_info.forks:,}\n"
                    f"  • Primary Language: {self.repo_info.language or 'Not specified'}\n"
                    f"  • Open Issues: {self.repo_info.open_issues if hasattr(self.repo_info, 'open_issues') else 'N/A'}\n"
                    f"  • Contributors: {len(self.repo_info.contributors) if hasattr(self.repo_info, 'contributors') else 'N/A'}\n\n"
                    f"[cyan]Generated Reports:[/cyan]\n" +
                    "\n".join([f"  • [bold]{fmt.upper()}[/bold]: {output_manager.get_report_dir(self.repo_info.full_name, 'github') / f'{self.repo_info.name}_report.{fmt}'}" 
                              for fmt in self.formats if fmt != 'pdf']),
                    title="Repository Analysis Complete",
                    border_style="green"
                ))
                
                return True
            else:
                console.print(f"[bold red]✗ Analysis of {self.repo_info.full_name} failed.[/bold red]")
                return False
            
        except Exception as e:
            console.print(f"[bold red]Error processing repository: {str(e)}[/bold red]")
            
            # For debugging, show full traceback
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
            
            # Try to clean up temporary files even if processing failed
            try:
                await self.cleanup()
            except Exception:
                pass
            
            return False

    async def _get_important_files(self, all_files: Dict[str, List[str]], max_files: int = 100) -> Dict[str, List[str]]:
        """
        Filter the most important files from a repository for analysis
        when dealing with very large repositories.
        """
        if not all_files:
            return {}
            
        # Calculate total file count
        total_files = sum(len(files) for files in all_files.values())
        
        # If under the limit, return all files
        if total_files <= max_files:
            return all_files
            
        console.print(f"[yellow]Repository is large ({total_files} files). Focusing on key files only.[/yellow]")
        
        # Prepare filtered files dictionary with same structure
        filtered_files = {category: [] for category in all_files.keys()}
        
        # Define priority patterns for each category
        priority_patterns = {
            "source": [
                # Main modules and entry points
                r"^main\.py$", r"__main__\.py$", r"app\.py$", r"server\.py$", r"cli\.py$",
                # Core modules 
                r"^core/", r"^lib/", r"^src/", r"^[^/]+/core/",
                # Main package files
                r"__init__\.py$", r"setup\.py$",
                # Framework entry points
                r"wsgi\.py$", r"asgi\.py$",
                # Common base classes and utilities
                r"base\.py$", r"utils\.py$", r"common\.py$", r"helpers\.py$"
            ],
            "docs": [
                # Main documentation files
                r"README\.md$", r"README\.rst$", r"CONTRIBUTING\.md$", r"CHANGELOG\.md$",
                r"LICENSE\.md$", r"^docs/index\.", r"^documentation/index\."
            ],
            "config": [
                # Important configuration files
                r"pyproject\.toml$", r"setup\.cfg$", r"requirements\.txt$", r"Dockerfile$",
                r"docker-compose\.ya?ml$", r"\.github/workflows/", r"\.circleci/", r"tox\.ini$"
            ],
            "tests": [
                # Main test files
                r"^tests/test_.*\.py$", r"^tests/conftest\.py$", r"^test/test_main\.py$" 
            ]
        }
        
        # Function to check if a file matches any priority patterns
        def is_priority_file(file_path, patterns):
            return any(re.search(pattern, file_path) for pattern in patterns)
        
        # First pass: include all high-priority files
        remaining_slots = max_files
        for category, files in all_files.items():
            if category in priority_patterns:
                patterns = priority_patterns[category]
                priority_files = [f for f in files if is_priority_file(f, patterns)]
                
                # Add priority files
                filtered_files[category].extend(priority_files)
                remaining_slots -= len(priority_files)
                
                # Track which files we've already included
                all_files[category] = [f for f in files if f not in priority_files]
        
        # If we still have slots remaining, add files proportionally from each category
        if remaining_slots > 0:
            # Calculate how many files remain in each category
            remaining_by_category = {cat: len(files) for cat, files in all_files.items()}
            total_remaining = sum(remaining_by_category.values())
            
            if total_remaining > 0:
                # Allocate slots proportionally
                for category, files in all_files.items():
                    if not files:  # Skip empty categories
                        continue
                        
                    # Calculate slots for this category
                    category_slots = int(remaining_slots * (len(files) / total_remaining))
                    if category_slots < 1 and files:
                        category_slots = 1  # Ensure at least one file per non-empty category
                        
                    # Sort files by size (ascending)
                    sorted_files = sorted(files[:category_slots])
                    
                    # Add files up to the allocated slots
                    filtered_files[category].extend(sorted_files[:category_slots])
        
        # Sort all lists for consistent output
        for category in filtered_files:
            filtered_files[category].sort()
            
        # Log how many files we're including
        total_filtered = sum(len(files) for files in filtered_files.values())
        console.print(f"[cyan]Including {total_filtered} most important files out of {total_files} total.[/cyan]")
        
        return filtered_files

    async def _check_memory_usage(self, warning_threshold_mb=500, critical_threshold_mb=800):
        """
        Check current memory usage and take action if it exceeds thresholds.
        Returns True if memory usage is acceptable, False if critical.
        """
        try:
            memory_usage_mb = get_memory_usage()
            
            if memory_usage_mb > critical_threshold_mb:
                console.print(f"[bold red]CRITICAL: Memory usage is very high ({memory_usage_mb:.1f} MB). Aborting operation.[/bold red]")
                return False
            elif memory_usage_mb > warning_threshold_mb:
                console.print(f"[yellow]WARNING: Memory usage is high ({memory_usage_mb:.1f} MB). Applying optimizations.[/yellow]")
                
                # Clear any large in-memory data
                if hasattr(self, 'repo_info') and hasattr(self.repo_info, 'readme_content') and self.repo_info.readme_content:
                    if len(self.repo_info.readme_content) > 100000:  # If README is very large
                        self.repo_info.readme_content = self.repo_info.readme_content[:50000] + "\n\n... [Content truncated to save memory] ...\n\n"
                        console.print("[yellow]Truncated large README content to save memory[/yellow]")
                
                # Force garbage collection
                import gc
                gc.collect()
                
                # Check if memory was freed
                new_memory_usage_mb = get_memory_usage()
                if new_memory_usage_mb < memory_usage_mb:
                    console.print(f"[green]Successfully reduced memory usage to {new_memory_usage_mb:.1f} MB[/green]")
                
                return True
            
            return True
        except Exception as e:
            console.print(f"[yellow]Error checking memory usage: {str(e)}[/yellow]")
            return True  # Continue operation despite error

def get_memory_usage():
    """Get the current memory usage of the process in MB"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / (1024 * 1024)  # Convert to MB
        return memory_usage
    except (ImportError, Exception):
        return 0  # Return 0 if psutil is not available or fails

async def optimize_for_large_repo(repo_dir, memory_limit_mb=500):
    """Apply optimizations for large repositories to prevent memory issues"""
    try:
        current_memory = get_memory_usage()
        
        # Count files to estimate repository size
        total_files = 0
        large_files = []
        for root, _, files in os.walk(repo_dir):
            total_files += len(files)
            # Identify large files (>10MB)
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    size_mb = os.path.getsize(file_path) / (1024 * 1024)
                    if size_mb > 10:  # Files larger than 10MB
                        large_files.append((file_path, size_mb))
                except:
                    continue
        
        # Skip very large repos or apply more aggressive filtering
        if total_files > 10000:
            console.print(f"[yellow]Repository is extremely large ({total_files:,} files). Applying strict filtering.[/yellow]")
            return {
                "skip_large_files": True,  # Skip files >5MB
                "max_files_to_process": 1000,  # Process at most 1000 files
                "skip_binary_check": True,  # Skip binary file content checks
                "source_file_limit": 100,  # Include maximum 100 source files
                "readme_only": total_files > 50000  # For massive repos, only analyze README
            }
        elif total_files > 5000:
            console.print(f"[yellow]Repository is very large ({total_files:,} files). Applying moderate filtering.[/yellow]")
            return {
                "skip_large_files": True,
                "max_files_to_process": 2000,
                "skip_binary_check": True,
                "source_file_limit": 200,
                "readme_only": False
            }
        elif total_files > 2000:
            console.print(f"[yellow]Repository is large ({total_files:,} files). Applying light filtering.[/yellow]")
            return {
                "skip_large_files": True,
                "max_files_to_process": 3000,
                "skip_binary_check": False,
                "source_file_limit": 300,
                "readme_only": False
            }
        
        # Regular repositories
        return {
            "skip_large_files": False,
            "max_files_to_process": 5000,
            "skip_binary_check": False,
            "source_file_limit": None,
            "readme_only": False
        }
    except Exception as e:
        console.print(f"[yellow]Error in optimization analysis: {str(e)}. Using default settings.[/yellow]")
        return {
            "skip_large_files": True,
            "max_files_to_process": 2000,
            "skip_binary_check": True,
            "source_file_limit": 200,
            "readme_only": False
        }

# Add GitHub-related API models and endpoints
@app.post("/analyze/github/{owner}/{repo}", response_model=AnalysisResponse)
async def analyze_github_repo(
    owner: str,
    repo: str,
    request: GitHubAnalysisRequest,
    background_tasks: BackgroundTasks
):
    """Start GitHub repository analysis"""
    repo_url = f"https://github.com/{owner}/{repo}"
    job_id = str(uuid.uuid4())
    job = {
        "id": job_id,
        "type": "github_analysis",
        "target": repo_url,
        "status": "pending",
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
        "result": None,
        "error": None
    }
    job_manager.jobs[job_id] = job
    
    background_tasks.add_task(
        process_github_async,
        repo_url,
        request.output_dir,
        request.formats,
        job_id
    )
    
    return AnalysisResponse(
        job_id=job_id,
        status="pending",
        package=f"github/{owner}/{repo}",
        created_at=job["created_at"]
    )

async def process_github_async(
    repo_url: str,
    output_dir: str,
    formats: List[str],
    job_id: str
):
    """Process GitHub repository asynchronously"""
    try:
        explorer = GitHubExplorer(repo_url, output_dir, formats)
        success = await explorer.process_repository()
        await explorer.cleanup()
        
        job_manager.update_job(
            job_id,
            "completed" if success else "failed",
            {
                "repository": repo_url,
                "reports": formats,
                "output_dir": output_dir
            } if success else None,
            "Repository processing failed" if not success else None
        )
    
    except Exception as e:
        job_manager.update_job(job_id, "failed", None, str(e))

# Update the main function to handle GitHub repositories
async def main(
    package_input: Optional[str] = None,
    output_dir: str = "llama_output",
    formats: List[str] = None,
    include_tests: bool = False,
    api_mode: bool = False,
    port: int = 8001
):
    """Main entry point for the llama-explorer"""
    if formats is None:
        formats = ["txt", "md", "json"]
    
    if api_mode:
        # Run as API server
        console.print("[bold green]Starting API server...[/bold green]")
        run_api(host="0.0.0.0", port=port)
        return
    
    # If no package is provided, show an error message
    if not package_input:
        console.print("[bold red]No package or URL specified. Use -i/--interactive for interactive mode.[/bold red]")
        console.print("Example usage:")
        console.print("  llama-explorer package_name")
        console.print("  llama-explorer https://pypi.org/project/package_name")
        console.print("  llama-explorer https://github.com/owner/repo")
        console.print("  llama-explorer -i")
        return
    
    # Check if it's a GitHub repository URL
    if package_input.startswith("https://github.com/"):
        # Extract owner and repo name from URL
        match = re.match(r'https://github\.com/([^/]+)/([^/]+)/?.*', package_input)
        if match:
            owner = match.group(1)
            repo = match.group(2)
            
            # Create job ID
            job_id = str(uuid.uuid4())
            
            # Create a GitHub explorer instance
            explorer = GitHubExplorer(package_input, output_dir, formats)
            
            # Process the repository
            success = await explorer.process_repository()
            await explorer.cleanup()
            
            if success:
                console.print(f"[bold green]✓ Successfully processed GitHub repository: {owner}/{repo}[/bold green]")
            else:
                console.print(f"[bold red]✗ Failed to process GitHub repository: {owner}/{repo}[/bold red]")
            
            return
    
    # Check if it's a PyPI user profile
    elif package_input.startswith("https://pypi.org/user/"):
        # Extract username
        match = re.match(r'https://pypi\.org/user/([^/]+)/?.*', package_input)
        if match:
            username = match.group(1)
            console.print(f"[bold blue]Processing PyPI user profile: {username}[/bold blue]")
            
            # Create job for profile analysis
            job_id = str(uuid.uuid4())
            
            # Future: Add user profile analysis here
            console.print("[yellow]PyPI user profile analysis is not yet implemented.[/yellow]")
            return
    
    # Check for standard GitHub repo format (owner/repo)
    elif '/' in package_input and not package_input.startswith('http'):
        parts = package_input.split('/')
        if len(parts) == 2:
            owner, repo = parts
            repo_url = f"https://github.com/{owner}/{repo}"
            
            console.print(f"[bold green]Processing GitHub repository: {owner}/{repo}[/bold green]")
            
            # Create a GitHub explorer instance
            explorer = GitHubExplorer(repo_url, output_dir, formats)
            
            # Process the repository
            success = await explorer.process_repository()
            await explorer.cleanup()
            
            if success:
                console.print(f"[bold green]✓ Successfully processed GitHub repository: {owner}/{repo}[/bold green]")
            else:
                console.print(f"[bold red]✗ Failed to process GitHub repository: {owner}/{repo}[/bold red]")
            
            return
    
    # For package URLs or package names (default case)
    # Create Config object for the package
    config = Config(
        pypi_url="https://pypi.org/pypi/",
        package_name=package_input,
        output_dir=output_dir,
        output_formats=formats,
        include_tests=include_tests,
        verbose=True
    )
    
    # Process the package
    explorer = LlamaExplorer(
        package_name=package_input,
        output_dir=output_dir,
        formats=formats,
        include_tests=include_tests
    )
    
    success = await explorer.process_package()
    
    if success:
        console.print(f"[bold green]✓ Successfully processed package: {package_input}[/bold green]")
    else:
        console.print(f"[bold red]✗ Failed to process package: {package_input}[/bold red]")

# Update help text to include GitHub repository support
def show_help():
    """Display help information"""
    help_text = """
[bold magenta]PyPI Llama Explorer - Commands[/bold magenta]

[cyan]Basic Usage:[/cyan]
- Enter a package name: [green]requests[/green]
- Enter a PyPI URL: [green]https://pypi.org/project/requests/[/green]
- Enter a user profile: [green]https://pypi.org/user/username/[/green]
- Enter a GitHub repository: [green]https://github.com/owner/repo[/green]

[cyan]Special Commands:[/cyan]
- [green]help[/green]: Show this help message
- [green]exit[/green]: Exit the program
- [green]clear[/green]: Clear the screen
- [green]config[/green]: Show current configuration
- [green]set format <formats>[/green]: Set output formats (txt,md,json,pdf)
- [green]set output <dir>[/green]: Set output directory
- [green]toggle tests[/green]: Toggle inclusion of tests
- [green]toggle verbose[/green]: Toggle verbose output
- [green]jobs[/green]: List active jobs
- [green]cancel <job_id>[/green]: Cancel a running job
- [green]cleanup[/green]: Clean up old reports and jobs

[cyan]Output Formats:[/cyan]
- TEXT: Detailed information in plain text
- MARKDOWN: Rich formatted documentation
- JSON: Machine-readable data
- PDF: Professional documentation (requires pdfkit)

[cyan]Features:[/cyan]
- Package analysis with dependency tracking
- GitHub repository analysis
- PyPI user profile exploration
- Parallel processing for faster results
- Caching for improved performance
- Comprehensive progress tracking
- Detailed error reporting
- Multiple output formats
- Background job processing
"""
    console.print(Panel(help_text, title="Help", border_style="cyan"))

# This allows the script to be run directly or imported as a module
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PyPI Llama Explorer")
    parser.add_argument("--api", action="store_true", help="Run in API mode")
    args = parser.parse_args()
    
    try:
        if args.api:
            run_api()
        else:
            asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[bold red]Fatal error: {str(e)}[/bold red]")
        sys.exit(1)

@app.post("/analyze/package/{package_name}", response_model=AnalysisResponse)
async def analyze_package(
    package_name: str,
    request: AnalysisRequest,
    background_tasks: BackgroundTasks
):
    """Start package analysis"""
    job_id = str(uuid.uuid4())
    job = {
        "id": job_id,
        "type": "package_analysis",
        "target": package_name,
        "status": "pending",
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc)
    }
    job_manager.jobs[job_id] = job
    
    background_tasks.add_task(
        process_package_async,
        package_name,
        request.output_dir,
        request.formats,
        request.include_tests,
        job_id
    )
    
    return AnalysisResponse(
        job_id=job_id,
        status="pending",
        package=package_name,
        created_at=job["created_at"]
    )

@app.post("/analyze/profile/{username}", response_model=AnalysisResponse)
async def analyze_profile(
    username: str,
    request: AnalysisRequest,
    background_tasks: BackgroundTasks
):
    """Start profile analysis"""
    job_id = str(uuid.uuid4())
    job = {
        "id": job_id,
        "type": "profile_analysis",
        "target": username,
        "status": "pending",
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc)
    }
    job_manager.jobs[job_id] = job
    
    background_tasks.add_task(
        process_profile_async,
        f"https://pypi.org/user/{username}/",
        request.output_dir,
        request.formats,
        job_id
    )
    
    return AnalysisResponse(
        job_id=job_id,
        status="pending",
        package=f"profile/{username}",
        created_at=job["created_at"]
    )

@app.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get job status"""
    if job_id not in job_manager.jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = job_manager.jobs[job_id]
    
    # Ensure all required fields are present
    if "result" not in job:
        job["result"] = None
    if "error" not in job:
        job["error"] = None
    
    return job

@app.get("/reports/{package_name}")
async def list_reports(package_name: str):
    """List available reports for a package"""
    package_dir = output_manager.reports_dir / package_name
    if not package_dir.exists():
        raise HTTPException(status_code=404, detail="No reports found for package")
    
    reports = []
    for version_dir in package_dir.iterdir():
        if version_dir.is_dir():
            report_files = []
            for report in version_dir.glob("*_report.*"):
                report_files.append({
                    "type": report.suffix[1:].upper(),
                    "path": f"/download/report/{package_name}/{version_dir.name}/{report.name}"
                })
            if report_files:
                reports.append({
                    "version": version_dir.name,
                    "reports": report_files
                })
    
    return {
        "package": package_name,
        "reports": reports
    }

@app.get("/download/report/{package_name}/{version}/{filename}")
async def download_report(package_name: str, version: str, filename: str):
    """Download a specific report file"""
    file_path = output_manager.reports_dir / package_name / version / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Report file not found")
    return FileResponse(file_path)

async def process_package_async(
    package_name: str,
    output_dir: str,
    formats: List[str],
    include_tests: bool,
    job_id: str
):
    """Process package asynchronously"""
    try:
        config = Config(
            pypi_url=f"https://pypi.org/pypi/{package_name}/",
            package_name=package_name,
            output_dir=output_dir,
            output_formats=formats,
            include_tests=include_tests,
            verbose=True
        )
        
        explorer = LlamaExplorer(config)
        success = await explorer.process_package()
        await explorer.cleanup()
        
        job_manager.update_job(
            job_id,
            "completed" if success else "failed",
            {
                "package": package_name,
                "reports": formats,
                "output_dir": str(output_manager.get_package_dir(package_name))
            } if success else None,
            "Package processing failed" if not success else None
        )
    
    except Exception as e:
        job_manager.update_job(job_id, "failed", None, str(e))

async def process_profile_async(
    profile_url: str,
    output_dir: str,
    formats: List[str],
    job_id: str
):
    """Process profile asynchronously"""
    try:
        profile_explorer = PyPIProfileExplorer(profile_url, output_dir, formats)
        success = await profile_explorer.process_all_packages()
        
        job_manager.update_job(
            job_id,
            "completed" if success else "failed",
            {
                "profile": profile_url,
                "output_dir": output_dir
            } if success else None,
            "Profile processing failed" if not success else None
        )
    
    except Exception as e:
        job_manager.update_job(job_id, "failed", None, str(e))

# Initialize global output manager
output_manager = OutputManager()

# Initialize global job manager
job_manager = JobManager()

# Initialize global UI manager
ui_manager = UIManager()

# Additional API endpoints
@app.get("/jobs", response_model=List[JobStatus])
async def list_jobs(status: Optional[str] = None):
    """List all jobs, optionally filtered by status"""
    return job_manager.list_jobs(status)

@app.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a running job"""
    await job_manager.cancel_job(job_id)
    return {"message": "Job cancelled successfully"}

@app.post("/cleanup")
async def cleanup_system(max_age_hours: int = 24, max_age_days: int = 30):
    """Clean up old jobs and reports"""
    job_manager.cleanup_old_jobs(max_age_hours)
    output_manager.cleanup_old_reports(max_age_days)
    return {"message": "Cleanup completed successfully"}

@app.get("/cache/info")
async def get_cache_info():
    """Get information about the cache"""
    cache_size = sum(f.stat().st_size for f in output_manager.cache_dir.glob("**/*") if f.is_file())
    cache_files = len(list(output_manager.cache_dir.glob("**/*")))
    return {
        "cache_size_bytes": cache_size,
        "cache_files": cache_files,
        "cache_dir": str(output_manager.cache_dir)
    }

@app.post("/cache/clear")
async def clear_cache():
    """Clear the cache"""
    output_manager.clear_cache()
    return {"message": "Cache cleared successfully"}

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    jobs = job_manager.list_jobs()
    return {
        "total_jobs": len(jobs),
        "active_jobs": len([j for j in jobs if j["status"] == "running"]),
        "completed_jobs": len([j for j in jobs if j["status"] == "completed"]),
        "failed_jobs": len([j for j in jobs if j["status"] == "failed"]),
        "reports_dir_size": sum(f.stat().st_size for f in output_manager.reports_dir.glob("**/*") if f.is_file()),
        "cache_dir_size": sum(f.stat().st_size for f in output_manager.cache_dir.glob("**/*") if f.is_file()),
        "temp_dir_size": sum(f.stat().st_size for f in output_manager.temp_dir.glob("**/*") if f.is_file())
    }

@app.post("/analyze/batch")
async def analyze_batch(
    request: List[Union[str, Dict[str, Any]]],
    background_tasks: BackgroundTasks
):
    """Start batch analysis of multiple packages or repositories"""
    results = []
    
    for item in request:
        try:
            if isinstance(item, str):
                # Simple string input (package name or URL)
                if "github.com" in item:
                    owner, repo = item.strip("/").split("/")[-2:]
                    result = await analyze_github_repo(
                        owner,
                        repo,
                        GitHubAnalysisRequest(),
                        background_tasks
                    )
                else:
                    package_name = item.strip("/").split("/")[-1]
                    result = await analyze_package(
                        package_name,
                        AnalysisRequest(),
                        background_tasks
                    )
            else:
                # Detailed configuration
                if item.get("type") == "github":
                    result = await analyze_github_repo(
                        item["owner"],
                        item["repo"],
                        GitHubAnalysisRequest(**item.get("config", {})),
                        background_tasks
                    )
                else:
                    result = await analyze_package(
                        item["package"],
                        AnalysisRequest(**item.get("config", {})),
                        background_tasks
                    )
            
            results.append({
                "input": item,
                "job_id": result.job_id,
                "status": "accepted"
            })
        
        except Exception as e:
            results.append({
                "input": item,
                "error": str(e),
                "status": "failed"
            })
    
    return results

@app.get("/search")
async def search_packages(
    query: str,
    type_: str = "package",
    limit: int = 10
):
    """Search for packages or repositories in the analyzed data"""
    results = []
    search_dir = output_manager.reports_dir / type_
    
    if not search_dir.exists():
        return results
    
    for item_dir in search_dir.iterdir():
        if not item_dir.is_dir():
            continue
        
        try:
            metadata = await output_manager.get_metadata(item_dir.name, type_)
            if metadata and any(
                query.lower() in str(v).lower() 
                for v in metadata.values() 
                if isinstance(v, (str, list))
            ):
                results.append({
                    "name": item_dir.name,
                    "type": type_,
                    "metadata": metadata
                })
                
                if len(results) >= limit:
                    break
        
        except Exception as e:
            ui_manager.warning(f"Error processing {item_dir}: {e}")
    
    return results

async def automated_mode(url: str, output_dir: str = "llama_output", formats=None) -> None:
    """
    Analyze a package or repository automatically based on the URL provided.
    
    Args:
        url (str): The URL, package name, or GitHub repository to analyze
        output_dir (str, optional): Directory to save reports. Defaults to "llama_output".
        formats (List[str], optional): Output formats to generate. Defaults to ["txt", "md", "json"].
    """
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
    
    console = Console()
    
    # Set default formats if not provided
    if formats is None:
        formats = ["txt", "md", "json"]
    
    # Display welcome panel
    console.print(Panel(
        f"[bold green]Analyzing:[/bold green] {url}\n\n"
        f"[bold]Output Directory:[/bold] {output_dir}\n"
        f"[bold]Report Type:[/bold] Comprehensive Master Report\n"
        f"[bold]Mode:[/bold] Automated Analysis",
        title="🦙 Llama Explorer",
        border_style="cyan"
    ))
    
    # Create progress tracking
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        # Main task for overall progress
        main_task = progress.add_task(f"Analyzing [cyan]{url}[/cyan]", total=100)
        
        try:
            # Step 1: Detect URL type (PyPI package or GitHub repo)
            if "/" in url and ("github.com" in url or len(url.split("/")) == 2):
                # GitHub repository
                console.print("Starting analysis of GitHub repository...")
                
                # Create explorer
                explorer = GitHubExplorer(url, base_output_dir=output_dir, formats=formats)
                
                # Process repository
                success = await explorer.process_repository()
                
                if success:
                    progress.update(main_task, completed=100, description=f"Analysis completed for {url}")
                    
                    # Try to generate a master report
                    try:
                        # Create a master file with all findings
                        master_file = os.path.join(output_dir, f"master_{explorer.repo_info.name}.txt")
                        await generate_master_file(explorer, explorer.repo_info.name, "github", output_dir)
                        console.print(f"\n✓ Analysis of {url} completed successfully!")
                        console.print(f"Master report saved to {master_file}")
                    except Exception as e:
                        console.print(f"Warning: Could not generate master report: {str(e)}")
                else:
                    progress.update(main_task, completed=100, description=f"Analysis failed for {url}")
                    console.print(f"✗ Analysis of {url} failed.")
            
            else:
                # PyPI package
                console.print("Starting analysis of PyPI package...")
                
                # Configure explorer
                config = Config(
                    pypi_url=f"https://pypi.org/pypi/{url}/",
                    package_name=url,
                    output_dir=output_dir,
                    output_formats=formats,
                    include_tests=True
                )
                
                # Create and run explorer
                explorer = LlamaExplorer(config)
                success = await explorer.process_package()
                
                if success:
                    progress.update(main_task, completed=100, description=f"Analysis completed for {url}")
                    
                    # Try to generate a master report
                    try:
                        # Create a master file with all findings
                        master_file = os.path.join(output_dir, f"master_{explorer.package_info.name}.txt")
                        await generate_master_file(explorer, explorer.package_info.name, "pypi", output_dir)
                        console.print(f"\n✓ Analysis of {url} completed successfully!")
                        console.print(f"Master report saved to {master_file}")
                    except Exception as e:
                        console.print(f"Warning: Could not generate master report: {str(e)}")
                else:
                    progress.update(main_task, completed=100, description=f"Analysis failed for {url}")
                    console.print(f"✗ Analysis of {url} failed.")
            
            # Display report summary
            try:
                report_summary = Panel(
                    console.capture(),
                    title="Report Summary",
                    border_style="green"
                )
                console.print(report_summary)
            except Exception:
                pass
                
        except Exception as e:
            progress.update(main_task, completed=100, description="Analysis failed")
            console.print(f"Error: {str(e)}")
            import traceback
            console.print(traceback.format_exc())
    
    # Final completion message
    console.print(Panel(
        f"Analysis Complete!\n\n"
        f"Your reports have been saved to the {output_dir} directory.\n"
        f"Thank you for using Llama Explorer!",
        title="🦙 Llama Explorer",
        border_style="green"
    ))

async def interactive_mode() -> None:
    """
    Start an interactive session to analyze a package or repository.
    The user will be prompted for input to configure the analysis.
    """
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt, Confirm
    
    console = Console()
    
    # Display welcome panel
    console.print(Panel(
        "[bold magenta]🦙 LLAMA EXPLORER[/bold magenta]\n"
        "[italic]Extract, analyze and explore PyPI packages and GitHub repositories[/italic]",
        border_style="magenta"
    ))
    
    try:
        # Step 1: Get URL or package name
        url = Prompt.ask(
            "\n[bold cyan]Enter a URL, package name, or GitHub repo (owner/repo)[/bold cyan]",
            default="requests"
        )
        
        # Step 2: Determine output formats
        formats_str = Prompt.ask(
            "\n[bold cyan]Output formats (comma-separated)[/bold cyan]",
            default="txt,md,json"
        )
        formats = [fmt.strip() for fmt in formats_str.split(",")]
        
        # Step 3: Get output directory
        output_dir = Prompt.ask(
            "\n[bold cyan]Output directory[/bold cyan]",
            default="llama_output"
        )
        
        # Confirm settings
        console.print("\n[bold]Analysis Configuration:[/bold]")
        console.print(f"Target: [cyan]{url}[/cyan]")
        console.print(f"Output formats: [cyan]{', '.join(formats)}[/cyan]")
        console.print(f"Output directory: [cyan]{output_dir}[/cyan]")
        
        if Confirm.ask("\n[bold yellow]Proceed with analysis?[/bold yellow]", default=True):
            # Run automated mode with the configured settings
            await automated_mode(url, output_dir, formats)
        else:
            console.print("[yellow]Analysis cancelled by user[/yellow]")
    
    except KeyboardInterrupt:
        console.print("\n[yellow]Analysis cancelled by user[/yellow]")
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {str(e)}")
        import traceback
        console.print(traceback.format_exc())

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "PyPI Llama Explorer API",
        "version": "1.0.0",
        "description": "API for exploring and analyzing PyPI packages"
    }
