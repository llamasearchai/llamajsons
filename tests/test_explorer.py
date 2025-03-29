"""Tests for the llama-explorer package"""
import pytest
from llama_explorer.explorer import Config, LlamaExplorer

def test_config_creation():
    """Test that Config can be created with proper values"""
    config = Config(
        pypi_url="https://pypi.org/project/requests/",
        package_name="requests",
        output_dir="output",
        output_formats=["txt", "md", "json"]
    )
    assert config.pypi_url == "https://pypi.org/project/requests/"
    assert config.package_name == "requests"
    assert config.output_dir == "output"
    assert config.output_formats == ["txt", "md", "json"]
    assert config.include_tests is False
    assert config.include_metadata is True
    assert config.include_toc is True
    assert config.temp_dir is None

def test_explorer_init():
    """Test that LlamaExplorer can be initialized with a Config"""
    config = Config(
        pypi_url="https://pypi.org/project/requests/",
        package_name="requests",
        output_dir="output"
    )
    explorer = LlamaExplorer(config)
    assert explorer.config == config
    assert explorer.config.temp_dir is not None  # tempdir is set in __init__ 