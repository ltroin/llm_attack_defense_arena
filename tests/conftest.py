"""Shared pytest fixtures for the LLM Jailbreak Arena project."""

import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Generator
from unittest.mock import Mock, MagicMock

import pytest
import pandas as pd


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    temp_path = tempfile.mkdtemp()
    try:
        yield Path(temp_path)
    finally:
        shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def temp_file(temp_dir: Path) -> Path:
    """Create a temporary file for tests."""
    temp_file_path = temp_dir / "temp_file.txt"
    temp_file_path.touch()
    return temp_file_path


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Sample configuration for testing."""
    return {
        "model": "test-model",
        "attack_type": "test-attack",
        "defense_type": "test-defense",
        "max_tokens": 100,
        "temperature": 0.7,
        "api_keys": {
            "openai": "test-openai-key",
            "anthropic": "test-anthropic-key"
        }
    }


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = Mock()
    mock_response.choices[0].message.content = "Test response"
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client for testing."""
    mock_client = Mock()
    mock_message = Mock()
    mock_message.content = [Mock()]
    mock_message.content[0].text = "Test response"
    mock_client.messages.create.return_value = mock_message
    return mock_client


@pytest.fixture
def mock_model():
    """Mock ML model for testing."""
    mock_model = Mock()
    mock_model.generate.return_value = ["Generated text"]
    mock_model.eval.return_value = mock_model
    mock_model.to.return_value = mock_model
    return mock_model


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer for testing."""
    mock_tokenizer = Mock()
    mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
    mock_tokenizer.decode.return_value = "Test text"
    mock_tokenizer.batch_encode_plus.return_value = {
        "input_ids": [[1, 2, 3, 4, 5]],
        "attention_mask": [[1, 1, 1, 1, 1]]
    }
    return mock_tokenizer


@pytest.fixture
def sample_prompts() -> list[str]:
    """Sample prompts for testing."""
    return [
        "Tell me about artificial intelligence",
        "What is the capital of France?",
        "How does machine learning work?",
        "Explain quantum computing"
    ]


@pytest.fixture
def sample_responses() -> list[str]:
    """Sample responses for testing."""
    return [
        "AI is a field of computer science...",
        "The capital of France is Paris.",
        "Machine learning is a subset of AI...",
        "Quantum computing uses quantum mechanics..."
    ]


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Sample DataFrame for testing."""
    return pd.DataFrame({
        'prompt': [
            'Test prompt 1',
            'Test prompt 2',
            'Test prompt 3'
        ],
        'response': [
            'Test response 1',
            'Test response 2',
            'Test response 3'
        ],
        'attack_type': [
            'AutoDAN',
            'DeepInception',
            'GPTFuzz'
        ],
        'success': [True, False, True]
    })


@pytest.fixture
def mock_gpu_available(monkeypatch):
    """Mock GPU availability."""
    def mock_is_available():
        return True
    
    def mock_device_count():
        return 1
    
    monkeypatch.setattr("torch.cuda.is_available", mock_is_available)
    monkeypatch.setattr("torch.cuda.device_count", mock_device_count)


@pytest.fixture
def mock_no_gpu(monkeypatch):
    """Mock no GPU availability."""
    def mock_is_available():
        return False
    
    def mock_device_count():
        return 0
    
    monkeypatch.setattr("torch.cuda.is_available", mock_is_available)
    monkeypatch.setattr("torch.cuda.device_count", mock_device_count)


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mock environment variables for testing."""
    test_env_vars = {
        "OPENAI_API_KEY": "test-openai-key",
        "ANTHROPIC_API_KEY": "test-anthropic-key",
        "AEGIS_API_KEY": "test-aegis-key",
        "GOOGLE_API_KEY": "test-google-key"
    }
    
    for key, value in test_env_vars.items():
        monkeypatch.setenv(key, value)
    
    return test_env_vars


@pytest.fixture
def results_dir(temp_dir: Path) -> Path:
    """Create a temporary results directory."""
    results_path = temp_dir / "Results"
    results_path.mkdir()
    return results_path


@pytest.fixture
def models_dir(temp_dir: Path) -> Path:
    """Create a temporary models directory."""
    models_path = temp_dir / "models"
    models_path.mkdir()
    return models_path


@pytest.fixture
def sample_attack_results() -> Dict[str, Any]:
    """Sample attack results for testing."""
    return {
        "attack_type": "test-attack",
        "model": "test-model", 
        "success_rate": 0.75,
        "total_attempts": 100,
        "successful_attacks": 75,
        "failed_attacks": 25,
        "average_response_time": 2.5,
        "prompts": [
            {"prompt": "Test prompt 1", "success": True, "response": "Test response 1"},
            {"prompt": "Test prompt 2", "success": False, "response": "I can't help with that"},
        ]
    }


@pytest.fixture
def sample_defense_results() -> Dict[str, Any]:
    """Sample defense results for testing."""
    return {
        "defense_type": "test-defense",
        "model": "test-model",
        "blocked_rate": 0.85,
        "total_attempts": 100,
        "blocked_attacks": 85,
        "allowed_attacks": 15,
        "false_positive_rate": 0.05,
        "average_response_time": 1.2
    }


@pytest.fixture(scope="session")
def slow_test_marker():
    """Marker for slow tests that should be skipped in quick test runs."""
    return pytest.mark.slow


@pytest.fixture(autouse=True)
def cleanup_test_files():
    """Automatically cleanup test files after each test."""
    yield
    # Cleanup any test files that might have been created
    test_files = [
        "test_output.txt",
        "test_results.json",
        "test_config.yaml",
        "test_data.csv"
    ]
    
    for filename in test_files:
        if os.path.exists(filename):
            os.remove(filename)


# Custom pytest markers
pytest_plugins = []


def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU hardware"
    )
    config.addinivalue_line(
        "markers", "api: mark test as requiring API keys"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle custom markers."""
    # Skip GPU tests if no GPU available
    try:
        import torch
        if not torch.cuda.is_available():
            skip_gpu = pytest.mark.skip(reason="GPU not available")
            for item in items:
                if "gpu" in item.keywords:
                    item.add_marker(skip_gpu)
    except ImportError:
        skip_gpu = pytest.mark.skip(reason="PyTorch not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
    
    # Skip API tests if no API keys available
    api_keys_available = all([
        os.getenv("OPENAI_API_KEY"),
        os.getenv("ANTHROPIC_API_KEY"),
    ])
    
    if not api_keys_available:
        skip_api = pytest.mark.skip(reason="API keys not available")
        for item in items:
            if "api" in item.keywords:
                item.add_marker(skip_api)