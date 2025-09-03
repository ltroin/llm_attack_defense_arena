"""Validation tests to ensure the testing infrastructure is working correctly."""

import os
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest


class TestBasicSetup:
    """Basic setup validation tests."""
    
    def test_python_version(self):
        """Test that Python version is compatible."""
        assert sys.version_info >= (3, 9), "Python 3.9 or higher required"
    
    def test_project_structure(self):
        """Test that essential project directories exist."""
        project_root = Path(__file__).parent.parent
        
        # Check key directories exist
        assert (project_root / "Attacks").exists()
        assert (project_root / "Defense").exists()
        assert (project_root / "Data").exists()
        assert (project_root / "tests").exists()
        assert (project_root / "tests" / "unit").exists()
        assert (project_root / "tests" / "integration").exists()
    
    def test_pyproject_toml_exists(self):
        """Test that pyproject.toml exists and is readable."""
        project_root = Path(__file__).parent.parent
        pyproject_path = project_root / "pyproject.toml"
        
        assert pyproject_path.exists(), "pyproject.toml not found"
        assert pyproject_path.is_file(), "pyproject.toml is not a file"
        
        # Test that it's readable
        content = pyproject_path.read_text()
        assert "[tool.poetry]" in content, "Not a Poetry project"
        assert "[tool.pytest.ini_options]" in content, "Pytest config missing"


class TestFixtures:
    """Test that fixtures are working correctly."""
    
    def test_temp_dir_fixture(self, temp_dir):
        """Test temporary directory fixture."""
        assert temp_dir.exists()
        assert temp_dir.is_dir()
        
        # Test we can create files in it
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")
        assert test_file.exists()
        assert test_file.read_text() == "test content"
    
    def test_sample_config_fixture(self, sample_config):
        """Test sample configuration fixture."""
        assert isinstance(sample_config, dict)
        assert "model" in sample_config
        assert "attack_type" in sample_config
        assert "api_keys" in sample_config
        assert isinstance(sample_config["api_keys"], dict)
    
    def test_sample_prompts_fixture(self, sample_prompts):
        """Test sample prompts fixture."""
        assert isinstance(sample_prompts, list)
        assert len(sample_prompts) > 0
        assert all(isinstance(prompt, str) for prompt in sample_prompts)
    
    def test_sample_dataframe_fixture(self, sample_dataframe):
        """Test sample DataFrame fixture."""
        pytest.importorskip("pandas")
        import pandas as pd
        
        assert isinstance(sample_dataframe, pd.DataFrame)
        assert not sample_dataframe.empty
        assert "prompt" in sample_dataframe.columns
        assert "response" in sample_dataframe.columns
        assert "attack_type" in sample_dataframe.columns
    
    def test_mock_clients(self, mock_openai_client, mock_anthropic_client):
        """Test mock API clients."""
        # Test OpenAI mock
        response = mock_openai_client.chat.completions.create(
            model="test",
            messages=[{"role": "user", "content": "test"}]
        )
        assert response.choices[0].message.content == "Test response"
        
        # Test Anthropic mock  
        message = mock_anthropic_client.messages.create(
            model="test",
            messages=[{"role": "user", "content": "test"}]
        )
        assert message.content[0].text == "Test response"


class TestMarkers:
    """Test custom pytest markers."""
    
    @pytest.mark.unit
    def test_unit_marker(self):
        """Test unit test marker."""
        assert True
    
    @pytest.mark.integration  
    def test_integration_marker(self):
        """Test integration test marker."""
        assert True
    
    @pytest.mark.slow
    def test_slow_marker(self):
        """Test slow test marker."""
        assert True
    
    @pytest.mark.gpu
    def test_gpu_marker(self):
        """Test GPU marker (will be skipped if no GPU)."""
        # This test will be automatically skipped if no GPU is available
        assert True
    
    @pytest.mark.api
    def test_api_marker(self):
        """Test API marker (will be skipped if no API keys)."""
        # This test will be automatically skipped if no API keys are available
        assert True


class TestImports:
    """Test that essential imports work."""
    
    def test_pytest_imports(self):
        """Test pytest and related imports."""
        import pytest
        from unittest.mock import Mock, MagicMock
        assert pytest is not None
        assert Mock is not None
        assert MagicMock is not None
    
    def test_basic_ml_imports(self):
        """Test basic ML library imports."""
        pytest.importorskip("numpy")
        pytest.importorskip("pandas")
        
        import numpy as np
        import pandas as pd
        
        # Test basic functionality
        arr = np.array([1, 2, 3])
        assert len(arr) == 3
        
        df = pd.DataFrame({"a": [1, 2, 3]})
        assert len(df) == 3
    
    def test_optional_ml_imports(self):
        """Test optional ML library imports."""
        try:
            import torch
            assert torch is not None
        except ImportError:
            pytest.skip("PyTorch not available")
        
        try:
            import transformers
            assert transformers is not None
        except ImportError:
            pytest.skip("Transformers not available")


class TestCoverage:
    """Tests to ensure coverage is working."""
    
    def test_coverage_includes_main_code(self):
        """Test that coverage will include main project code."""
        project_root = Path(__file__).parent.parent
        
        # Check that main Python files exist to be covered
        main_files = [
            "main.py",
            "attack.py", 
            "defence.py",
            "global_config.py",
            "models.py"
        ]
        
        existing_files = []
        for filename in main_files:
            if (project_root / filename).exists():
                existing_files.append(filename)
        
        assert len(existing_files) > 0, "No main Python files found for coverage"
    
    def test_exclusions_work(self):
        """Test that coverage exclusions are working."""
        # This function should be excluded from coverage
        def excluded_function():  # pragma: no cover
            raise NotImplementedError("This should not be covered")
        
        # We don't call the function, so it won't affect coverage
        assert excluded_function is not None


def test_this_file_runs():
    """Basic smoke test that this validation file runs."""
    assert True


if __name__ == "__main__":
    # Allow running this file directly for quick validation
    pytest.main([__file__, "-v"])