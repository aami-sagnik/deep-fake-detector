"""
Test configuration and fixtures.
"""

import pytest
from pathlib import Path


@pytest.fixture
def test_data_dir():
    """Get test data directory."""
    return Path(__file__).parent.parent / "data"


@pytest.fixture
def sample_video():
    """Get sample video path."""
    # This would be populated with actual test videos
    return None


@pytest.fixture
def sample_image():
    """Get sample image path."""
    return None
