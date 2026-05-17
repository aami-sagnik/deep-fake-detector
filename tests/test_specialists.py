"""
Unit tests for specialist models.
"""

import pytest
import numpy as np
from deep_fake_detector.models import AnalysisResult, AnalysisType


class TestVisualSpecialist:
    """Tests for visual deep-fake specialist."""
    
    def test_analysis_result_creation(self):
        """Test creating an AnalysisResult."""
        result = AnalysisResult(
            analyzer_type=AnalysisType.VISUAL,
            confidence=0.75,
            is_fake=True,
            findings={"test": "data"}
        )
        
        assert result.analyzer_type == AnalysisType.VISUAL
        assert result.confidence == 0.75
        assert result.is_fake is True
        assert result.findings == {"test": "data"}
    
    def test_analysis_result_to_dict(self):
        """Test converting AnalysisResult to dict."""
        result = AnalysisResult(
            analyzer_type=AnalysisType.AUDIO,
            confidence=0.5,
            is_fake=False,
            findings={"key": "value"}
        )
        
        data = result.to_dict()
        assert data["analyzer_type"] == "audio"
        assert data["confidence"] == 0.5
        assert data["is_fake"] is False


class TestAudioSpecialist:
    """Tests for audio spoofing specialist."""
    
    def test_audio_analysis_types(self):
        """Test audio analysis data types."""
        # Create mock audio
        audio = np.random.randn(16000).astype(np.float32)
        
        # Audio should be processable
        assert audio.dtype == np.float32
        assert len(audio) == 16000


class TestBiometricSpecialist:
    """Tests for biometric specialist."""
    
    def test_frame_validation(self):
        """Test frame validation."""
        # Create mock frame (BGR image)
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        assert frame.shape == (480, 640, 3)
        assert frame.dtype == np.uint8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
