"""
Tests for the orchestrator.
"""

import pytest
from deep_fake_detector.orchestrator import GemmaOrchestrator
from deep_fake_detector.models import AnalysisResult, AnalysisType


class TestGemmaOrchestrator:
    """Tests for Gemma orchestrator."""
    
    def test_orchestrator_initialization(self):
        """Test orchestrator can be initialized."""
        orchestrator = GemmaOrchestrator()
        assert orchestrator is not None
    
    def test_aggregate_confidence_calculation(self):
        """Test confidence calculation."""
        orchestrator = GemmaOrchestrator()
        
        specialist_results = {
            "visual": AnalysisResult(
                analyzer_type=AnalysisType.VISUAL,
                confidence=0.8,
                is_fake=True,
                findings={}
            ),
            "audio": AnalysisResult(
                analyzer_type=AnalysisType.AUDIO,
                confidence=0.6,
                is_fake=False,
                findings={}
            )
        }
        
        confidence = orchestrator._calculate_aggregate_confidence(specialist_results)
        assert 0.0 <= confidence <= 1.0
    
    def test_verdict_determination(self):
        """Test verdict determination logic."""
        orchestrator = GemmaOrchestrator()
        
        specialist_results = {
            "visual": AnalysisResult(
                analyzer_type=AnalysisType.VISUAL,
                confidence=0.85,
                is_fake=True,
                findings={}
            )
        }
        
        verdict = orchestrator._determine_verdict(0.85, specialist_results)
        assert verdict in ["REAL", "FAKE", "UNCERTAIN"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
