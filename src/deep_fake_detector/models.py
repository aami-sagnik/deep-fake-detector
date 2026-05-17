"""
Data types and models for deep-fake detection results.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum
import json


class AnalysisType(str, Enum):
    """Types of analysis performed."""
    VISUAL = "visual"
    AUDIO = "audio"
    BIOMETRIC = "biometric"
    METADATA = "metadata"


@dataclass
class AnalysisResult:
    """Result from a specialist analyzer."""
    
    analyzer_type: AnalysisType
    confidence: float  # 0.0-1.0, probability of deepfake
    is_fake: bool  # True if confidence > threshold
    findings: Dict[str, Any]  # Detailed findings
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "analyzer_type": self.analyzer_type.value,
            "confidence": self.confidence,
            "is_fake": self.is_fake,
            "findings": self.findings,
            "error": self.error,
        }


@dataclass
class AggregatedReport:
    """Final forensic report aggregated from all specialists."""
    
    overall_confidence: float  # 0.0-1.0, probability of deepfake
    verdict: str  # "REAL", "FAKE", "UNCERTAIN"
    specialist_results: Dict[str, AnalysisResult] = field(default_factory=dict)
    reasoning: str = ""  # Gemma 4's reasoning about conflicting signals
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)  # Input metadata
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "overall_confidence": self.overall_confidence,
            "verdict": self.verdict,
            "specialist_results": {
                k: v.to_dict() for k, v in self.specialist_results.items()
            },
            "reasoning": self.reasoning,
            "recommendations": self.recommendations,
            "metadata": self.metadata,
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class MediaInput:
    """Input media for analysis."""
    
    file_path: Optional[str] = None  # Local file path
    url: Optional[str] = None  # Remote URL
    media_type: str = "video"  # "video", "image", "audio"
    duration: Optional[float] = None  # Duration in seconds (for videos)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "file_path": self.file_path,
            "url": self.url,
            "media_type": self.media_type,
            "duration": self.duration,
        }
