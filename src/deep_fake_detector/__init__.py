"""
Deep-Fake Detection System
A modular deep-fake detection system using Gemma 4 orchestrator and specialized micro-expert models.
"""

__version__ = "0.1.0"
__author__ = "Deep-Fake Detection Team"

from deep_fake_detector.config import settings
from deep_fake_detector.logger import logger
from deep_fake_detector.analyzer import DeepFakeAnalyzer
from deep_fake_detector.gradio_app import GradioApp

__all__ = [
    "settings",
    "logger",
    "DeepFakeAnalyzer",
    "GradioApp",
]
