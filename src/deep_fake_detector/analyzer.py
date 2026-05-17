"""
Main Analyzer Module
Coordinates the entire deep-fake detection pipeline.
"""

import asyncio
from typing import Dict, Any, List, Optional
from pathlib import Path

from deep_fake_detector.logger import logger
from deep_fake_detector.config import settings
from deep_fake_detector.media_processor import MediaProcessor
from deep_fake_detector.specialist.visual import VisualSpecialist
from deep_fake_detector.specialist.audio import AudioSpecialist
from deep_fake_detector.specialist.biometric import BiometricSpecialist
from deep_fake_detector.specialist.metadata import MetadataSpecialist
from deep_fake_detector.orchestrator import GemmaOrchestrator
from deep_fake_detector.models import AnalysisResult, AggregatedReport


class DeepFakeAnalyzer:
    """
    Main analyzer that coordinates the complete deep-fake detection pipeline.
    """
    
    def __init__(self):
        """Initialize the analyzer with all specialist modules."""
        logger.info("Initializing Deep-Fake Analyzer")
        
        # Initialize specialists
        self.visual_specialist = None
        self.audio_specialist = None
        self.biometric_specialist = None
        self.metadata_specialist = MetadataSpecialist()
        self.media_processor = MediaProcessor()
        self.orchestrator = GemmaOrchestrator()
        
        # Lazy load expensive models
        self._visual_loaded = False
        self._audio_loaded = False
        self._biometric_loaded = False
        
        logger.info("Analyzer initialization complete")
    
    def _load_specialists(self):
        """Lazy load specialist models on first use."""
        if not self._visual_loaded:
            try:
                self.visual_specialist = VisualSpecialist()
                self._visual_loaded = True
            except Exception as e:
                logger.error(f"Failed to load visual specialist: {e}")
        
        if not self._audio_loaded:
            try:
                self.audio_specialist = AudioSpecialist()
                self._audio_loaded = True
            except Exception as e:
                logger.error(f"Failed to load audio specialist: {e}")
        
        if not self._biometric_loaded:
            try:
                self.biometric_specialist = BiometricSpecialist()
                self._biometric_loaded = True
            except Exception as e:
                logger.error(f"Failed to load biometric specialist: {e}")
    
    def analyze_file(self, file_path: str) -> AggregatedReport:
        """
        Analyze a media file for deepfake indicators.
        
        Args:
            file_path: Path to media file
            
        Returns:
            AggregatedReport with forensic findings
        """
        try:
            logger.info(f"Starting analysis of: {file_path}")
            
            # Load specialists
            self._load_specialists()
            
            # Process media
            logger.info("Processing media file")
            processing_result = self.media_processor.process_video(file_path)
            
            if not processing_result["success"]:
                raise RuntimeError(processing_result.get("error", "Processing failed"))
            
            # Prepare metadata
            media_metadata = {
                "file_path": file_path,
                "duration": processing_result["duration"],
                "frame_count": len(processing_result["frames"]),
                "has_audio": processing_result["audio"] is not None,
            }
            
            # Run specialists in parallel
            specialist_results = {}
            
            # Visual analysis
            if processing_result["frames"] and self.visual_specialist:
                logger.info("Running visual analysis")
                specialist_results["visual"] = self.visual_specialist.analyze(
                    processing_result["frames"]
                )
            else:
                logger.warning("Skipping visual analysis")
            
            # Audio analysis
            if processing_result["audio"] is not None and self.audio_specialist:
                logger.info("Running audio analysis")
                specialist_results["audio"] = self.audio_specialist.analyze(
                    processing_result["audio"]
                )
            else:
                logger.warning("Skipping audio analysis")
            
            # Biometric analysis
            if processing_result["frames"] and self.biometric_specialist:
                logger.info("Running biometric analysis")
                specialist_results["biometric"] = self.biometric_specialist.analyze(
                    processing_result["frames"]
                )
            else:
                logger.warning("Skipping biometric analysis")
            
            # Metadata analysis
            logger.info("Running metadata analysis")
            specialist_results["metadata"] = self.metadata_specialist.analyze(
                file_path
            )
            
            # Orchestrate findings
            logger.info("Orchestrating specialist findings")
            report = self.orchestrator.orchestrate(
                specialist_results,
                media_metadata
            )
            
            logger.info(f"Analysis complete: {report.verdict}")
            return report
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}", exc_info=True)
            # Return error report
            return AggregatedReport(
                overall_confidence=0.5,
                verdict="UNCERTAIN",
                specialist_results={},
                reasoning=f"Analysis failed: {str(e)}",
                recommendations=["Manual review recommended"],
                metadata={"file_path": file_path, "error": str(e)}
            )
    
    def analyze_image(self, image_path: str) -> AggregatedReport:
        """
        Analyze an image file for deepfake indicators.
        
        Args:
            image_path: Path to image file
            
        Returns:
            AggregatedReport with forensic findings
        """
        try:
            from PIL import Image
            import cv2
            import numpy as np
            
            logger.info(f"Analyzing image: {image_path}")
            self._load_specialists()
            
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                raise RuntimeError(f"Cannot load image: {image_path}")
            
            frames = [img]
            
            specialist_results = {}
            
            # Visual analysis
            if self.visual_specialist:
                logger.info("Running visual analysis on image")
                specialist_results["visual"] = self.visual_specialist.analyze(frames)
            
            # Biometric analysis
            if self.biometric_specialist:
                logger.info("Running biometric analysis on image")
                specialist_results["biometric"] = self.biometric_specialist.analyze(frames)
            
            # Metadata analysis
            logger.info("Running metadata analysis on image")
            specialist_results["metadata"] = self.metadata_specialist.analyze(
                image_path
            )
            
            media_metadata = {
                "file_path": image_path,
                "media_type": "image",
                "duration": None,
                "frame_count": 1,
            }
            
            # Orchestrate
            report = self.orchestrator.orchestrate(
                specialist_results,
                media_metadata
            )
            
            logger.info(f"Image analysis complete: {report.verdict}")
            return report
            
        except Exception as e:
            logger.error(f"Image analysis failed: {e}", exc_info=True)
            return AggregatedReport(
                overall_confidence=0.5,
                verdict="UNCERTAIN",
                specialist_results={},
                reasoning=f"Analysis failed: {str(e)}",
                recommendations=["Manual review recommended"],
                metadata={"file_path": image_path, "error": str(e)}
            )
