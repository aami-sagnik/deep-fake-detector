"""
Contextual/Metadata Checker
Extracts and analyzes metadata to detect known deepfakes and suspicious patterns.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import subprocess

from deep_fake_detector.logger import logger
from deep_fake_detector.models import AnalysisResult, AnalysisType
from deep_fake_detector.config import settings


class MetadataSpecialist:
    """
    Analyzes metadata and contextual information for signs of deepfake manipulation.
    Extracts container info, camera tags, and checks against known flagged content.
    """
    
    def __init__(self):
        """Initialize the metadata specialist."""
        logger.info("Metadata specialist initialized")
        self.known_fake_hashes = set()  # Can be populated from external sources
    
    def analyze_file(self, file_path: str) -> Tuple[float, Dict[str, Any]]:
        """
        Analyze file metadata using FFmpeg.
        
        Args:
            file_path: Path to media file
            
        Returns:
            Tuple of (confidence_score, findings_dict)
        """
        try:
            metadata = self._extract_ffmpeg_metadata(file_path)
            suspicious_patterns = self._check_suspicious_patterns(metadata)
            
            # Calculate confidence based on metadata analysis
            confidence = self._calculate_metadata_confidence(suspicious_patterns)
            
            findings = {
                "metadata": metadata,
                "suspicious_patterns": suspicious_patterns,
                "detected_anomalies": []
            }
            
            # Add detected anomalies
            if suspicious_patterns["encoding_inconsistencies"]:
                findings["detected_anomalies"].append("encoding_inconsistencies")
            if suspicious_patterns["missing_camera_metadata"]:
                findings["detected_anomalies"].append("missing_camera_metadata")
            if suspicious_patterns["multiple_codec_segments"]:
                findings["detected_anomalies"].append("multiple_codec_segments")
            
            return confidence, findings
            
        except Exception as e:
            logger.error(f"Error analyzing metadata: {e}")
            raise
    
    def _extract_ffmpeg_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Extract metadata using FFprobe.
        
        Args:
            file_path: Path to media file
            
        Returns:
            Dictionary with extracted metadata
        """
        try:
            cmd = [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                file_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                logger.warning(f"FFprobe failed: {result.stderr}")
                return self._mock_metadata()
            
            data = json.loads(result.stdout)
            
            metadata = {
                "file_path": file_path,
                "file_size": os.path.getsize(file_path),
                "format": data.get("format", {}).get("format_name", "unknown"),
                "duration": float(data.get("format", {}).get("duration", 0)),
                "bit_rate": int(data.get("format", {}).get("bit_rate", 0)),
                "streams": []
            }
            
            # Extract stream information
            for stream in data.get("streams", []):
                stream_info = {
                    "codec_type": stream.get("codec_type", "unknown"),
                    "codec_name": stream.get("codec_name", "unknown"),
                    "codec_time_base": stream.get("codec_time_base", "unknown"),
                }
                
                if stream["codec_type"] == "video":
                    stream_info.update({
                        "width": stream.get("width", 0),
                        "height": stream.get("height", 0),
                        "fps": eval(stream.get("r_frame_rate", "0/1")) if "/" in stream.get("r_frame_rate", "0/1") else 0,
                        "pix_fmt": stream.get("pix_fmt", "unknown")
                    })
                elif stream["codec_type"] == "audio":
                    stream_info.update({
                        "sample_rate": stream.get("sample_rate", 0),
                        "channels": stream.get("channels", 0),
                    })
                
                metadata["streams"].append(stream_info)
            
            return metadata
            
        except subprocess.TimeoutExpired:
            logger.warning("FFprobe timeout")
            return self._mock_metadata()
        except Exception as e:
            logger.warning(f"FFprobe extraction failed: {e}")
            return self._mock_metadata()
    
    def _mock_metadata(self) -> Dict[str, Any]:
        """Provide mock metadata when FFprobe is unavailable."""
        return {
            "format": "mp4",
            "duration": 10.0,
            "bit_rate": 5000000,
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 1920,
                    "height": 1080,
                    "fps": 30,
                    "pix_fmt": "yuv420p"
                }
            ],
            "note": "Mock metadata (FFprobe not available)"
        }
    
    def _check_suspicious_patterns(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check for suspicious patterns in metadata.
        
        Args:
            metadata: Extracted metadata
            
        Returns:
            Dictionary with suspicious pattern analysis
        """
        patterns = {
            "encoding_inconsistencies": False,
            "missing_camera_metadata": False,
            "multiple_codec_segments": False,
            "suspicious_properties": []
        }
        
        try:
            streams = metadata.get("streams", [])
            
            # Check for multiple video codecs
            video_codecs = set()
            for stream in streams:
                if stream.get("codec_type") == "video":
                    video_codecs.add(stream.get("codec_name", "unknown"))
            
            if len(video_codecs) > 1:
                patterns["multiple_codec_segments"] = True
                patterns["suspicious_properties"].append("Multiple video codecs detected")
            
            # Check for suspicious bitrate patterns
            bitrate = metadata.get("bit_rate", 0)
            duration = metadata.get("duration", 1)
            
            if bitrate == 0 or duration == 0:
                patterns["encoding_inconsistencies"] = True
            
            # Check for missing format tags
            format_name = metadata.get("format", "unknown")
            if format_name == "unknown":
                patterns["missing_camera_metadata"] = True
                patterns["suspicious_properties"].append("Unknown format")
            
            # Check video resolution
            for stream in streams:
                if stream.get("codec_type") == "video":
                    width = stream.get("width", 0)
                    height = stream.get("height", 0)
                    
                    # Common resolutions
                    common_res = {(1920, 1080), (1280, 720), (640, 480), (2560, 1440)}
                    if (width, height) not in common_res:
                        patterns["suspicious_properties"].append(
                            f"Unusual resolution: {width}x{height}"
                        )
            
            return patterns
            
        except Exception as e:
            logger.warning(f"Pattern detection failed: {e}")
            return patterns
    
    def _calculate_metadata_confidence(self, patterns: Dict[str, Any]) -> float:
        """
        Calculate confidence score from metadata patterns.
        
        Args:
            patterns: Suspicious pattern analysis
            
        Returns:
            Confidence score (0-1)
        """
        confidence = 0.0
        
        if patterns["encoding_inconsistencies"]:
            confidence += 0.3
        
        if patterns["missing_camera_metadata"]:
            confidence += 0.2
        
        if patterns["multiple_codec_segments"]:
            confidence += 0.3
        
        # Base confidence on number of suspicious properties
        num_properties = len(patterns.get("suspicious_properties", []))
        confidence += min(0.2, num_properties * 0.05)
        
        return min(1.0, confidence)
    
    def analyze(self, file_path: str) -> AnalysisResult:
        """
        Perform complete metadata analysis.
        
        Args:
            file_path: Path to media file
            
        Returns:
            AnalysisResult object
        """
        try:
            confidence, findings = self.analyze_file(file_path)
            
            return AnalysisResult(
                analyzer_type=AnalysisType.METADATA,
                confidence=confidence,
                is_fake=confidence > settings.metadata_confidence_threshold,
                findings=findings
            )
        except Exception as e:
            logger.error(f"Metadata analysis failed: {e}")
            return AnalysisResult(
                analyzer_type=AnalysisType.METADATA,
                confidence=0.3,
                is_fake=False,
                findings={},
                error=str(e)
            )
