"""
Media Processing Pipeline
Handles video/audio separation, frame extraction, and preprocessing.
"""

import cv2
import numpy as np
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional
import librosa
import soundfile as sf

from deep_fake_detector.logger import logger
from deep_fake_detector.config import settings


class MediaProcessor:
    """
    Handles media file processing including extraction, conversion, and preprocessing.
    """
    
    def __init__(self):
        """Initialize the media processor."""
        logger.info("Media processor initialized")
    
    def extract_frames(
        self,
        video_path: str,
        sample_rate: int = 5,
        max_frames: Optional[int] = None
    ) -> List[np.ndarray]:
        """
        Extract frames from a video file.
        
        Args:
            video_path: Path to video file
            sample_rate: Sample every Nth frame
            max_frames: Maximum number of frames to extract
            
        Returns:
            List of numpy arrays (BGR frames)
        """
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video file: {video_path}")
            
            frames = []
            frame_count = 0
            extracted_count = 0
            
            logger.info(f"Extracting frames from {video_path}")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Sample every Nth frame
                if frame_count % sample_rate == 0:
                    frames.append(frame)
                    extracted_count += 1
                    
                    if max_frames and extracted_count >= max_frames:
                        break
                
                frame_count += 1
            
            cap.release()
            
            logger.info(f"Extracted {len(frames)} frames from video")
            return frames
            
        except Exception as e:
            logger.error(f"Failed to extract frames: {e}")
            raise
    
    def extract_audio(
        self,
        video_path: str,
        sample_rate: int = 16000
    ) -> Tuple[Optional[np.ndarray], Optional[int]]:
        """
        Extract audio from video file using FFmpeg.
        Returns (None, None) if no audio stream exists.
        """

        temp_wav = None

        try:
            logger.info(f"Extracting audio from {video_path}")

            video_path = str(Path(video_path).resolve())

            with tempfile.NamedTemporaryFile(
                suffix=".wav",
                delete=False
            ) as tmp_file:
                temp_wav = tmp_file.name

            command = [
                "ffmpeg",
                "-y",
                "-i",
                video_path,
                "-vn",
                "-acodec", "pcm_s16le",
                "-ar", str(sample_rate),
                "-ac", "1",
                temp_wav
            ]

            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            stderr = result.stderr.lower()

            # Detect missing audio stream
            if (
                "output file does not contain any stream" in stderr
                or "stream map" in stderr
                or "does not contain any stream" in stderr
            ):
                logger.warning(
                    f"No audio stream found in {video_path}"
                )
                return None, None

            if result.returncode != 0:
                raise RuntimeError(
                    f"FFmpeg failed:\n{result.stderr}"
                )

            audio, sr = librosa.load(
                temp_wav,
                sr=sample_rate,
                mono=True
            )

            logger.info(
                f"Extracted audio: {len(audio)} samples at {sr}Hz"
            )

            return audio, sr

        except Exception as e:
            logger.exception("Audio extraction failed")
            raise RuntimeError(
                f"Audio extraction failed: {str(e)}"
            ) from e

        finally:
            if temp_wav and Path(temp_wav).exists():
                Path(temp_wav).unlink(missing_ok=True)
    
    def get_video_duration(self, video_path: str) -> float:
        """
        Get video duration in seconds.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Duration in seconds
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video file: {video_path}")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            duration = frame_count / fps if fps > 0 else 0
            logger.info(f"Video duration: {duration:.2f} seconds")
            return duration
            
        except Exception as e:
            logger.error(f"Failed to get video duration: {e}")
            return 0.0
    
    def preprocess_frames(
        self,
        frames: List[np.ndarray],
        target_size: Optional[Tuple[int, int]] = None
    ) -> List[np.ndarray]:
        """
        Preprocess frames for analysis.
        
        Args:
            frames: List of input frames
            target_size: Optional target size (width, height)
            
        Returns:
            List of preprocessed frames
        """
        try:
            processed = []
            
            for frame in frames:
                # Resize if needed
                if target_size:
                    frame = cv2.resize(frame, target_size)
                
                # Normalize
                frame = frame.astype(np.float32) / 255.0
                
                processed.append(frame)
            
            logger.info(f"Preprocessed {len(processed)} frames")
            return processed
            
        except Exception as e:
            logger.error(f"Frame preprocessing failed: {e}")
            raise
    
    def preprocess_audio(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000
    ) -> np.ndarray:
        """
        Preprocess audio for analysis.
        
        Args:
            audio: Input audio array
            sample_rate: Sample rate
            
        Returns:
            Preprocessed audio array
        """
        try:
            # Resample if needed
            if sample_rate != settings.audio_sample_rate:
                audio = librosa.resample(
                    audio,
                    orig_sr=sample_rate,
                    target_sr=settings.audio_sample_rate
                )
            
            # Normalize
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))
            
            logger.info(f"Preprocessed audio: {len(audio)} samples")
            return audio
            
        except Exception as e:
            logger.error(f"Audio preprocessing failed: {e}")
            raise
    
    def validate_media(self, file_path: str) -> bool:
        """
        Validate that a media file can be processed.
        
        Args:
            file_path: Path to media file
            
        Returns:
            True if valid, False otherwise
        """
        try:
            if not Path(file_path).exists():
                logger.error(f"File does not exist: {file_path}")
                return False
            
            # Try to open with OpenCV
            cap = cv2.VideoCapture(file_path)
            is_valid = cap.isOpened()
            cap.release()
            
            if is_valid:
                logger.info(f"Media file validated: {file_path}")
            else:
                logger.error(f"Cannot open media file: {file_path}")
            
            return is_valid
            
        except Exception as e:
            logger.error(f"Media validation failed: {e}")
            return False
    
    def process_video(
        self,
        video_path: str,
        extract_frames_flag: bool = True,
        extract_audio_flag: bool = True,
        max_frames: Optional[int] = None
    ) -> dict:
        """
        Complete video processing pipeline.
        
        Args:
            video_path: Path to video file
            extract_frames_flag: Whether to extract frames
            extract_audio_flag: Whether to extract audio
            max_frames: Maximum number of frames to extract
            
        Returns:
            Dictionary with processed data
        """
        try:
            result = {
                "file_path": video_path,
                "duration": 0.0,
                "frames": [],
                "audio": None,
                "audio_sr": None,
                "success": False,
                "error": None
            }
            
            # Validate
            if not self.validate_media(video_path):
                result["error"] = "Invalid media file"
                return result
            
            # Get duration
            result["duration"] = self.get_video_duration(video_path)
            
            # Check duration
            if result["duration"] > settings.max_video_duration:
                logger.warning(
                    f"Video duration {result['duration']}s exceeds max "
                    f"{settings.max_video_duration}s"
                )
                # Adjust max frames based on duration limit
                max_frames = int(
                    settings.max_video_duration *
                    30 / settings.frame_sample_rate
                )  # Assuming 30 fps
            
            # Extract frames
            if extract_frames_flag:
                result["frames"] = self.extract_frames(
                    video_path,
                    sample_rate=settings.frame_sample_rate,
                    max_frames=max_frames
                )
            
            # Extract audio
            if extract_audio_flag:
                audio, sr = self.extract_audio(
                    video_path,
                    sample_rate=settings.audio_sample_rate
                )

                if audio is not None:
                    result["audio"] = self.preprocess_audio(audio, sr)
                    result["audio_sr"] = sr

                else:
                    logger.warning(
                        "No audio available for this video"
                    )
            
            result["success"] = True
            logger.info(f"Successfully processed video: {video_path}")
            return result
            
        except Exception as e:
            logger.error(f"Video processing failed: {e}")
            result["error"] = str(e)
            return result
