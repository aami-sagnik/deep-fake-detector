"""
Audio Spoofing Specialist
Detects synthetic speech, voice cloning, and acoustic anomalies using Wav2Vec2.
"""

import torch
import librosa
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import soundfile as sf

from transformers import AutoProcessor, AutoModelForAudioClassification
from deep_fake_detector.logger import logger
from deep_fake_detector.models import AnalysisResult, AnalysisType
from deep_fake_detector.config import settings


class AudioSpecialist:
    """
    Analyzes audio content for signs of voice cloning and synthesis artifacts.
    Uses Wav2Vec2 for acoustic feature analysis.
    """
    
    def __init__(self, model_name: str = "facebook/wav2vec2-base-960h"):
        """
        Initialize the audio specialist.
        
        Args:
            model_name: HuggingFace model name for audio analysis
        """
        self.device = settings.device
        self.model_name = model_name
        self.processor = None
        self.model = None
        self.sample_rate = settings.audio_sample_rate
        
        logger.info(f"Loading audio specialist model: {model_name}")
        self._load_model()
    
    def _load_model(self):
        """Load the audio model and processor."""
        try:
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                cache_dir=str(settings.cache_dir)
            )
            self.model = AutoModelForAudioClassification.from_pretrained(
                self.model_name,
                cache_dir=str(settings.cache_dir),
                num_labels=2
            )
            self.model = self.model.to(self.device)
            self.model.eval()
            logger.info("Audio model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load audio model: {e}")
            raise
    
    def analyze_audio_array(self, audio: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        """
        Analyze raw audio array for synthesis artifacts.
        
        Args:
            audio: numpy array with audio samples
            
        Returns:
            Tuple of (confidence_score, findings_dict)
        """
        try:
            # Resample if necessary
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            
            # Normalize audio
            audio = audio / (np.max(np.abs(audio)) + 1e-6)
            
            # Prepare input for model
            inputs = self.processor(
                audio,
                sampling_rate=self.sample_rate,
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Get predictions
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            synthetic_prob = probs[0, 1].item() if logits.shape[1] > 1 else probs[0, 0].item()
            
            # Analyze acoustic features
            spectral_features = self._analyze_spectrogram(audio)
            phase_coherence = self._check_phase_coherence(audio)
            silence_patterns = self._analyze_silence_patterns(audio)
            
            findings = {
                "model_confidence": float(synthetic_prob),
                "spectral_flatness": float(spectral_features["flatness"]),
                "phase_coherence": float(phase_coherence),
                "silence_ratio": float(silence_patterns["silence_ratio"]),
                "detected_anomalies": []
            }
            
            # Add anomalies if detected
            if spectral_features["flatness"] > 0.7:
                findings["detected_anomalies"].append("unnatural_spectral_profile")
            if phase_coherence < 0.3:
                findings["detected_anomalies"].append("phase_inconsistencies")
            if silence_patterns["abnormal"]:
                findings["detected_anomalies"].append("unnatural_silence_patterns")
            
            return synthetic_prob, findings
            
        except Exception as e:
            logger.error(f"Error analyzing audio: {e}")
            raise
    
    def _analyze_spectrogram(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Analyze spectrogram for synthesis artifacts.
        
        Args:
            audio: numpy array with audio samples
            
        Returns:
            Dictionary with spectral features
        """
        try:
            # Compute spectrogram
            S = librosa.feature.melspectrogram(
                y=audio,
                sr=self.sample_rate,
                n_mels=128
            )
            S_db = librosa.power_to_db(S, ref=np.max)
            
            # Compute spectral flatness (Wiener entropy)
            flatness = np.mean(
                np.sum(S_db, axis=1) / (np.max(S_db, axis=1) + 1e-6)
            )
            flatness = min(1.0, flatness / 128.0)  # Normalize
            
            # Compute centroid and bandwidth
            centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
            mean_centroid = np.mean(centroid)
            
            return {
                "flatness": flatness,
                "centroid": float(mean_centroid),
                "centroid_variance": float(np.var(centroid))
            }
        except Exception as e:
            logger.warning(f"Spectrogram analysis failed: {e}")
            return {"flatness": 0.5, "centroid": 0.5, "centroid_variance": 0.5}
    
    def _check_phase_coherence(self, audio: np.ndarray) -> float:
        """
        Check phase coherence in the audio signal.
        
        Args:
            audio: numpy array with audio samples
            
        Returns:
            Phase coherence score (0-1)
        """
        try:
            # Compute STFT
            D = librosa.stft(audio)
            magnitude = np.abs(D)
            phase = np.angle(D)
            
            # Compute phase consistency
            phase_diff = np.diff(phase, axis=1)
            phase_consistency = np.mean(
                np.cos(phase_diff)
            )  # Cosine of phase difference
            
            # Normalize to 0-1
            coherence = (phase_consistency + 1) / 2
            return float(coherence)
        except Exception as e:
            logger.warning(f"Phase coherence check failed: {e}")
            return 0.5
    
    def _analyze_silence_patterns(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Analyze silence patterns for unnatural pauses.
        
        Args:
            audio: numpy array with audio samples
            
        Returns:
            Dictionary with silence analysis
        """
        try:
            # Compute energy frame-by-frame
            frame_length = 2048
            hop_length = 512
            
            S = librosa.feature.melspectrogram(
                y=audio,
                sr=self.sample_rate,
                n_fft=frame_length,
                hop_length=hop_length
            )
            energy = np.sum(S, axis=0)
            
            # Threshold for silence
            threshold = np.percentile(energy, 10)
            silent_frames = energy < threshold
            silence_ratio = np.sum(silent_frames) / len(silent_frames)
            
            # Check for abnormal silence patterns
            # Synthetic audio often has unnatural silence at boundaries
            abnormal = silence_ratio > 0.3  # More than 30% silence is suspicious
            
            return {
                "silence_ratio": float(silence_ratio),
                "abnormal": bool(abnormal)
            }
        except Exception as e:
            logger.warning(f"Silence pattern analysis failed: {e}")
            return {"silence_ratio": 0.0, "abnormal": False}
    
    def analyze(self, audio: np.ndarray) -> AnalysisResult:
        """
        Perform complete audio analysis.
        
        Args:
            audio: numpy array with audio samples
            
        Returns:
            AnalysisResult object
        """
        try:
            confidence, findings = self.analyze_audio_array(audio)
            
            return AnalysisResult(
                analyzer_type=AnalysisType.AUDIO,
                confidence=confidence,
                is_fake=confidence > settings.audio_confidence_threshold,
                findings=findings
            )
        except Exception as e:
            logger.error(f"Audio analysis failed: {e}")
            return AnalysisResult(
                analyzer_type=AnalysisType.AUDIO,
                confidence=0.5,
                is_fake=False,
                findings={},
                error=str(e)
            )
