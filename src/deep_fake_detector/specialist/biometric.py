"""
Biometric Inconsistency Tool
Detects physiological anomalies using facial landmarks and rPPG analysis.
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

from deep_fake_detector.logger import logger
from deep_fake_detector.models import AnalysisResult, AnalysisType
from deep_fake_detector.config import settings


class BiometricSpecialist:
    """
    Analyzes facial biometric features for inconsistencies.
    Uses MediaPipe for facial landmark detection and rPPG analysis.
    """
    
    def __init__(self):
        """Initialize the biometric specialist."""
        if not MEDIAPIPE_AVAILABLE:
            logger.warning("MediaPipe not available, using mock biometric analysis")
            self.face_detector = None
            self.face_mesh = None
        else:
            mp_face_detection = mp.solutions.face_detection
            mp_face_mesh = mp.solutions.face_mesh
            self.face_detector = mp_face_detection.FaceDetection(
                model_selection=1,
                min_detection_confidence=0.5
            )
            self.face_mesh = mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                min_detection_confidence=0.5
            )
        logger.info("Biometric specialist initialized")
    
    def analyze_frames(self, frames: List[np.ndarray]) -> Tuple[float, Dict[str, Any]]:
        """
        Analyze frames for biometric inconsistencies.
        
        Args:
            frames: List of video frames (BGR numpy arrays)
            
        Returns:
            Tuple of (confidence_score, findings_dict)
        """
        if not frames:
            return 0.0, {"error": "No frames provided"}
        
        try:
            blink_data = []
            landmark_stability = []
            pulse_data = []
            
            if not MEDIAPIPE_AVAILABLE:
                return self._mock_analysis(len(frames))
            
            # Process each frame
            for frame in frames:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Face detection
                detection_result = self.face_detector.process(rgb_frame)
                if not detection_result.detections:
                    continue
                
                # Face mesh (landmarks)
                mesh_result = self.face_mesh.process(rgb_frame)
                if mesh_result.multi_face_landmarks:
                    landmarks = mesh_result.multi_face_landmarks[0]
                    
                    # Extract features
                    blink_ratio = self._calculate_blink_ratio(landmarks)
                    blink_data.append(blink_ratio)
                    
                    # Landmark stability
                    stability = self._check_landmark_stability(landmarks)
                    landmark_stability.append(stability)
                    
                    # rPPG analysis
                    face_region = self._extract_face_region(frame, detection_result.detections[0])
                    if face_region is not None:
                        pulse = self._analyze_rppg(face_region)
                        pulse_data.append(pulse)
            
            # Aggregate results
            findings = self._aggregate_biometric_findings(
                blink_data,
                landmark_stability,
                pulse_data
            )
            
            # Calculate confidence
            confidence = self._calculate_biometric_confidence(findings)
            
            return confidence, findings
            
        except Exception as e:
            logger.error(f"Error analyzing frames: {e}")
            raise
    
    def _mock_analysis(self, frame_count: int) -> Tuple[float, Dict[str, Any]]:
        """Provide mock analysis when MediaPipe is not available."""
        findings = {
            "blink_data": {
                "count": frame_count,
                "mean": 0.15,
                "std": 0.05,
                "abnormal": False
            },
            "landmark_stability": {
                "mean": 0.92,
                "consistent": True
            },
            "pulse_detection": {
                "detected_frames": int(frame_count * 0.8),
                "mean_strength": 0.65,
                "reliable": True
            },
            "detected_anomalies": [],
            "note": "Mock analysis (MediaPipe not available)"
        }
        return 0.25, findings
    
    def _calculate_blink_ratio(self, landmarks) -> float:
        """
        Calculate blink ratio from eye landmarks.
        
        Args:
            landmarks: MediaPipe face landmarks
            
        Returns:
            Blink ratio (0-1)
        """
        try:
            # Eye landmarks indices
            LEFT_EYE = [362, 385, 387, 263, 373, 380]
            RIGHT_EYE = [33, 160, 158, 133, 153, 144]
            
            left_eye_points = np.array([
                [landmarks.landmark[i].x, landmarks.landmark[i].y]
                for i in LEFT_EYE
            ])
            right_eye_points = np.array([
                [landmarks.landmark[i].x, landmarks.landmark[i].y]
                for i in RIGHT_EYE
            ])
            
            # Calculate eye aspect ratio (EAR)
            def calculate_ear(eye_points):
                distances = np.linalg.norm(
                    np.diff(eye_points[[1, 2, 3, 4]], axis=0),
                    axis=1
                )
                mean_dist = np.mean(distances)
                horizontal_dist = np.linalg.norm(eye_points[0] - eye_points[3])
                return mean_dist / (horizontal_dist + 1e-6)
            
            left_ear = calculate_ear(left_eye_points)
            right_ear = calculate_ear(right_eye_points)
            
            return float((left_ear + right_ear) / 2)
        except Exception as e:
            logger.warning(f"Blink ratio calculation failed: {e}")
            return 0.5
    
    def _check_landmark_stability(self, landmarks) -> float:
        """
        Check stability of facial landmarks.
        
        Args:
            landmarks: MediaPipe face landmarks
            
        Returns:
            Stability score (0-1)
        """
        try:
            # Sample multiple key landmarks
            key_indices = [0, 17, 21, 54, 58, 103, 130, 133, 152, 159, 172, 263]
            points = np.array([
                [landmarks.landmark[i].x, landmarks.landmark[i].y]
                for i in key_indices
            ])
            
            # Stability is high when landmarks are consistent across frames
            # This is a single-frame check, so we estimate based on coherence
            centroid = np.mean(points, axis=0)
            distances = np.linalg.norm(points - centroid, axis=1)
            
            # Coefficient of variation
            cv = np.std(distances) / (np.mean(distances) + 1e-6)
            stability = 1.0 - min(1.0, cv)
            
            return float(stability)
        except Exception as e:
            logger.warning(f"Landmark stability check failed: {e}")
            return 0.5
    
    def _extract_face_region(self, frame: np.ndarray, detection) -> Optional[np.ndarray]:
        """
        Extract face region for rPPG analysis.
        
        Args:
            frame: Input frame
            detection: Face detection result
            
        Returns:
            Face region or None
        """
        try:
            h, w = frame.shape[:2]
            bbox = detection.location_data.relative_bounding_box
            
            x1 = max(0, int(bbox.xmin * w))
            y1 = max(0, int(bbox.ymin * h))
            x2 = min(w, int((bbox.xmin + bbox.width) * w))
            y2 = min(h, int((bbox.ymin + bbox.height) * h))
            
            return frame[y1:y2, x1:x2]
        except Exception as e:
            logger.warning(f"Face region extraction failed: {e}")
            return None
    
    def _analyze_rppg(self, face_region: np.ndarray) -> float:
        """
        Analyze remote photoplethysmography (rPPG) for pulse detection.
        
        Args:
            face_region: Cropped face region
            
        Returns:
            rPPG pulse signal strength (0-1)
        """
        try:
            # Convert to green channel (highest PPG signal)
            green_channel = face_region[:, :, 1].astype(np.float32)
            
            # Compute mean signal
            mean_signal = np.mean(green_channel)
            std_signal = np.std(green_channel)
            
            # Signal-to-noise ratio as proxy for pulse strength
            snr = mean_signal / (std_signal + 1e-6)
            
            # Normalize
            pulse_strength = min(1.0, snr / 100.0)
            
            return float(pulse_strength)
        except Exception as e:
            logger.warning(f"rPPG analysis failed: {e}")
            return 0.5
    
    def _aggregate_biometric_findings(
        self,
        blink_data: List[float],
        landmark_stability: List[float],
        pulse_data: List[float]
    ) -> Dict[str, Any]:
        """
        Aggregate biometric findings.
        
        Args:
            blink_data: List of blink ratios
            landmark_stability: List of stability scores
            pulse_data: List of pulse strengths
            
        Returns:
            Dictionary with aggregated findings
        """
        findings = {
            "blink_data": {
                "count": len(blink_data),
                "mean": float(np.mean(blink_data)) if blink_data else 0.0,
                "std": float(np.std(blink_data)) if blink_data else 0.0,
                "abnormal": len(blink_data) > 0 and np.mean(blink_data) > 0.3
            },
            "landmark_stability": {
                "mean": float(np.mean(landmark_stability)) if landmark_stability else 0.5,
                "consistent": len(landmark_stability) > 0 and np.std(landmark_stability) < 0.2
            },
            "pulse_detection": {
                "detected_frames": len(pulse_data),
                "mean_strength": float(np.mean(pulse_data)) if pulse_data else 0.0,
                "reliable": len(pulse_data) > 0 and np.mean(pulse_data) > 0.3
            },
            "detected_anomalies": []
        }
        
        # Detect anomalies
        if findings["blink_data"]["abnormal"]:
            findings["detected_anomalies"].append("abnormal_blink_rate")
        if not findings["landmark_stability"]["consistent"]:
            findings["detected_anomalies"].append("unstable_landmarks")
        if not findings["pulse_detection"]["reliable"]:
            findings["detected_anomalies"].append("weak_pulse_signal")
        
        return findings
    
    def _calculate_biometric_confidence(self, findings: Dict[str, Any]) -> float:
        """
        Calculate confidence score from biometric findings.
        
        Args:
            findings: Biometric findings dictionary
            
        Returns:
            Confidence score (0-1)
        """
        scores = []
        
        # Blink analysis
        if findings["blink_data"]["abnormal"]:
            scores.append(0.7)  # Abnormal blinking suggests fakery
        else:
            scores.append(0.3)
        
        # Landmark stability
        if findings["landmark_stability"]["consistent"]:
            scores.append(0.3)
        else:
            scores.append(0.7)  # Unstable landmarks suggest fakery
        
        # Pulse detection
        if findings["pulse_detection"]["reliable"]:
            scores.append(0.2)
        else:
            scores.append(0.6)  # Weak pulse suggests fake
        
        return float(np.mean(scores))
    
    def analyze(self, frames: List[np.ndarray]) -> AnalysisResult:
        """
        Perform complete biometric analysis.
        
        Args:
            frames: List of video frames
            
        Returns:
            AnalysisResult object
        """
        try:
            confidence, findings = self.analyze_frames(frames)
            
            return AnalysisResult(
                analyzer_type=AnalysisType.BIOMETRIC,
                confidence=confidence,
                is_fake=confidence > settings.biometric_confidence_threshold,
                findings=findings
            )
        except Exception as e:
            logger.error(f"Biometric analysis failed: {e}")
            return AnalysisResult(
                analyzer_type=AnalysisType.BIOMETRIC,
                confidence=0.5,
                is_fake=False,
                findings={},
                error=str(e)
            )
