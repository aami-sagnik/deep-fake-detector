"""
Visual Deep-fake Specialist
Uses pretrained HuggingFace deepfake detector.
"""

import cv2
import numpy as np
import torch

from PIL import Image
from typing import List, Dict, Any, Tuple, Optional

from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification
)

from deep_fake_detector.logger import logger
from deep_fake_detector.models import (
    AnalysisResult,
    AnalysisType
)
from deep_fake_detector.config import settings


class VisualSpecialist:
    """
    Visual deepfake analysis using pretrained
    HuggingFace deepfake detector.
    """

    def __init__(
        self,
        model_name: str = (
            "buildborderless/CommunityForensics-DeepfakeDet-ViT"
        )
    ):

        self.device = torch.device(settings.device)

        logger.info(
            f"Loading visual model: {model_name}"
        )

        # HuggingFace processor
        self.processor = (
            AutoImageProcessor.from_pretrained(
                model_name
            )
        )

        # Classification model
        self.model = (
            AutoModelForImageClassification
            .from_pretrained(model_name)
            .to(self.device)
        )

        self.model.eval()

        # OpenCV Haar cascade face detector
        self.face_cascade = (
            cv2.CascadeClassifier(
                cv2.data.haarcascades +
                "haarcascade_frontalface_default.xml"
            )
        )

        logger.info("Visual specialist loaded")

    def detect_face(
        self,
        frame: np.ndarray
    ) -> Optional[np.ndarray]:

        try:

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(40, 40)
            )

            if len(faces) == 0:
                return None

            # Select largest face
            faces = sorted(
                faces,
                key=lambda x: x[2] * x[3],
                reverse=True
            )

            x, y, w, h = faces[0]

            face = frame[y:y+h, x:x+w]

            if face.size == 0:
                return None

            return face

        except Exception as e:

            logger.warning(
                f"Face detection failed: {e}"
            )

            return None

    def analyze_image(
        self,
        image: Image.Image
    ) -> Tuple[float, Dict[str, Any]]:

        try:

            # Force model input resolution
            image = image.resize((384, 384))

            self.processor.size = {
                "height": 384,
                "width": 384
            }

            self.processor.crop_size = {
                "height": 384,
                "width": 384
            }

            inputs = self.processor(
                images=image,
                return_tensors="pt"
            )

            inputs = {
                k: v.to(self.device)
                for k, v in inputs.items()
            }

            with torch.no_grad():

                outputs = self.model(**inputs)

                logits = outputs.logits

                probs = torch.softmax(
                    logits,
                    dim=-1
                )

            probs = probs[0].cpu().numpy()

            labels = self.model.config.id2label

            predictions = []

            fake_score = 0.0
            real_score = 0.0

            for idx, prob in enumerate(probs):

                label = labels[idx]

                predictions.append({
                    "label": label,
                    "score": float(prob)
                })

                # CommunityForensics mapping
                # LABEL_0 = REAL
                # LABEL_1 = FAKE

                if label == "LABEL_1":
                    fake_score = float(prob)

                elif label == "LABEL_0":
                    real_score = float(prob)

            findings = {
                "fake_score": fake_score,
                "real_score": real_score,
                "raw_predictions": predictions,
            }

            logger.info(
                f"Fake score: {fake_score:.4f}"
            )

            return fake_score, findings

        except Exception as e:

            logger.error(
                f"Image analysis failed: {e}"
            )

            raise

    def analyze_frames(
        self,
        frames: List[np.ndarray]
    ) -> Tuple[float, Dict[str, Any]]:

        if not frames:

            return 0.0, {
                "error": "No frames provided"
            }

        try:

            scores = []

            # Sample every 10th frame
            sampled_frames = frames[::10]

            logger.info(
                f"Analyzing "
                f"{len(sampled_frames)} frames"
            )

            for idx, frame in enumerate(sampled_frames):

                face = self.detect_face(frame)

                if face is None:

                    logger.warning(
                        f"No face detected "
                        f"in frame {idx}"
                    )

                    continue

                logger.info(
                    f"Face shape: {face.shape}"
                )

                rgb = cv2.cvtColor(
                    face,
                    cv2.COLOR_BGR2RGB
                )

                image = Image.fromarray(rgb)

                score, _ = self.analyze_image(image)

                scores.append(score)

            if len(scores) == 0:

                return 0.5, {
                    "error": "No faces detected"
                }

            avg_score = float(np.mean(scores))

            frame_std = (
                float(np.std(scores))
                if len(scores) > 1
                else 0.0
            )

            findings = {
                "frames_analyzed": len(scores),
                "average_fake_score": avg_score,
                "score_std": frame_std,
                "frame_scores": scores,
            }

            return avg_score, findings

        except Exception as e:

            logger.error(
                f"Frame analysis failed: {e}"
            )

            raise

    def analyze(
        self,
        frames: List[np.ndarray]
    ) -> AnalysisResult:

        try:

            confidence, findings = (
                self.analyze_frames(frames)
            )

            return AnalysisResult(
                analyzer_type=AnalysisType.VISUAL,
                confidence=confidence,
                is_fake=(
                    confidence >
                    settings.visual_confidence_threshold
                ),
                findings=findings
            )

        except Exception as e:

            logger.error(
                f"Visual analysis failed: {e}"
            )

            return AnalysisResult(
                analyzer_type=AnalysisType.VISUAL,
                confidence=0.5,
                is_fake=False,
                findings={},
                error=str(e)
            )