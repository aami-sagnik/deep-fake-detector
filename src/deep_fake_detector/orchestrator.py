"""
Gemma 4 Orchestrator Agent
Orchestrates specialist models and synthesizes findings into forensic reports.
"""

import json
from typing import Dict, Any, List, Optional
from openai import OpenAI

from deep_fake_detector.logger import logger
from deep_fake_detector.config import settings
from deep_fake_detector.models import AnalysisResult, AggregatedReport


class GemmaOrchestrator:
    """
    Gemma 4 agent that orchestrates specialist models and reasons about findings.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Gemma orchestrator.
        
        Args:
            api_key: Google Generative AI API key (unused, kept for compatibility)
        """
        self.ollama_host = settings.ollama_host
        self.ollama_model = settings.ollama_model
        self.ollama_timeout = settings.ollama_timeout
        
        # Configure OpenAI client to use Ollama
        api_base = self.ollama_host.rstrip('/') + '/v1'
        self.client = OpenAI(api_key="not-needed", base_url=api_base)
        
        logger.info(f"Using Ollama at {self.ollama_host} with model {self.ollama_model}")
        
        self.system_prompt = self._create_system_prompt()
    
    def _create_system_prompt(self) -> str:
        """
        Create the system prompt for Gemma.
        
        Returns:
            System prompt string
        """
        return """You are a Chief Forensic Investigator specializing in deep-fake detection.

Your role is to:
1. Analyze detailed findings from specialist micro-expert models
2. Identify conflicting signals and anomalies
3. Apply forensic reasoning to synthesize a final verdict
4. Provide clear, actionable recommendations

Specialist models you work with:
- Visual Specialist: Detects artifacts, blending lines, frame inconsistencies
- Audio Specialist: Identifies voice synthesis and acoustic anomalies
- Biometric Specialist: Analyzes facial landmarks, blinking, pulse consistency
- Metadata Specialist: Checks encoding, container info, and known flagged content

Your reasoning process:
1. Weight evidence based on specialist reliability
2. Look for corroborating signals across multiple specialists
3. Identify contradictions and resolve them
4. Assign final confidence scores
5. Provide specific recommendations for further investigation

Output format:
- Reasoning: Clear explanation of key findings and their significance
- Verdict: REAL, FAKE, or UNCERTAIN with confidence
- Recommendations: Specific next steps for investigation
"""
    
    def orchestrate(
        self,
        specialist_results: Dict[str, AnalysisResult],
        media_metadata: Dict[str, Any]
    ) -> AggregatedReport:
        """
        Orchestrate specialist findings and generate forensic report.
        
        Args:
            specialist_results: Dictionary of specialist analysis results
            media_metadata: Metadata about the analyzed media
            
        Returns:
            AggregatedReport with synthesized findings
        """
        try:
            logger.info("Orchestrating specialist findings")
            
            # Prepare specialist summaries
            specialist_summary = self._prepare_specialist_summary(specialist_results)
            
            # Get Ollama reasoning
            reasoning = self._get_reasoning(specialist_summary)
            
            # Calculate aggregated confidence
            overall_confidence = self._calculate_aggregate_confidence(
                specialist_results
            )
            
            # Determine verdict
            verdict = self._determine_verdict(overall_confidence, specialist_results)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                specialist_results,
                verdict,
                reasoning
            )
            
            # Create report
            report = AggregatedReport(
                overall_confidence=overall_confidence,
                verdict=verdict,
                specialist_results=specialist_results,
                reasoning=reasoning,
                recommendations=recommendations,
                metadata=media_metadata
            )
            
            logger.info(f"Report generated: Verdict={verdict}, Confidence={overall_confidence:.2%}")
            return report
            
        except Exception as e:
            logger.error(f"Orchestration failed: {e}")
            # Return default uncertain report
            return AggregatedReport(
                overall_confidence=0.5,
                verdict="UNCERTAIN",
                specialist_results=specialist_results,
                reasoning=f"Error during orchestration: {str(e)}",
                recommendations=["Manual review recommended"],
                metadata=media_metadata
            )
    
    def _prepare_specialist_summary(
        self,
        specialist_results: Dict[str, AnalysisResult]
    ) -> str:
        """
        Prepare specialist findings summary for Gemma.
        
        Args:
            specialist_results: Dictionary of specialist results
            
        Returns:
            Formatted summary string
        """
        summary = "SPECIALIST FINDINGS:\n\n"
        
        for name, result in specialist_results.items():
            summary += f"## {name.upper()} ({result.analyzer_type.value})\n"
            summary += f"Confidence: {result.confidence:.2%}\n"
            summary += f"Is Fake: {result.is_fake}\n"
            summary += f"Key Findings:\n"
            
            findings = result.findings
            for key, value in findings.items():
                if key == "detected_anomalies" and isinstance(value, list):
                    summary += f"  - Anomalies: {', '.join(value)}\n"
                elif not isinstance(value, dict):
                    summary += f"  - {key}: {value}\n"
            
            if result.error:
                summary += f"  - Error: {result.error}\n"
            
            summary += "\n"
        
        return summary
    
    def _get_reasoning(self, specialist_summary: str) -> str:
        """
        Get reasoning from Ollama via OpenAI-compatible API.
        
        Args:
            specialist_summary: Summary of specialist findings
            
        Returns:
            Reasoning text
        """
        prompt = f"""{specialist_summary}

Based on these specialist findings, provide your forensic analysis:
1. Which signals are most reliable?
2. Are there any contradictions?
3. What is your overall assessment?
4. What confidence level would you assign?

Keep your response concise but comprehensive."""

        try:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ]

            resp = self.client.chat.completions.create(
                model=self.ollama_model,
                messages=messages,
                temperature=0.3,
                max_tokens=1000,
                timeout=self.ollama_timeout,
            )

            # Extract text from response (new API returns structured object)
            reasoning = resp.choices[0].message.content.strip() if resp.choices else ""
            logger.info("Ollama reasoning generated successfully")
            return reasoning

        except Exception as e:
            logger.error(f"Failed to get reasoning from Ollama: {e}")
            raise
    
    
    def _calculate_aggregate_confidence(
        self,
        specialist_results: Dict[str, AnalysisResult]
    ) -> float:
        """
        Calculate weighted aggregate confidence from specialists.
        
        Args:
            specialist_results: Dictionary of specialist results
            
        Returns:
            Weighted confidence score (0-1)
        """
        # Weights for each specialist
        weights = {
            "visual": 0.35,
            "audio": 0,
            "biometric": 0,
            "metadata": 0,
            # "audio": 0.30,
            # "biometric": 0.25,
            # "metadata": 0.10,
        }
        
        weighted_sum = 0.0
        total_weight = 0.0

        print("RESULTS:", specialist_results)
        
        for name, result in specialist_results.items():
            # Infer type from result
            analyzer_type = result.analyzer_type.value
            weight = weights.get(analyzer_type, 0.25)
            
            # Only include if no error
            if not result.error:
                weighted_sum += result.confidence * weight
                total_weight += weight
        
        # Normalize
        if total_weight > 0:
            confidence = weighted_sum / total_weight
        else:
            confidence = 0.5
        
        return float(confidence)
    
    def _determine_verdict(
        self,
        confidence: float,
        specialist_results: Dict[str, AnalysisResult]
    ) -> str:
        """
        Determine overall verdict based on confidence and specialist agreement.
        
        Args:
            confidence: Overall confidence score
            specialist_results: Dictionary of specialist results
            
        Returns:
            Verdict string: "REAL", "FAKE", or "UNCERTAIN"
        """
        # Count specialist opinions
        fake_count = sum(
            1 for r in specialist_results.values()
            if r.is_fake and not r.error
        )
        total_valid = sum(
            1 for r in specialist_results.values()
            if not r.error
        )
        
        # Determine verdict
        if confidence > 0.5:
            verdict = "FAKE"
        elif confidence < 0.3:
            verdict = "REAL"
        else:
            # Check specialist agreement
            if total_valid > 0 and fake_count / total_valid > 0.5:
                verdict = "FAKE"
            elif total_valid > 0 and fake_count / total_valid < 0.3:
                verdict = "REAL"
            else:
                verdict = "UNCERTAIN"
        
        return verdict
    
    def _generate_recommendations(
        self,
        specialist_results: Dict[str, AnalysisResult],
        verdict: str,
        reasoning: str
    ) -> List[str]:
        """
        Generate recommendations for further investigation.
        
        Args:
            specialist_results: Dictionary of specialist results
            verdict: Overall verdict
            reasoning: Gemma's reasoning
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if verdict == "FAKE":
            recommendations.append("Content shows signs of manipulation")
            
            # Add specific recommendations based on specialists
            for name, result in specialist_results.items():
                if result.is_fake and not result.error:
                    if "anomalies" in result.analyzer_type.value:
                        recommendations.append(
                            f"Further investigation of {name} anomalies recommended"
                        )
        
        elif verdict == "UNCERTAIN":
            recommendations.append("Content requires human review")
            recommendations.append("Consider cross-reference with known databases")
            recommendations.append("Analyze with higher quality samples if available")
        
        else:  # REAL
            recommendations.append("Content appears authentic")
            recommendations.append("No major manipulation indicators detected")
        
        # Add general recommendations
        recommendations.append("Document source and distribution chain")
        recommendations.append("Perform reverse image search for similar content")
        
        return recommendations
