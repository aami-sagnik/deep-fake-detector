"""
Gradio Web Interface
User-facing interface for uploading media and viewing deep-fake analysis results.
"""

import gradio as gr
import json
from pathlib import Path
from typing import Optional, Dict, Any

from deep_fake_detector.logger import logger
from deep_fake_detector.config import settings
from deep_fake_detector.analyzer import DeepFakeAnalyzer


class GradioApp:
    """Gradio web interface for deep-fake detection."""
    
    def __init__(self):
        """Initialize the Gradio app."""
        self.analyzer = DeepFakeAnalyzer()
        logger.info("Gradio app initialized")
    
    def analyze_media(
        self,
        file_input: Optional[str]
    ) -> Dict[str, Any]:
        """
        Analyze uploaded media file.
        
        Args:
            file_input: Path to uploaded file
            
        Returns:
            Dictionary with analysis results
        """
        if not file_input:
            return {"error": "No file provided"}
        
        try:
            logger.info(f"Analyzing media: {file_input}")
            
            # Perform analysis
            report = self.analyzer.analyze_file(file_input)
            
            # Format results
            results = {
                "verdict": report.verdict,
                "overall_confidence": f"{report.overall_confidence:.2%}",
                "reasoning": report.reasoning,
                "recommendations": "\n".join(report.recommendations),
                "detailed_results": json.dumps(
                    report.to_dict(),
                    indent=2
                )
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return {"error": f"Analysis failed: {str(e)}"}
    
    def build_interface(self):
        """Build the Gradio interface."""
        with gr.Blocks(
            title="Deep-Fake Detection System",
            theme=gr.themes.Soft()
        ) as demo:
            gr.Markdown("""
            # 🔍 Deep-Fake Detection System
            
            Upload a video or image to analyze for signs of deepfake manipulation.
            The system uses specialized AI models to detect visual, audio, biometric,
            and metadata anomalies.
            """)
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Upload Media")
                    file_input = gr.File(
                        label="Upload Video or Image",
                        file_count="single",
                        file_types=["video", "image"]
                    )
                    analyze_btn = gr.Button(
                        "🔍 Analyze",
                        variant="primary",
                        scale=2
                    )
                
                with gr.Column():
                    gr.Markdown("### Quick Info")
                    gr.Markdown("""
                    **Supported formats:**
                    - Video: MP4, AVI, MOV, MKV
                    - Image: JPG, PNG, BMP
                    
                    **Processing time:** 1-5 minutes
                    """)
            
            with gr.Row():
                verdict_output = gr.Textbox(
                    label="Verdict",
                    interactive=False
                )
                confidence_output = gr.Textbox(
                    label="Confidence",
                    interactive=False
                )
            
            reasoning_output = gr.Textbox(
                label="Analysis Reasoning",
                interactive=False,
                lines=5
            )
            
            recommendations_output = gr.Textbox(
                label="Recommendations",
                interactive=False,
                lines=4
            )
            
            with gr.Row():
                gr.Markdown("### Detailed Results")
            
            detailed_output = gr.Textbox(
                label="Full Report (JSON)",
                interactive=False,
                lines=10
            )
            
            # Set up analysis callback
            def on_analyze(file_obj):
                """Handle analysis button click."""
                if not file_obj:
                    return ("", "", "Please upload a file", "", "")
                
                results = self.analyze_media(file_obj.name)
                
                return (
                    results.get("verdict", "ERROR"),
                    results.get("overall_confidence", "N/A"),
                    results.get("reasoning", results.get("error", "No reasoning")),
                    results.get("recommendations", ""),
                    results.get("detailed_results", "")
                )
            
            analyze_btn.click(
                on_analyze,
                inputs=[file_input],
                outputs=[
                    verdict_output,
                    confidence_output,
                    reasoning_output,
                    recommendations_output,
                    detailed_output
                ]
            )
            
            # Error output
            error_output = gr.Textbox(
                label="Messages",
                interactive=False,
                lines=2
            )
            
            gr.Markdown("""
            ---
            
            ### About This System
            
            This deep-fake detection system uses a modular orchestrator architecture with
            specialized "micro-expert" models:
            
            1. **Visual Specialist**: Detects artifacts, blending lines, and frame inconsistencies
            2. **Audio Specialist**: Identifies voice synthesis and acoustic anomalies  
            3. **Biometric Specialist**: Analyzes facial landmarks and physiological consistency
            4. **Metadata Specialist**: Checks encoding, container info, and file integrity
            
            All findings are synthesized by Gemma 4 for comprehensive forensic analysis.
            """)
        
        return demo
    
    def launch(
        self,
        server_name: Optional[str] = None,
        server_port: Optional[int] = None,
        share: bool = False
    ):
        """
        Launch the Gradio app.
        
        Args:
            server_name: Server hostname
            server_port: Server port
            share: Whether to create a public link
        """
        server_name = server_name or settings.gradio_server_name
        server_port = server_port or settings.gradio_server_port
        share = share or settings.gradio_share
        
        demo = self.build_interface()
        
        logger.info(f"Launching Gradio app on {server_name}:{server_port}")
        demo.launch(
            server_name=server_name,
            server_port=server_port,
            share=share
        )


def main():
    """Main entry point for Gradio app."""
    app = GradioApp()
    app.launch()


if __name__ == "__main__":
    main()
