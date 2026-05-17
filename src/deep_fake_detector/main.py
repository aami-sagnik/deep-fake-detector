"""
Main entry point for the deep-fake detection system.
"""

import sys
import argparse
from pathlib import Path

from deep_fake_detector.logger import logger
from deep_fake_detector.analyzer import DeepFakeAnalyzer
from deep_fake_detector.gradio_app import GradioApp


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Deep-Fake Detection System"
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Launch web UI"
    )
    parser.add_argument(
        "--analyze",
        type=str,
        help="Analyze a media file"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port for web UI"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host for web UI"
    )
    
    args = parser.parse_args()
    
    try:
        if args.ui:
            # Launch Gradio UI
            logger.info("Launching web UI")
            app = GradioApp()
            app.launch(
                server_name=args.host,
                server_port=args.port
            )
        
        elif args.analyze:
            # Analyze a single file
            file_path = Path(args.analyze)
            if not file_path.exists():
                logger.error(f"File not found: {args.analyze}")
                sys.exit(1)
            
            logger.info(f"Analyzing: {args.analyze}")
            analyzer = DeepFakeAnalyzer()
            report = analyzer.analyze_file(str(file_path))
            
            # Print report
            print("\n" + "="*50)
            print(f"VERDICT: {report.verdict}")
            print(f"CONFIDENCE: {report.overall_confidence:.2%}")
            print("="*50)
            print(f"\nREASONING:\n{report.reasoning}")
            print(f"\nRECOMMENDATIONS:")
            for i, rec in enumerate(report.recommendations, 1):
                print(f"  {i}. {rec}")
            print(f"\nFULL REPORT:\n{report.to_json()}")
        
        else:
            # Default: launch UI
            logger.info("No arguments provided, launching UI...")
            app = GradioApp()
            app.launch(
                server_name=args.host,
                server_port=args.port
            )
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
