# Deep-Fake Detection System

A comprehensive deep-fake detection system using Gemma 4 as an orchestrator with specialized "micro-expert" models for visual, audio, biometric, and metadata analysis.

## Features

- **Modular Architecture**: Specialized models for different analysis dimensions
- **Gemma 4 Orchestration**: LLM-based reasoning and synthesis of findings
- **Multi-Modal Analysis**:
  - Visual: Artifact detection, blending anomalies, frame consistency
  - Audio: Voice synthesis detection, acoustic anomalies
  - Biometric: Facial landmarks, blink rate, pulse analysis
  - Metadata: Container integrity, encoding analysis, known flagged content
- **Gradio Web UI**: Easy-to-use interface for analysis
- **Comprehensive Reporting**: Detailed forensic reports with confidence scores and recommendations

## Architecture

```
User Interface (Gradio)
         ↓
Media Processing Pipeline (FFmpeg)
         ↓
Specialist Models (in parallel)
  ├── Visual Specialist
  ├── Audio Specialist
  ├── Biometric Specialist
  └── Metadata Specialist
         ↓
Gemma 4 Orchestrator
         ↓
Aggregation & Reporting Engine
         ↓
Forensic Report (JSON/Markdown)
```

## Installation

### Prerequisites

- Python 3.10+
- FFmpeg (for media processing)
- GPU recommended (CUDA/cuDNN for faster processing)

### Setup

1. Clone the repository:
```bash
cd c:\Users\Sagnik\ Ghosh\Projects\deep-fake-detector
```

2. Install using `uv`:
```bash
uv sync
```

3. Copy and configure environment variables:
```bash
cp .env.example .env
# Edit .env with your Google API key
```

4. Install system dependencies:
```bash
# On Windows: Download FFmpeg from https://ffmpeg.org/download.html
# On macOS:
brew install ffmpeg

# On Linux (Ubuntu/Debian):
sudo apt-get install ffmpeg
```

## Usage

### Web UI (Recommended)

```bash
uv run python -m deep_fake_detector.main --ui
```

Then open http://localhost:7860 in your browser.

### Command Line

Analyze a single file:
```bash
uv run python -m deep_fake_detector.main --analyze path/to/video.mp4
```

### Python API

```python
from deep_fake_detector import DeepFakeAnalyzer

analyzer = DeepFakeAnalyzer()
report = analyzer.analyze_file("path/to/video.mp4")

print(f"Verdict: {report.verdict}")
print(f"Confidence: {report.overall_confidence:.2%}")
print(f"Reasoning: {report.reasoning}")
```

## Configuration

Edit `.env` to customize:

- `GOOGLE_API_KEY`: Your Google Generative AI API key (required for Gemma 4)
- `DEVICE`: `cuda` or `cpu` (GPU recommended)
- `MAX_VIDEO_DURATION`: Maximum video length to process (seconds)
- `FRAME_SAMPLE_RATE`: Process every Nth frame
- `AUDIO_SAMPLE_RATE`: Audio analysis sample rate (Hz)
- Threshold values for each specialist (0-1)

## Models Used

- **Visual Analysis**: Vision Transformer (ViT-base-patch16-224)
- **Audio Analysis**: Wav2Vec2 (facebook/wav2vec2-base-960h)
- **Biometric Analysis**: MediaPipe Face Detection and Face Mesh
- **Metadata Analysis**: FFprobe
- **Orchestration**: Gemini 1.5 Pro (via Google Generative AI)

## Report Structure

Each analysis generates a report with:

```json
{
  "verdict": "REAL|FAKE|UNCERTAIN",
  "overall_confidence": 0.85,
  "specialist_results": {
    "visual": { "confidence": 0.80, "findings": {...} },
    "audio": { "confidence": 0.90, "findings": {...} },
    "biometric": { "confidence": 0.75, "findings": {...} },
    "metadata": { "confidence": 0.60, "findings": {...} }
  },
  "reasoning": "Detailed forensic reasoning from Gemma 4",
  "recommendations": ["Follow-up actions"],
  "metadata": "Input file metadata"
}
```

## Development

### Running Tests

```bash
uv run pytest tests/
```

### Development Dependencies

```bash
uv sync --extra dev
```

### Code Quality

```bash
# Format code
uv run black src/

# Lint
uv run ruff check src/

# Type checking
uv run mypy src/
```

## Troubleshooting

### FFmpeg not found
- Install FFmpeg from https://ffmpeg.org/
- Ensure it's in your system PATH

### Model download issues
- Models are cached in `~/.cache/deep-fake-detector/`
- First run will download models (~5-10GB)
- Ensure sufficient disk space

### GPU not detected
- Install PyTorch with CUDA support
- Check NVIDIA driver version
- Set `DEVICE=cpu` in `.env` to use CPU

### API Key issues
- Get your API key from https://ai.google.dev/
- Ensure it has Generative AI API access
- Check `.env` is properly configured

## Limitations

- Analysis quality depends on media quality
- Large files (>10 mins) may take significant time
- Some edge cases may require manual review
- Model accuracy varies by deepfake technique

## Security & Privacy

- Uploaded files are processed locally
- API calls to Gemini only send summarized findings (no raw video/audio)
- No data is stored permanently
- Consider privacy implications when analyzing sensitive content

## Contributing

Contributions welcome! Areas for improvement:
- Additional specialist models
- Performance optimization
- Enhanced UI features
- Expanded test coverage

## License

MIT License - see LICENSE file

## Citation

If you use this system in research, please cite:
```
@software{deepfake_detector_2024,
  title={Deep-Fake Detection System with Gemma Orchestration},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/deep-fake-detector}
}
```

## Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check existing documentation
- Review logs in `~/.data/deep-fake-detector/`

## Disclaimer

This system is provided for research and educational purposes. While it employs advanced deep-fake detection techniques, it should not be considered 100% accurate. Always conduct human review of critical findings.
