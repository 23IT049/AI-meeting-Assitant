# AI-Driven Meeting Assistant

A live AI-powered meeting assistant that provides real-time transcription, speaker identification, emotion detection, jargon explanations, and actionable insights during meetings.

## Features

- **Live Transcription**: Real-time speech-to-text using Faster-Whisper
- **Speaker Identification**: Automatic speaker detection with pyannote.audio
- **Emotion Detection**: Real-time sentiment analysis using HuggingFace models
- **Jargon Detection**: Instant technical term explanations with spaCy + KeyBERT
- **Live Dashboard**: Streamlit-based real-time interface
- **Meeting Summaries**: Automated actionable takeaways generation
- **Industry Customization**: Adaptable for various industries

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download required models:
```bash
python setup_models.py
```

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Configure your audio input device
3. Start recording your meeting
4. View real-time insights on the dashboard

## Components

- `app.py` - Main Streamlit application
- `audio_capture.py` - Audio recording functionality
- `speech_recognition.py` - Real-time transcription
- `speaker_identification.py` - Speaker detection
- `emotion_detection.py` - Sentiment analysis
- `jargon_detection.py` - Technical term identification
- `meeting_summary.py` - Summary generation
- `utils/` - Utility functions and helpers

## Requirements

- Python 3.8+
- Microphone access
- 4GB+ RAM recommended
- GPU optional (for faster processing)

## Supported Platforms

- Windows
- macOS
- Linux

Compatible with any meeting platform (Zoom, Teams, Google Meet, etc.)
