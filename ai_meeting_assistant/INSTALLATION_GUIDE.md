# AI Meeting Assistant - Installation Guide

## Overview

This guide will help you install and set up the AI Meeting Assistant, a comprehensive real-time meeting analysis tool that provides live transcription, speaker identification, emotion detection, jargon explanations, and meeting summaries.

## System Requirements

### Minimum Requirements
- **Operating System**: Windows 10/11, macOS 10.14+, or Linux (Ubuntu 18.04+)
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 5GB free space for models and dependencies
- **Microphone**: Any USB or built-in microphone
- **Internet**: Required for initial model downloads

### Recommended Requirements
- **RAM**: 16GB for optimal performance
- **GPU**: NVIDIA GPU with CUDA support (optional, for faster processing)
- **CPU**: Multi-core processor (Intel i5/AMD Ryzen 5 or better)

## Installation Steps

### 1. Clone or Download the Project

```bash
# If using Git
git clone <repository-url>
cd ai_meeting_assistant

# Or download and extract the ZIP file
```

### 2. Create Python Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv meeting_assistant_env

# Activate virtual environment
# On Windows:
meeting_assistant_env\Scripts\activate

# On macOS/Linux:
source meeting_assistant_env/bin/activate
```

### 3. Install Dependencies

**Option A: Quick Fix (Recommended for Windows)**
```bash
# Run the automated installation script
python quick_install.py
```

**Option B: Full Automated Installation**
```bash
# Run the comprehensive installation script
python install_dependencies.py
```

**Option C: Manual Installation**
```bash
# Install PyTorch first (CPU version for compatibility)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install PyAudio (Windows)
pip install pipwin
pipwin install pyaudio

# Install remaining dependencies
pip install -r requirements_minimal.txt
```

**Note**: If you encounter PyTorch installation issues, this is usually due to:
- Outdated pip version (run: `python -m pip install --upgrade pip`)
- Python version compatibility
- Network/proxy issues

For PyAudio issues on different platforms:
```bash
# Windows (if pipwin fails):
# Download wheel from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio

# macOS:
brew install portaudio
pip install pyaudio

# Linux (Ubuntu/Debian):
sudo apt-get install portaudio19-dev python3-pyaudio
pip install pyaudio
```

### 4. Run Component Tests

```bash
# Test if all components can be loaded
python test_components.py
```

This will check if all dependencies are properly installed and identify any missing components.

### 5. Download AI Models

```bash
# Download and set up all required AI models
python setup_models.py
```

This process may take 10-30 minutes depending on your internet connection. The script will:
- Download Whisper speech recognition models
- Install spaCy language models
- Download HuggingFace emotion detection models
- Download summarization models
- Set up speaker identification models (may require HuggingFace token)

### 6. Configure Speaker Identification (Optional)

For advanced speaker identification features, you'll need a HuggingFace token:

1. Visit [HuggingFace](https://huggingface.co/) and create an account
2. Go to [pyannote/speaker-diarization](https://huggingface.co/pyannote/speaker-diarization) and accept the user conditions
3. Get your access token from [HuggingFace Settings](https://huggingface.co/settings/tokens)
4. Set the token in your environment:

```bash
# Windows
set HF_TOKEN=your_token_here

# macOS/Linux
export HF_TOKEN=your_token_here
```

Or add it to the `config.py` file created by the setup script.

## Running the Application

### Start the Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

### First-Time Setup

1. **Audio Device Selection**: In the sidebar, select your microphone from the dropdown
2. **Industry Focus**: Choose relevant industries for better jargon detection
3. **Display Settings**: Configure transcript display preferences

### Using the Application

1. **Start Meeting**: Click "🎯 Start Meeting" in the sidebar
2. **Live Features**: 
   - View real-time transcript in the "Live Transcript" tab
   - Monitor emotions in the "Emotions" tab
   - See jargon explanations in the "Jargon" tab
   - Track speakers in the "Speakers" tab
3. **Stop Meeting**: Click "⏹️ Stop Meeting" when done
4. **Generate Summary**: Go to "Summary" tab and click "Generate Meeting Summary"

## Troubleshooting

### Common Issues

#### 1. PyAudio Installation Issues

**Windows**:
```bash
pip install pipwin
pipwin install pyaudio
```

**macOS**:
```bash
brew install portaudio
pip install pyaudio
```

**Linux (Ubuntu/Debian)**:
```bash
sudo apt-get install portaudio19-dev python3-pyaudio
pip install pyaudio
```

#### 2. CUDA/GPU Issues

If you have GPU issues, force CPU usage by setting:
```bash
export CUDA_VISIBLE_DEVICES=""
```

#### 3. Model Download Failures

- Check your internet connection
- Try running `python setup_models.py` again
- For HuggingFace models, ensure you have accepted user agreements

#### 4. Memory Issues

- Close other applications to free up RAM
- Use smaller model sizes in `config.py`:
  ```python
  WHISPER_MODEL_SIZE = "tiny"  # Instead of "base"
  ```

#### 5. Microphone Not Detected

- Check microphone permissions in your OS settings
- Try different USB ports for USB microphones
- Restart the application after connecting microphone

### Performance Optimization

#### For Better Performance:
1. **Use GPU**: Install CUDA and PyTorch with GPU support
2. **Increase RAM**: Close unnecessary applications
3. **SSD Storage**: Store models on SSD for faster loading
4. **Smaller Models**: Use "tiny" or "base" Whisper models instead of "large"

#### For Lower Resource Usage:
1. **Disable Features**: Comment out unused components in `app.py`
2. **Reduce Buffer Sizes**: Modify audio buffer settings in `config.py`
3. **Lower Sample Rate**: Use 8kHz instead of 16kHz for audio

## Configuration

### Audio Settings
Edit `config.py` to modify audio parameters:
```python
AUDIO_SAMPLE_RATE = 16000  # Audio quality
AUDIO_CHUNK_SIZE = 1024    # Buffer size
```

### Model Settings
Choose different model sizes for performance vs. accuracy:
```python
WHISPER_MODEL_SIZE = "base"  # tiny, base, small, medium, large
```

### Processing Settings
Adjust sensitivity thresholds:
```python
CONFIDENCE_THRESHOLD = 0.3  # Lower = more sensitive
MIN_TEXT_LENGTH = 10        # Minimum text length to process
```

## Advanced Features

### Custom Industry Glossaries

Add custom jargon terms by modifying `jargon_detection.py`:

```python
# Add custom glossary
detector.add_custom_glossary(
    name="Custom Industry",
    terms={
        "term1": "Definition 1",
        "term2": "Definition 2"
    },
    keywords={"keyword1", "keyword2"}
)
```

### API Integration

The components can be used programmatically:

```python
from speech_recognition import SpeechRecognizer
from emotion_detection import EmotionDetector

# Initialize components
recognizer = SpeechRecognizer()
emotion_detector = EmotionDetector()

# Set up callbacks
def on_transcription(segment):
    print(f"Transcribed: {segment.text}")
    emotion_detector.add_text(segment.text)

recognizer.add_transcription_callback(on_transcription)

# Start processing
recognizer.start_processing()
emotion_detector.start_processing()
```

## Support and Updates

### Getting Help
1. Check this installation guide
2. Run `python test_components.py` to diagnose issues
3. Check the console output for error messages
4. Ensure all dependencies are properly installed

### Updating the Application
```bash
# Pull latest changes (if using Git)
git pull origin main

# Update dependencies
pip install -r requirements.txt --upgrade

# Re-run setup if needed
python setup_models.py
```

## Security and Privacy

### Data Privacy
- All processing is done locally on your machine
- No audio or transcript data is sent to external servers
- Models run offline after initial download

### Microphone Permissions
- The application requires microphone access
- Grant permissions when prompted by your operating system
- You can revoke permissions at any time in system settings

## Performance Benchmarks

### Typical Performance (on modern hardware):
- **Transcription Latency**: 1-3 seconds
- **Speaker Identification**: 2-5 seconds
- **Emotion Detection**: <1 second
- **Jargon Detection**: <1 second
- **Memory Usage**: 2-8GB depending on models

### Optimization Tips:
- Use smaller models for real-time performance
- Close unnecessary browser tabs and applications
- Use wired internet connection for model downloads
- Consider GPU acceleration for large models

---

**Congratulations!** You should now have a fully functional AI Meeting Assistant. Start a meeting and experience real-time AI-powered meeting insights!
