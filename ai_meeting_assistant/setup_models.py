"""
Model Setup Script
Downloads and prepares all required AI models for the meeting assistant
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def install_spacy_model():
    """Install spaCy English model"""
    try:
        logger.info("Installing spaCy English model...")
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
        logger.info("✓ spaCy model installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ Failed to install spaCy model: {e}")
        return False

def download_whisper_model():
    """Download Faster-Whisper model"""
    try:
        logger.info("Downloading Whisper model (this may take a while)...")
        from faster_whisper import WhisperModel
        
        # Download base model
        model = WhisperModel("base", device="cpu", compute_type="float32")
        logger.info("✓ Whisper model downloaded successfully")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to download Whisper model: {e}")
        return False

def setup_pyannote_models():
    """Setup pyannote.audio models"""
    try:
        logger.info("Setting up pyannote.audio models...")
        logger.info("Note: You may need to accept user conditions at https://huggingface.co/pyannote/speaker-diarization")
        
        # Try to load the pipeline to trigger download
        from pyannote.audio import Pipeline
        
        try:
            pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
            logger.info("✓ pyannote speaker diarization model ready")
        except Exception as e:
            logger.warning(f"Speaker diarization model not available: {e}")
            logger.info("You may need to:")
            logger.info("1. Visit https://huggingface.co/pyannote/speaker-diarization")
            logger.info("2. Accept the user conditions")
            logger.info("3. Get an access token from https://huggingface.co/settings/tokens")
            logger.info("4. Set the token in your environment: export HF_TOKEN=your_token")
        
        return True
    except Exception as e:
        logger.error(f"✗ Failed to setup pyannote models: {e}")
        return False

def download_huggingface_models():
    """Download HuggingFace models for emotion detection and summarization"""
    try:
        logger.info("Downloading HuggingFace models...")
        
        from transformers import pipeline
        
        # Sentiment analysis model
        logger.info("Downloading sentiment analysis model...")
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest"
        )
        logger.info("✓ Sentiment analysis model ready")
        
        # Emotion detection model
        logger.info("Downloading emotion detection model...")
        emotion_pipeline = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base"
        )
        logger.info("✓ Emotion detection model ready")
        
        # Summarization model
        logger.info("Downloading summarization model...")
        summarization_pipeline = pipeline(
            "summarization",
            model="facebook/bart-large-cnn"
        )
        logger.info("✓ Summarization model ready")
        
        return True
    except Exception as e:
        logger.error(f"✗ Failed to download HuggingFace models: {e}")
        return False

def create_config_file():
    """Create configuration file"""
    config_content = """# AI Meeting Assistant Configuration

# Audio Settings
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHUNK_SIZE = 1024
AUDIO_CHANNELS = 1

# Model Settings
WHISPER_MODEL_SIZE = "base"  # tiny, base, small, medium, large-v2, large-v3
WHISPER_DEVICE = "auto"  # cpu, cuda, auto
WHISPER_COMPUTE_TYPE = "float16"  # float16, float32, int8

# Sentiment Analysis
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# Emotion Detection
EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"

# Summarization
SUMMARIZATION_MODEL = "facebook/bart-large-cnn"

# Speaker Identification
SPEAKER_MIN_SPEAKERS = 1
SPEAKER_MAX_SPEAKERS = 10

# Processing Settings
MIN_TEXT_LENGTH = 10
MAX_TEXT_LENGTH = 512
CONFIDENCE_THRESHOLD = 0.3

# HuggingFace Token (optional, for pyannote models)
# HF_TOKEN = "your_huggingface_token_here"
"""
    
    try:
        with open("config.py", "w") as f:
            f.write(config_content)
        logger.info("✓ Configuration file created")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to create config file: {e}")
        return False

def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = [
        "streamlit",
        "pyaudio",
        "faster-whisper",
        "torch",
        "transformers",
        "spacy",
        "keybert",
        "pyannote.audio",
        "numpy",
        "pandas",
        "plotly",
        "wikipedia"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            logger.info(f"✓ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            logger.warning(f"✗ {package} is missing")
    
    if missing_packages:
        logger.error(f"Missing packages: {', '.join(missing_packages)}")
        logger.info("Please install missing packages with: pip install -r requirements.txt")
        return False
    
    return True

def main():
    """Main setup function"""
    logger.info("=== AI Meeting Assistant Setup ===")
    logger.info("This script will download and prepare all required models.")
    logger.info("This may take several minutes and require internet connection.")
    
    # Check dependencies
    logger.info("\n1. Checking dependencies...")
    if not check_dependencies():
        logger.error("Please install missing dependencies first.")
        return False
    
    # Create config file
    logger.info("\n2. Creating configuration file...")
    create_config_file()
    
    # Install spaCy model
    logger.info("\n3. Installing spaCy model...")
    install_spacy_model()
    
    # Download Whisper model
    logger.info("\n4. Downloading Whisper model...")
    download_whisper_model()
    
    # Setup pyannote models
    logger.info("\n5. Setting up speaker identification models...")
    setup_pyannote_models()
    
    # Download HuggingFace models
    logger.info("\n6. Downloading emotion and summarization models...")
    download_huggingface_models()
    
    logger.info("\n=== Setup Complete ===")
    logger.info("You can now run the application with: streamlit run app.py")
    logger.info("\nNote: For speaker identification, you may need to:")
    logger.info("1. Visit https://huggingface.co/pyannote/speaker-diarization")
    logger.info("2. Accept the user conditions")
    logger.info("3. Get an access token and set it in config.py")
    
    return True

if __name__ == "__main__":
    main()
