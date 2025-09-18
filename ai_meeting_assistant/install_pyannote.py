"""
PyAnnote Installation Script
Handles the installation of pyannote.audio with proper dependencies
"""

import subprocess
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_command(cmd, description=""):
    """Run command and return success status"""
    try:
        logger.info(f"🔄 {description or cmd}")
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        logger.info(f"✅ Success: {description or cmd}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed: {description or cmd}")
        logger.error(f"Error: {e.stderr}")
        return False

def install_pyannote():
    """Install pyannote.audio and its dependencies"""
    logger.info("=== Installing pyannote.audio ===")
    
    # Step 1: Install PyTorch if not available
    logger.info("Step 1: Ensuring PyTorch is installed...")
    if not run_command("python -c \"import torch; print('PyTorch available')\"", "Checking PyTorch"):
        logger.info("Installing PyTorch...")
        if not run_command("pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu", "Installing PyTorch"):
            logger.error("Failed to install PyTorch. Please install manually.")
            return False
    
    # Step 2: Install audio processing dependencies
    logger.info("Step 2: Installing audio dependencies...")
    audio_deps = [
        "soundfile",
        "librosa", 
        "scipy",
        "numpy"
    ]
    
    for dep in audio_deps:
        run_command(f"pip install {dep}", f"Installing {dep}")
    
    # Step 3: Install pyannote.audio
    logger.info("Step 3: Installing pyannote.audio...")
    if run_command("pip install pyannote.audio", "Installing pyannote.audio"):
        logger.info("✅ pyannote.audio installed successfully!")
        
        # Test the installation
        logger.info("Testing pyannote.audio installation...")
        test_code = """
try:
    from pyannote.audio import Pipeline
    print("✅ pyannote.audio import successful")
except Exception as e:
    print(f"❌ pyannote.audio import failed: {e}")
"""
        run_command(f"python -c \"{test_code}\"", "Testing pyannote import")
        
        return True
    else:
        logger.error("❌ Failed to install pyannote.audio")
        return False

def main():
    """Main installation function"""
    logger.info("PyAnnote.audio Installation Helper")
    logger.info("=" * 40)
    
    success = install_pyannote()
    
    if success:
        logger.info("\n🎉 Installation completed!")
        logger.info("\nNote: For advanced speaker diarization features, you may need:")
        logger.info("1. Accept user conditions at: https://huggingface.co/pyannote/speaker-diarization")
        logger.info("2. Get HuggingFace token from: https://huggingface.co/settings/tokens")
        logger.info("3. Set token: export HF_TOKEN=your_token_here")
        logger.info("\nThe system will work with basic speaker identification even without the token.")
    else:
        logger.error("\n❌ Installation failed!")
        logger.info("\nDon't worry! The meeting assistant will still work with:")
        logger.info("- Basic speaker identification (voice activity detection)")
        logger.info("- All other features (transcription, emotions, jargon, summaries)")
        logger.info("\nYou can run the app without pyannote.audio")
    
    logger.info("\nNext steps:")
    logger.info("1. Run: python test_components.py")
    logger.info("2. Run: streamlit run app.py")

if __name__ == "__main__":
    main()
