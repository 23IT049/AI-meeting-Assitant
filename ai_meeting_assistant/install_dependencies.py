"""
Dependency Installation Script
Handles PyTorch and other dependencies with proper platform detection
"""

import sys
import subprocess
import platform
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_command(command, description=""):
    """Run a command and handle errors"""
    try:
        logger.info(f"Running: {description or command}")
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        logger.info(f"✅ Success: {description or command}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed: {description or command}")
        logger.error(f"Error: {e.stderr}")
        return False

def detect_system():
    """Detect system information"""
    system_info = {
        'os': platform.system().lower(),
        'architecture': platform.machine().lower(),
        'python_version': sys.version_info
    }
    
    logger.info(f"Detected system: {system_info['os']} {system_info['architecture']}")
    logger.info(f"Python version: {sys.version}")
    
    return system_info

def install_pytorch(system_info):
    """Install PyTorch based on system"""
    logger.info("Installing PyTorch...")
    
    # Check if CUDA is available (optional)
    cuda_available = False
    try:
        result = subprocess.run("nvidia-smi", shell=True, capture_output=True)
        cuda_available = result.returncode == 0
    except:
        pass
    
    if cuda_available:
        logger.info("CUDA detected - installing PyTorch with CUDA support")
        pytorch_command = "pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118"
    else:
        logger.info("No CUDA detected - installing CPU-only PyTorch")
        pytorch_command = "pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu"
    
    return run_command(pytorch_command, "PyTorch installation")

def install_pyaudio(system_info):
    """Install PyAudio with platform-specific handling"""
    logger.info("Installing PyAudio...")
    
    if system_info['os'] == 'windows':
        # Try pipwin first for Windows
        logger.info("Attempting Windows-specific PyAudio installation...")
        if run_command("pip install pipwin", "Installing pipwin"):
            if run_command("pipwin install pyaudio", "Installing PyAudio via pipwin"):
                return True
        
        # Fallback to regular pip
        logger.info("Trying regular pip installation...")
        return run_command("pip install pyaudio", "PyAudio via pip")
    
    elif system_info['os'] == 'darwin':  # macOS
        logger.info("macOS detected - installing PortAudio first...")
        run_command("brew install portaudio", "Installing PortAudio via Homebrew")
        return run_command("pip install pyaudio", "PyAudio installation")
    
    else:  # Linux
        logger.info("Linux detected - installing system dependencies...")
        run_command("sudo apt-get update", "Updating package list")
        run_command("sudo apt-get install -y portaudio19-dev python3-pyaudio", "Installing PortAudio")
        return run_command("pip install pyaudio", "PyAudio installation")

def install_other_dependencies():
    """Install remaining dependencies"""
    logger.info("Installing other dependencies...")
    
    # Core dependencies (excluding torch and pyaudio)
    core_deps = [
        "streamlit>=1.28.0",
        "faster-whisper>=0.9.0",
        "transformers>=4.30.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0"
    ]
    
    for dep in core_deps:
        run_command(f"pip install '{dep}'", f"Installing {dep}")
    
    # NLP dependencies
    nlp_deps = [
        "spacy>=3.6.0",
        "keybert>=0.8.0",
        "sentence-transformers>=2.2.0"
    ]
    
    for dep in nlp_deps:
        run_command(f"pip install '{dep}'", f"Installing {dep}")
    
    # Audio processing
    audio_deps = [
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "scipy>=1.11.0"
    ]
    
    for dep in audio_deps:
        run_command(f"pip install '{dep}'", f"Installing {dep}")
    
    # Visualization
    viz_deps = [
        "plotly>=5.15.0",
        "matplotlib>=3.7.0"
    ]
    
    for dep in viz_deps:
        run_command(f"pip install '{dep}'", f"Installing {dep}")
    
    # Utilities
    util_deps = [
        "wikipedia>=1.4.0",
        "requests>=2.31.0",
        "python-dotenv>=1.0.0",
        "pydub>=0.25.0"
    ]
    
    for dep in util_deps:
        run_command(f"pip install '{dep}'", f"Installing {dep}")

def install_optional_dependencies():
    """Install optional dependencies that might fail"""
    logger.info("Installing optional dependencies...")
    
    # Try to install pyannote.audio (might need HuggingFace token)
    if not run_command("pip install pyannote.audio>=3.1.0", "pyannote.audio (optional)"):
        logger.warning("⚠️ pyannote.audio installation failed - speaker identification will use fallback")
    
    # Try to install datasets
    run_command("pip install datasets>=2.14.0", "datasets (optional)")

def main():
    """Main installation function"""
    logger.info("=== AI Meeting Assistant Dependency Installation ===")
    
    # Detect system
    system_info = detect_system()
    
    # Check Python version
    if sys.version_info < (3, 8):
        logger.error("❌ Python 3.8 or higher is required")
        return False
    
    # Upgrade pip first
    run_command("python -m pip install --upgrade pip", "Upgrading pip")
    
    # Install PyTorch
    if not install_pytorch(system_info):
        logger.error("❌ PyTorch installation failed")
        logger.info("You can try manual installation:")
        logger.info("Visit: https://pytorch.org/get-started/locally/")
        return False
    
    # Install PyAudio
    if not install_pyaudio(system_info):
        logger.error("❌ PyAudio installation failed")
        logger.info("Manual installation options:")
        if system_info['os'] == 'windows':
            logger.info("1. Download wheel from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio")
            logger.info("2. Install with: pip install downloaded_wheel.whl")
        return False
    
    # Install other dependencies
    install_other_dependencies()
    
    # Install optional dependencies
    install_optional_dependencies()
    
    logger.info("=== Installation Summary ===")
    logger.info("✅ Core dependencies installed")
    logger.info("✅ PyTorch installed")
    logger.info("✅ PyAudio installed")
    
    logger.info("\nNext steps:")
    logger.info("1. Run: python test_components.py")
    logger.info("2. Run: python setup_models.py")
    logger.info("3. Run: streamlit run app.py")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        logger.error("\n❌ Installation failed. Please check the errors above.")
        sys.exit(1)
    else:
        logger.info("\n🎉 Installation completed successfully!")
        sys.exit(0)
