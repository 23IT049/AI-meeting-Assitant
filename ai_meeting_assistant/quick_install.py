"""
Quick Installation Fix
Resolves PyTorch installation issues on Windows
"""

import subprocess
import sys

def run_cmd(cmd, description):
    """Run command and show result"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} - SUCCESS")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - FAILED")
        print(f"Error: {e.stderr}")
        return False

def main():
    print("🚀 Quick Fix for AI Meeting Assistant Installation")
    print("=" * 50)
    
    # Step 1: Upgrade pip
    run_cmd("python -m pip install --upgrade pip", "Upgrading pip")
    
    # Step 2: Install PyTorch CPU version (most compatible)
    print("\n📦 Installing PyTorch (CPU version for compatibility)...")
    pytorch_cmd = "pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu"
    if run_cmd(pytorch_cmd, "PyTorch CPU installation"):
        print("✅ PyTorch installed successfully!")
    else:
        print("❌ PyTorch installation failed. Trying alternative...")
        # Try older version
        alt_cmd = "pip install torch==1.13.1 torchaudio==0.13.1"
        run_cmd(alt_cmd, "PyTorch alternative version")
    
    # Step 3: Install PyAudio (Windows-specific)
    print("\n🎤 Installing PyAudio...")
    if run_cmd("pip install pipwin", "Installing pipwin helper"):
        if run_cmd("pipwin install pyaudio", "PyAudio via pipwin"):
            print("✅ PyAudio installed successfully!")
        else:
            print("⚠️ Trying alternative PyAudio installation...")
            run_cmd("pip install pyaudio", "PyAudio direct installation")
    
    # Step 4: Install core dependencies
    print("\n📚 Installing core dependencies...")
    core_packages = [
        "streamlit",
        "faster-whisper",
        "transformers",
        "numpy",
        "pandas",
        "plotly",
        "requests",
        "wikipedia"
    ]
    
    for package in core_packages:
        run_cmd(f"pip install {package}", f"Installing {package}")
    
    # Step 5: Install NLP packages
    print("\n🧠 Installing NLP packages...")
    nlp_packages = [
        "spacy",
        "keybert", 
        "sentence-transformers"
    ]
    
    for package in nlp_packages:
        run_cmd(f"pip install {package}", f"Installing {package}")
    
    # Step 6: Install audio processing
    print("\n🔊 Installing audio processing packages...")
    audio_packages = [
        "librosa",
        "soundfile",
        "scipy"
    ]
    
    for package in audio_packages:
        run_cmd(f"pip install {package}", f"Installing {package}")
    
    print("\n" + "=" * 50)
    print("🎉 Quick installation completed!")
    print("\nNext steps:")
    print("1. Run: python test_components.py")
    print("2. If tests pass, run: python setup_models.py")
    print("3. Then run: streamlit run app.py")
    print("\nIf you still have issues, try:")
    print("- python install_dependencies.py (for full automated install)")
    print("- Check INSTALLATION_GUIDE.md for troubleshooting")

if __name__ == "__main__":
    main()
