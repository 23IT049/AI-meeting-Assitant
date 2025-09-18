"""
Compute Type Fix Script
Resolves float16 compute type issues by testing different configurations
"""

import logging
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_whisper_configurations():
    """Test different Whisper model configurations"""
    logger.info("🔧 Testing Whisper Model Configurations")
    logger.info("=" * 50)
    
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        logger.error("❌ faster-whisper not installed. Run: pip install faster-whisper")
        return False
    
    # Test configurations in order of preference
    test_configs = [
        ("cpu", "float32", "Most compatible - CPU with float32"),
        ("cpu", "int8", "CPU with int8 (smaller memory)"),
        ("auto", "float32", "Auto device with float32"),
        ("cuda", "float32", "GPU with float32 (if available)"),
        ("cuda", "float16", "GPU with float16 (if supported)"),
    ]
    
    working_configs = []
    
    for device, compute_type, description in test_configs:
        try:
            logger.info(f"🧪 Testing: {description}")
            model = WhisperModel(
                "tiny",  # Use tiny model for faster testing
                device=device,
                compute_type=compute_type
            )
            logger.info(f"✅ SUCCESS: {description}")
            working_configs.append((device, compute_type, description))
            
            # Clean up
            del model
            
        except Exception as e:
            logger.warning(f"❌ FAILED: {description} - {str(e)[:100]}...")
    
    # Report results
    logger.info("\n" + "=" * 50)
    logger.info("📊 TEST RESULTS")
    logger.info("=" * 50)
    
    if working_configs:
        logger.info(f"✅ Found {len(working_configs)} working configurations:")
        for i, (device, compute_type, desc) in enumerate(working_configs, 1):
            logger.info(f"  {i}. {desc}")
            logger.info(f"     Device: {device}, Compute Type: {compute_type}")
        
        # Recommend best configuration
        best_config = working_configs[0]
        logger.info(f"\n🎯 RECOMMENDED CONFIGURATION:")
        logger.info(f"   Device: {best_config[0]}")
        logger.info(f"   Compute Type: {best_config[1]}")
        logger.info(f"   Description: {best_config[2]}")
        
        return True
    else:
        logger.error("❌ No working configurations found!")
        logger.info("\nTroubleshooting steps:")
        logger.info("1. Ensure faster-whisper is installed: pip install faster-whisper")
        logger.info("2. Try installing CPU-only PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cpu")
        logger.info("3. Check available disk space (models need ~1GB)")
        return False

def test_speech_recognizer():
    """Test the updated SpeechRecognizer class"""
    logger.info("\n🎤 Testing Updated Speech Recognizer")
    logger.info("=" * 50)
    
    try:
        from speech_recognition import SpeechRecognizer
        
        # Test with auto-detection
        logger.info("Testing SpeechRecognizer with auto-detection...")
        recognizer = SpeechRecognizer(
            model_size="tiny",  # Use tiny for faster testing
            device="auto",
            compute_type="auto"
        )
        
        logger.info(f"✅ SpeechRecognizer initialized successfully!")
        logger.info(f"   Final device: {recognizer.device}")
        logger.info(f"   Final compute type: {recognizer.compute_type}")
        logger.info(f"   Model size: {recognizer.model_size}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ SpeechRecognizer test failed: {e}")
        return False

def main():
    """Main testing function"""
    logger.info("🚀 Compute Type Fix and Test Script")
    logger.info("This script will test different Whisper configurations")
    logger.info("to find the best one for your system.\n")
    
    # Test 1: Raw Whisper configurations
    whisper_ok = test_whisper_configurations()
    
    # Test 2: Updated SpeechRecognizer class
    if whisper_ok:
        recognizer_ok = test_speech_recognizer()
        
        if recognizer_ok:
            logger.info("\n🎉 ALL TESTS PASSED!")
            logger.info("Your system is ready to run the AI Meeting Assistant.")
            logger.info("\nNext steps:")
            logger.info("1. Run: streamlit run app.py")
            logger.info("2. The app will automatically use the best configuration")
        else:
            logger.error("\n⚠️ SpeechRecognizer test failed, but basic Whisper works.")
            logger.info("The app should still work with manual configuration.")
    else:
        logger.error("\n❌ No working Whisper configurations found.")
        logger.info("Please check the troubleshooting steps above.")
    
    return whisper_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
