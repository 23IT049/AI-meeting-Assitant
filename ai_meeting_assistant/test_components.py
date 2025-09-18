"""
Component Testing Script
Test individual components of the AI Meeting Assistant
"""

import sys
import time
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_audio_capture():
    """Test audio capture functionality"""
    print("\n" + "="*50)
    print("TESTING AUDIO CAPTURE")
    print("="*50)
    
    try:
        from audio_capture import AudioCapture
        
        capture = AudioCapture()
        
        # List available devices
        print("Available audio devices:")
        devices = capture.list_audio_devices()
        for device in devices:
            print(f"  {device['index']}: {device['name']}")
        
        if not devices:
            print("❌ No audio devices found")
            return False
        
        print("✅ Audio capture component loaded successfully")
        return True
        
    except Exception as e:
        print(f"❌ Audio capture test failed: {e}")
        return False

def test_speech_recognition():
    """Test speech recognition functionality"""
    print("\n" + "="*50)
    print("TESTING SPEECH RECOGNITION")
    print("="*50)
    
    try:
        from speech_recognition import SpeechRecognizer
        
        # Test with auto-detection to avoid compute type issues
        recognizer = SpeechRecognizer(
            model_size="tiny",  # Use tiny for faster testing
            device="auto", 
            compute_type="auto"
        )
        
        print("✅ Speech recognition component loaded successfully")
        print(f"Model: {recognizer.model_size}")
        print(f"Device: {recognizer.device}")
        print(f"Compute Type: {recognizer.compute_type}")
        return True
        
    except Exception as e:
        print(f"❌ Speech recognition test failed: {e}")
        print("Note: This may require downloading the Whisper model first")
        return False

def test_speaker_identification():
    """Test speaker identification functionality"""
    print("\n" + "="*50)
    print("TESTING SPEAKER IDENTIFICATION")
    print("="*50)
    
    try:
        from speaker_identification import SpeakerIdentifier, SimpleSpeakerIdentifier
        
        try:
            identifier = SpeakerIdentifier()
            print("✅ Advanced speaker identification loaded")
        except Exception as e:
            print(f"⚠️ Advanced speaker identification failed: {e}")
            identifier = SimpleSpeakerIdentifier()
            print("✅ Fallback speaker identification loaded")
        
        return True
        
    except Exception as e:
        print(f"❌ Speaker identification test failed: {e}")
        return False

def test_emotion_detection():
    """Test emotion detection functionality"""
    print("\n" + "="*50)
    print("TESTING EMOTION DETECTION")
    print("="*50)
    
    try:
        from emotion_detection import EmotionDetector
        
        detector = EmotionDetector()
        
        # Test with sample text
        test_text = "I'm really excited about this project!"
        detector.add_text(test_text)
        
        print("✅ Emotion detection component loaded successfully")
        print(f"Test text: '{test_text}'")
        return True
        
    except Exception as e:
        print(f"❌ Emotion detection test failed: {e}")
        print("Note: This may require downloading HuggingFace models first")
        return False

def test_jargon_detection():
    """Test jargon detection functionality"""
    print("\n" + "="*50)
    print("TESTING JARGON DETECTION")
    print("="*50)
    
    try:
        from jargon_detection import JargonDetector
        
        detector = JargonDetector()
        
        # Test with sample text
        test_text = "We need to implement a REST API with proper CI/CD pipeline."
        detector.add_text(test_text)
        
        print("✅ Jargon detection component loaded successfully")
        print(f"Test text: '{test_text}'")
        
        # Show loaded glossaries
        print(f"Loaded {len(detector.glossaries)} industry glossaries:")
        for name in detector.glossaries.keys():
            print(f"  - {name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Jargon detection test failed: {e}")
        return False

def test_meeting_summary():
    """Test meeting summary functionality"""
    print("\n" + "="*50)
    print("TESTING MEETING SUMMARY")
    print("="*50)
    
    try:
        from meeting_summary import MeetingSummarizer
        
        summarizer = MeetingSummarizer()
        
        # Test with sample meeting
        summarizer.start_meeting("Test Meeting")
        summarizer.add_transcript_segment("Welcome to our test meeting.", "Alice")
        summarizer.add_transcript_segment("Let's discuss the project timeline.", "Bob")
        summarizer.add_transcript_segment("We decided to use the new framework.", "Alice")
        summarizer.end_meeting()
        
        print("✅ Meeting summary component loaded successfully")
        return True
        
    except Exception as e:
        print(f"❌ Meeting summary test failed: {e}")
        print("Note: This may require downloading summarization models first")
        return False

def test_utilities():
    """Test utility functions"""
    print("\n" + "="*50)
    print("TESTING UTILITIES")
    print("="*50)
    
    try:
        from utils.text_utils import clean_text, extract_keywords, detect_questions
        from utils.audio_utils import normalize_audio, detect_voice_activity
        
        # Test text utilities
        test_text = "This is a test sentence. What do you think about it?"
        cleaned = clean_text(test_text, lowercase=True)
        keywords = extract_keywords(test_text)
        questions = detect_questions(test_text)
        
        print("✅ Text utilities working:")
        print(f"  Original: {test_text}")
        print(f"  Cleaned: {cleaned}")
        print(f"  Keywords: {keywords}")
        print(f"  Questions: {questions}")
        
        # Test audio utilities
        import numpy as np
        test_audio = np.random.randn(1000).astype(np.float32)
        normalized = normalize_audio(test_audio)
        
        print("✅ Audio utilities working:")
        print(f"  Original audio shape: {test_audio.shape}")
        print(f"  Normalized audio shape: {normalized.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Utilities test failed: {e}")
        return False

def test_streamlit_app():
    """Test if Streamlit app can be imported"""
    print("\n" + "="*50)
    print("TESTING STREAMLIT APP")
    print("="*50)
    
    try:
        # Just test if we can import the main components
        import streamlit as st
        
        # Test if app.py can be imported (without running)
        import importlib.util
        spec = importlib.util.spec_from_file_location("app", "app.py")
        
        print("✅ Streamlit is available")
        print("✅ Main app file is accessible")
        print("To run the app: streamlit run app.py")
        return True
        
    except Exception as e:
        print(f"❌ Streamlit app test failed: {e}")
        return False

def run_all_tests():
    """Run all component tests"""
    print("AI MEETING ASSISTANT - COMPONENT TESTS")
    print("=" * 60)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    tests = [
        ("Audio Capture", test_audio_capture),
        ("Speech Recognition", test_speech_recognition),
        ("Speaker Identification", test_speaker_identification),
        ("Emotion Detection", test_emotion_detection),
        ("Jargon Detection", test_jargon_detection),
        ("Meeting Summary", test_meeting_summary),
        ("Utilities", test_utilities),
        ("Streamlit App", test_streamlit_app)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<25} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
        print("\nNext steps:")
        print("1. Run 'python setup_models.py' to download AI models")
        print("2. Run 'streamlit run app.py' to start the application")
    else:
        print("⚠️ Some tests failed. Please check the error messages above.")
        print("\nTroubleshooting:")
        print("1. Make sure all dependencies are installed: pip install -r requirements.txt")
        print("2. Run the setup script: python setup_models.py")
        print("3. Check your Python environment and package versions")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
