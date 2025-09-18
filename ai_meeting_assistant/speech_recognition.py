"""
Speech Recognition Module
Real-time speech-to-text using Faster-Whisper
"""

import numpy as np
import threading
import queue
import time
from typing import Optional, List, Dict, Callable
import logging
from faster_whisper import WhisperModel
from dataclasses import dataclass
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TranscriptionSegment:
    """Represents a transcribed speech segment"""
    text: str
    start_time: float
    end_time: float
    confidence: float
    timestamp: datetime
    speaker_id: Optional[str] = None

class SpeechRecognizer:
    def __init__(self, 
                 model_size: str = "base",
                 device: str = "auto",
                 compute_type: str = "auto"):
        """
        Initialize speech recognizer
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large-v2, large-v3)
            device: Device to run on (cpu, cuda, auto)
            compute_type: Computation type (auto, float16, float32, int8)
        """
        self.model_size = model_size
        self.device = device
        
        # Auto-detect best compute type
        if compute_type == "auto":
            self.compute_type = self._detect_compute_type(device)
        else:
            self.compute_type = compute_type
        
        # Initialize model with fallback handling
        logger.info(f"Loading Whisper model: {model_size} (device: {device}, compute_type: {self.compute_type})")
        self.model = self._initialize_model_with_fallback()
        
        # Processing queue and threading
        self.audio_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.is_processing = False
        self.processing_thread = None
        
        # Callbacks
        self.transcription_callbacks = []
        
        # Configuration
        self.min_audio_length = 1.0  # Minimum audio length to process (seconds)
        self.sample_rate = 16000
        
        # Buffer for continuous audio
        self.audio_buffer = []
        self.buffer_duration = 0.0
        self.max_buffer_duration = 30.0  # Maximum buffer duration
    
    def _detect_compute_type(self, device: str) -> str:
        """Auto-detect the best compute type for the device"""
        try:
            import torch
            
            # Check if CUDA is available and device supports it
            if device == "cuda" or (device == "auto" and torch.cuda.is_available()):
                # Check GPU compute capability for float16 support
                if torch.cuda.is_available():
                    gpu_props = torch.cuda.get_device_properties(0)
                    # Modern GPUs (compute capability >= 7.0) support efficient float16
                    if gpu_props.major >= 7:
                        logger.info("GPU supports efficient float16 - using float16")
                        return "float16"
                    else:
                        logger.info("GPU doesn't support efficient float16 - using float32")
                        return "float32"
                else:
                    logger.info("CUDA not available - using CPU with float32")
                    return "float32"
            else:
                # CPU - use float32 for better compatibility
                logger.info("Using CPU - selecting float32 for compatibility")
                return "float32"
                
        except ImportError:
            logger.warning("PyTorch not available - using float32")
            return "float32"
        except Exception as e:
            logger.warning(f"Error detecting compute type: {e} - defaulting to float32")
            return "float32"
    
    def _initialize_model_with_fallback(self):
        """Initialize Whisper model with fallback options"""
        fallback_options = [
            (self.device, self.compute_type),
            ("cpu", "float32"),
            ("cpu", "int8"),
        ]
        
        for device, compute_type in fallback_options:
            try:
                logger.info(f"Attempting to load model with device={device}, compute_type={compute_type}")
                model = WhisperModel(
                    self.model_size,
                    device=device,
                    compute_type=compute_type
                )
                logger.info(f"✅ Successfully loaded Whisper model with device={device}, compute_type={compute_type}")
                
                # Update instance variables with working configuration
                self.device = device
                self.compute_type = compute_type
                
                return model
                
            except Exception as e:
                logger.warning(f"Failed to load model with device={device}, compute_type={compute_type}: {e}")
                continue
        
        # If all options fail, raise the last exception
        raise RuntimeError("Failed to initialize Whisper model with any configuration")
        
    def add_transcription_callback(self, callback: Callable[[TranscriptionSegment], None]):
        """Add callback for transcription results"""
        self.transcription_callbacks.append(callback)
    
    def _notify_transcription_callbacks(self, segment: TranscriptionSegment):
        """Notify all transcription callbacks"""
        for callback in self.transcription_callbacks:
            try:
                callback(segment)
            except Exception as e:
                logger.error(f"Error in transcription callback: {e}")
    
    def add_audio(self, audio_data: np.ndarray):
        """Add audio data for processing"""
        if not self.is_processing:
            return
        
        # Add to buffer
        self.audio_buffer.extend(audio_data)
        self.buffer_duration = len(self.audio_buffer) / self.sample_rate
        
        # If buffer is long enough, queue for processing
        if self.buffer_duration >= self.min_audio_length:
            # Create a copy of current buffer
            audio_copy = np.array(self.audio_buffer, dtype=np.float32)
            self.audio_queue.put((audio_copy, time.time()))
            
            # Keep some overlap for context
            overlap_samples = int(0.5 * self.sample_rate)  # 0.5 second overlap
            if len(self.audio_buffer) > overlap_samples:
                self.audio_buffer = self.audio_buffer[-overlap_samples:]
                self.buffer_duration = len(self.audio_buffer) / self.sample_rate
            else:
                self.audio_buffer = []
                self.buffer_duration = 0.0
    
    def _processing_worker(self):
        """Worker thread for processing audio"""
        logger.info("Speech recognition processing started")
        
        while self.is_processing:
            try:
                # Get audio from queue
                audio_data, timestamp = self.audio_queue.get(timeout=1.0)
                
                # Process with Whisper
                segments = self._transcribe_audio(audio_data, timestamp)
                
                # Send results to callbacks
                for segment in segments:
                    self._notify_transcription_callbacks(segment)
                    self.result_queue.put(segment)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in speech processing: {e}")
    
    def _transcribe_audio(self, audio_data: np.ndarray, timestamp: float) -> List[TranscriptionSegment]:
        """Transcribe audio data using Whisper"""
        try:
            # Ensure audio is the right format
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Normalize audio
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data))
            
            # Transcribe
            segments, info = self.model.transcribe(
                audio_data,
                beam_size=5,
                language="en",  # Can be made configurable
                condition_on_previous_text=False,
                vad_filter=True,  # Voice activity detection
                vad_parameters=dict(min_silence_duration_ms=500)
            )
            
            results = []
            for segment in segments:
                if segment.text.strip():  # Only add non-empty segments
                    transcription_segment = TranscriptionSegment(
                        text=segment.text.strip(),
                        start_time=segment.start,
                        end_time=segment.end,
                        confidence=getattr(segment, 'avg_logprob', 0.0),
                        timestamp=datetime.now()
                    )
                    results.append(transcription_segment)
            
            return results
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return []
    
    def start_processing(self):
        """Start speech recognition processing"""
        if self.is_processing:
            logger.warning("Speech recognition already running")
            return
        
        self.is_processing = True
        self.processing_thread = threading.Thread(target=self._processing_worker)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        logger.info("Speech recognition started")
    
    def stop_processing(self):
        """Stop speech recognition processing"""
        if not self.is_processing:
            return
        
        self.is_processing = False
        
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)
        
        logger.info("Speech recognition stopped")
    
    def get_recent_transcriptions(self, max_count: int = 10) -> List[TranscriptionSegment]:
        """Get recent transcription results"""
        results = []
        try:
            while len(results) < max_count:
                segment = self.result_queue.get_nowait()
                results.append(segment)
        except queue.Empty:
            pass
        
        return results
    
    def transcribe_file(self, audio_file_path: str) -> List[TranscriptionSegment]:
        """Transcribe an audio file"""
        try:
            segments, info = self.model.transcribe(
                audio_file_path,
                beam_size=5,
                language="en"
            )
            
            results = []
            for segment in segments:
                if segment.text.strip():
                    transcription_segment = TranscriptionSegment(
                        text=segment.text.strip(),
                        start_time=segment.start,
                        end_time=segment.end,
                        confidence=getattr(segment, 'avg_logprob', 0.0),
                        timestamp=datetime.now()
                    )
                    results.append(transcription_segment)
            
            return results
            
        except Exception as e:
            logger.error(f"Error transcribing file {audio_file_path}: {e}")
            return []

class TranscriptionManager:
    """Manages transcription history and provides search/filtering capabilities"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.transcriptions: List[TranscriptionSegment] = []
        self.lock = threading.Lock()
    
    def add_transcription(self, segment: TranscriptionSegment):
        """Add a transcription segment"""
        with self.lock:
            self.transcriptions.append(segment)
            
            # Maintain max history
            if len(self.transcriptions) > self.max_history:
                self.transcriptions = self.transcriptions[-self.max_history:]
    
    def get_recent_transcriptions(self, 
                                duration_minutes: float = 5.0) -> List[TranscriptionSegment]:
        """Get transcriptions from the last N minutes"""
        with self.lock:
            cutoff_time = datetime.now().timestamp() - (duration_minutes * 60)
            
            recent = []
            for segment in reversed(self.transcriptions):
                if segment.timestamp.timestamp() >= cutoff_time:
                    recent.append(segment)
                else:
                    break
            
            return list(reversed(recent))
    
    def search_transcriptions(self, query: str, 
                            max_results: int = 10) -> List[TranscriptionSegment]:
        """Search transcriptions for specific text"""
        with self.lock:
            query_lower = query.lower()
            results = []
            
            for segment in reversed(self.transcriptions):
                if query_lower in segment.text.lower():
                    results.append(segment)
                    if len(results) >= max_results:
                        break
            
            return results
    
    def get_full_transcript(self, 
                          start_time: Optional[datetime] = None,
                          end_time: Optional[datetime] = None) -> str:
        """Get full transcript as text"""
        with self.lock:
            segments = self.transcriptions
            
            if start_time:
                segments = [s for s in segments if s.timestamp >= start_time]
            if end_time:
                segments = [s for s in segments if s.timestamp <= end_time]
            
            return " ".join([segment.text for segment in segments])
    
    def clear_history(self):
        """Clear transcription history"""
        with self.lock:
            self.transcriptions.clear()

if __name__ == "__main__":
    # Test speech recognition
    recognizer = SpeechRecognizer(model_size="base")
    manager = TranscriptionManager()
    
    def on_transcription(segment: TranscriptionSegment):
        print(f"[{segment.timestamp.strftime('%H:%M:%S')}] {segment.text}")
        manager.add_transcription(segment)
    
    recognizer.add_transcription_callback(on_transcription)
    
    print("Speech recognition test - speak into your microphone...")
    recognizer.start_processing()
    
    # Simulate audio input (in real usage, this would come from AudioCapture)
    try:
        time.sleep(10)  # Run for 10 seconds
    except KeyboardInterrupt:
        pass
    
    recognizer.stop_processing()
    
    # Show recent transcriptions
    recent = manager.get_recent_transcriptions(1.0)
    print(f"\nCaptured {len(recent)} transcription segments")
    
    full_transcript = manager.get_full_transcript()
    print(f"Full transcript: {full_transcript}")
