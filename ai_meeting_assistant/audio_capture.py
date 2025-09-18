"""
Audio Capture Module
Handles real-time audio recording from microphone using pyaudio
"""

import pyaudio
import numpy as np
import threading
import queue
import time
from typing import Optional, Callable
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioCapture:
    def __init__(self, 
                 sample_rate: int = 16000,
                 chunk_size: int = 1024,
                 channels: int = 1,
                 format: int = pyaudio.paInt16):
        """
        Initialize audio capture
        
        Args:
            sample_rate: Audio sample rate (Hz)
            chunk_size: Number of frames per buffer
            channels: Number of audio channels (1 for mono, 2 for stereo)
            format: Audio format
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.format = format
        
        self.audio = pyaudio.PyAudio()
        self.stream: Optional[pyaudio.Stream] = None
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.callbacks = []
        
    def list_audio_devices(self):
        """List available audio input devices"""
        devices = []
        for i in range(self.audio.get_device_count()):
            device_info = self.audio.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0:
                devices.append({
                    'index': i,
                    'name': device_info['name'],
                    'channels': device_info['maxInputChannels'],
                    'sample_rate': device_info['defaultSampleRate']
                })
        return devices
    
    def add_callback(self, callback: Callable[[np.ndarray], None]):
        """Add callback function to be called with audio data"""
        self.callbacks.append(callback)
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Internal callback for pyaudio stream"""
        if status:
            logger.warning(f"Audio callback status: {status}")
        
        # Convert audio data to numpy array
        audio_data = np.frombuffer(in_data, dtype=np.int16)
        
        # Normalize to float32
        audio_data = audio_data.astype(np.float32) / 32768.0
        
        # Add to queue for processing
        self.audio_queue.put(audio_data)
        
        # Call registered callbacks
        for callback in self.callbacks:
            try:
                callback(audio_data)
            except Exception as e:
                logger.error(f"Error in audio callback: {e}")
        
        return (None, pyaudio.paContinue)
    
    def start_recording(self, device_index: Optional[int] = None):
        """Start audio recording"""
        if self.is_recording:
            logger.warning("Already recording")
            return
        
        try:
            self.stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )
            
            self.stream.start_stream()
            self.is_recording = True
            logger.info("Audio recording started")
            
        except Exception as e:
            logger.error(f"Failed to start audio recording: {e}")
            raise
    
    def stop_recording(self):
        """Stop audio recording"""
        if not self.is_recording:
            return
        
        try:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
            
            self.is_recording = False
            logger.info("Audio recording stopped")
            
        except Exception as e:
            logger.error(f"Error stopping audio recording: {e}")
    
    def get_audio_chunk(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """Get audio chunk from queue"""
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_volume_level(self) -> float:
        """Get current audio volume level (0.0 to 1.0)"""
        try:
            audio_chunk = self.audio_queue.get_nowait()
            return float(np.sqrt(np.mean(audio_chunk ** 2)))
        except queue.Empty:
            return 0.0
    
    def __del__(self):
        """Cleanup resources"""
        self.stop_recording()
        if hasattr(self, 'audio'):
            self.audio.terminate()

class AudioBuffer:
    """Circular buffer for storing audio data"""
    
    def __init__(self, max_duration: float = 30.0, sample_rate: int = 16000):
        """
        Initialize audio buffer
        
        Args:
            max_duration: Maximum duration to store (seconds)
            sample_rate: Audio sample rate
        """
        self.max_samples = int(max_duration * sample_rate)
        self.sample_rate = sample_rate
        self.buffer = np.zeros(self.max_samples, dtype=np.float32)
        self.write_pos = 0
        self.is_full = False
        self.lock = threading.Lock()
    
    def add_audio(self, audio_data: np.ndarray):
        """Add audio data to buffer"""
        with self.lock:
            data_len = len(audio_data)
            
            if self.write_pos + data_len <= self.max_samples:
                # Simple case: data fits without wrapping
                self.buffer[self.write_pos:self.write_pos + data_len] = audio_data
                self.write_pos += data_len
            else:
                # Wrap around case
                first_part = self.max_samples - self.write_pos
                self.buffer[self.write_pos:] = audio_data[:first_part]
                remaining = data_len - first_part
                self.buffer[:remaining] = audio_data[first_part:]
                self.write_pos = remaining
                self.is_full = True
    
    def get_recent_audio(self, duration: float) -> np.ndarray:
        """Get recent audio data"""
        with self.lock:
            samples_needed = int(duration * self.sample_rate)
            samples_needed = min(samples_needed, self.max_samples)
            
            if not self.is_full and self.write_pos < samples_needed:
                # Not enough data yet
                return self.buffer[:self.write_pos]
            
            if samples_needed >= self.max_samples:
                # Return entire buffer
                if self.is_full:
                    # Reconstruct proper order
                    return np.concatenate([
                        self.buffer[self.write_pos:],
                        self.buffer[:self.write_pos]
                    ])
                else:
                    return self.buffer[:self.write_pos]
            
            # Return recent samples
            start_pos = (self.write_pos - samples_needed) % self.max_samples
            if start_pos + samples_needed <= self.max_samples:
                return self.buffer[start_pos:start_pos + samples_needed]
            else:
                # Wrap around
                return np.concatenate([
                    self.buffer[start_pos:],
                    self.buffer[:samples_needed - (self.max_samples - start_pos)]
                ])

if __name__ == "__main__":
    # Test audio capture
    capture = AudioCapture()
    
    print("Available audio devices:")
    devices = capture.list_audio_devices()
    for device in devices:
        print(f"  {device['index']}: {device['name']}")
    
    # Test recording for 5 seconds
    print("\nTesting audio capture for 5 seconds...")
    buffer = AudioBuffer()
    
    def audio_callback(audio_data):
        buffer.add_audio(audio_data)
        volume = np.sqrt(np.mean(audio_data ** 2))
        print(f"Volume: {volume:.3f}", end='\r')
    
    capture.add_callback(audio_callback)
    capture.start_recording()
    
    time.sleep(5)
    
    capture.stop_recording()
    print("\nRecording complete!")
    
    # Get recent audio
    recent_audio = buffer.get_recent_audio(2.0)
    print(f"Captured {len(recent_audio)} samples ({len(recent_audio)/16000:.2f} seconds)")
