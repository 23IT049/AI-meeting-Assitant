"""
Speaker Identification Module
Identifies and tracks different speakers using pyannote.audio
"""

import numpy as np
import threading
import queue
import time
from typing import Dict, List, Optional, Tuple, Callable
import logging
from dataclasses import dataclass
from datetime import datetime
import tempfile

# Optional imports with fallbacks
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available - using simplified speaker identification")

try:
    from pyannote.audio import Pipeline
    from pyannote.core import Segment, Annotation
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False
    logging.warning("pyannote.audio not available - using fallback speaker identification")

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False
    logging.warning("soundfile not available - some audio features disabled")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SpeakerSegment:
    """Represents a speaker segment"""
    speaker_id: str
    start_time: float
    end_time: float
    confidence: float
    timestamp: datetime

class SpeakerIdentifier:
    def __init__(self, 
                 use_auth_token: Optional[str] = None,
                 min_speakers: int = 1,
                 max_speakers: int = 10):
        """
        Initialize speaker identification
        
        Args:
            use_auth_token: HuggingFace auth token for pyannote models
            min_speakers: Minimum number of speakers to detect
            max_speakers: Maximum number of speakers to detect
        """
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.use_auth_token = use_auth_token
        
        # Initialize pipeline
        self.pipeline = None
        
        if PYANNOTE_AVAILABLE:
            logger.info("Loading speaker diarization pipeline...")
            try:
                self.pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=use_auth_token
                )
                logger.info("✅ Advanced speaker diarization loaded")
            except Exception as e:
                logger.warning(f"Failed to load speaker diarization pipeline: {e}")
                logger.info("Falling back to speaker segmentation...")
                try:
                    self.pipeline = Pipeline.from_pretrained(
                        "pyannote/segmentation-3.0",
                        use_auth_token=use_auth_token
                    )
                    logger.info("✅ Speaker segmentation loaded")
                except Exception as e2:
                    logger.warning(f"Failed to load any speaker model: {e2}")
                    self.pipeline = None
        else:
            logger.info("pyannote.audio not available - using fallback speaker identification")
        
        # Processing queue and threading
        self.audio_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.is_processing = False
        self.processing_thread = None
        
        # Callbacks
        self.speaker_callbacks = []
        
        # Speaker tracking
        self.speaker_history: List[SpeakerSegment] = []
        self.current_speaker: Optional[str] = None
        self.speaker_embeddings: Dict[str, np.ndarray] = {}
        
        # Configuration
        self.sample_rate = 16000
        self.min_segment_duration = 1.0  # Minimum segment duration
        
    def add_speaker_callback(self, callback: Callable[[SpeakerSegment], None]):
        """Add callback for speaker detection results"""
        self.speaker_callbacks.append(callback)
    
    def _notify_speaker_callbacks(self, segment: SpeakerSegment):
        """Notify all speaker callbacks"""
        for callback in self.speaker_callbacks:
            try:
                callback(segment)
            except Exception as e:
                logger.error(f"Error in speaker callback: {e}")
    
    def add_audio(self, audio_data: np.ndarray, timestamp: float):
        """Add audio data for speaker identification"""
        if not self.is_processing or self.pipeline is None:
            return
        
        self.audio_queue.put((audio_data.copy(), timestamp))
    
    def _processing_worker(self):
        """Worker thread for processing audio"""
        logger.info("Speaker identification processing started")
        
        audio_buffer = []
        buffer_timestamps = []
        buffer_duration = 0.0
        max_buffer_duration = 10.0  # Process every 10 seconds
        
        while self.is_processing:
            try:
                # Get audio from queue
                audio_data, timestamp = self.audio_queue.get(timeout=1.0)
                
                # Add to buffer
                audio_buffer.extend(audio_data)
                buffer_timestamps.append(timestamp)
                buffer_duration = len(audio_buffer) / self.sample_rate
                
                # Process when buffer is full enough
                if buffer_duration >= max_buffer_duration:
                    audio_array = np.array(audio_buffer, dtype=np.float32)
                    start_timestamp = buffer_timestamps[0]
                    
                    # Process speaker identification
                    segments = self._identify_speakers(audio_array, start_timestamp)
                    
                    # Send results to callbacks
                    for segment in segments:
                        self._notify_speaker_callbacks(segment)
                        self.result_queue.put(segment)
                        self.speaker_history.append(segment)
                    
                    # Keep some overlap for continuity
                    overlap_duration = 2.0  # 2 second overlap
                    overlap_samples = int(overlap_duration * self.sample_rate)
                    
                    if len(audio_buffer) > overlap_samples:
                        audio_buffer = audio_buffer[-overlap_samples:]
                        buffer_timestamps = buffer_timestamps[-1:]
                        buffer_duration = len(audio_buffer) / self.sample_rate
                    else:
                        audio_buffer = []
                        buffer_timestamps = []
                        buffer_duration = 0.0
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in speaker processing: {e}")
    
    def _identify_speakers(self, audio_data: np.ndarray, start_timestamp: float) -> List[SpeakerSegment]:
        """Identify speakers in audio data"""
        if self.pipeline is None:
            # Use simple fallback detection
            return self._simple_speaker_detection(audio_data, start_timestamp)
        
        try:
            if not SOUNDFILE_AVAILABLE:
                # Fallback: create simple speaker segments based on voice activity
                return self._simple_speaker_detection(audio_data, start_timestamp)
            
            # Create temporary audio file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                sf.write(temp_file.name, audio_data, self.sample_rate)
                temp_path = temp_file.name
            
            # Run speaker diarization
            diarization = self.pipeline(temp_path)
            
            # Convert results to SpeakerSegment objects
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                if turn.duration >= self.min_segment_duration:
                    segment = SpeakerSegment(
                        speaker_id=speaker,
                        start_time=turn.start,
                        end_time=turn.end,
                        confidence=1.0,  # pyannote doesn't provide confidence scores
                        timestamp=datetime.now()
                    )
                    segments.append(segment)
            
            # Clean up temporary file
            import os
            try:
                os.unlink(temp_path)
            except:
                pass
            
            return segments
            
        except Exception as e:
            logger.error(f"Error identifying speakers: {e}")
            return []
    
    def _simple_speaker_detection(self, audio_data: np.ndarray, start_timestamp: float) -> List[SpeakerSegment]:
        """Simple fallback speaker detection based on voice activity"""
        try:
            # Simple voice activity detection
            frame_size = int(0.025 * self.sample_rate)  # 25ms frames
            hop_size = frame_size // 2
            
            # Calculate energy for each frame
            energy_threshold = 0.01
            voice_frames = []
            
            for i in range(0, len(audio_data) - frame_size, hop_size):
                frame = audio_data[i:i + frame_size]
                energy = np.sqrt(np.mean(frame ** 2))
                voice_frames.append(energy > energy_threshold)
            
            # Find voice segments
            segments = []
            in_voice = False
            segment_start = 0
            current_speaker = "Speaker_1"  # Default speaker
            
            for i, is_voice in enumerate(voice_frames):
                time_pos = i * hop_size / self.sample_rate
                
                if is_voice and not in_voice:
                    # Start of voice segment
                    segment_start = time_pos
                    in_voice = True
                elif not is_voice and in_voice:
                    # End of voice segment
                    if time_pos - segment_start >= self.min_segment_duration:
                        segment = SpeakerSegment(
                            speaker_id=current_speaker,
                            start_time=segment_start,
                            end_time=time_pos,
                            confidence=0.5,  # Low confidence for simple detection
                            timestamp=datetime.now()
                        )
                        segments.append(segment)
                    in_voice = False
            
            # Handle final segment
            if in_voice:
                final_time = len(audio_data) / self.sample_rate
                if final_time - segment_start >= self.min_segment_duration:
                    segment = SpeakerSegment(
                        speaker_id=current_speaker,
                        start_time=segment_start,
                        end_time=final_time,
                        confidence=0.5,
                        timestamp=datetime.now()
                    )
                    segments.append(segment)
            
            return segments
            
        except Exception as e:
            logger.error(f"Simple speaker detection failed: {e}")
            return []
    
    def start_processing(self):
        """Start speaker identification processing"""
        if self.is_processing:
            logger.warning("Speaker identification already running")
            return
        
        if self.pipeline is None and not PYANNOTE_AVAILABLE:
            logger.info("Using fallback speaker identification (voice activity detection)")
        elif self.pipeline is None:
            logger.warning("No speaker identification pipeline available - limited functionality")
        
        self.is_processing = True
        self.processing_thread = threading.Thread(target=self._processing_worker)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        logger.info("Speaker identification started")
    
    def stop_processing(self):
        """Stop speaker identification processing"""
        if not self.is_processing:
            return
        
        self.is_processing = False
        
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)
        
        logger.info("Speaker identification stopped")
    
    def get_current_speaker(self) -> Optional[str]:
        """Get the currently active speaker"""
        if not self.speaker_history:
            return None
        
        # Find the most recent speaker segment
        current_time = time.time()
        for segment in reversed(self.speaker_history):
            segment_age = current_time - segment.timestamp.timestamp()
            if segment_age < 5.0:  # Within last 5 seconds
                return segment.speaker_id
        
        return None
    
    def get_speaker_statistics(self) -> Dict[str, Dict]:
        """Get statistics about speakers"""
        stats = {}
        
        for segment in self.speaker_history:
            speaker_id = segment.speaker_id
            if speaker_id not in stats:
                stats[speaker_id] = {
                    'total_duration': 0.0,
                    'segment_count': 0,
                    'first_appearance': segment.timestamp,
                    'last_appearance': segment.timestamp
                }
            
            duration = segment.end_time - segment.start_time
            stats[speaker_id]['total_duration'] += duration
            stats[speaker_id]['segment_count'] += 1
            
            if segment.timestamp < stats[speaker_id]['first_appearance']:
                stats[speaker_id]['first_appearance'] = segment.timestamp
            if segment.timestamp > stats[speaker_id]['last_appearance']:
                stats[speaker_id]['last_appearance'] = segment.timestamp
        
        return stats
    
    def get_recent_speakers(self, duration_minutes: float = 5.0) -> List[str]:
        """Get list of speakers active in recent time"""
        cutoff_time = datetime.now().timestamp() - (duration_minutes * 60)
        recent_speakers = set()
        
        for segment in self.speaker_history:
            if segment.timestamp.timestamp() >= cutoff_time:
                recent_speakers.add(segment.speaker_id)
        
        return list(recent_speakers)

class SpeakerManager:
    """Manages speaker information and provides naming/tracking capabilities"""
    
    def __init__(self):
        self.speaker_names: Dict[str, str] = {}  # speaker_id -> custom_name
        self.speaker_colors: Dict[str, str] = {}  # speaker_id -> color
        self.available_colors = [
            "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", 
            "#FFEAA7", "#DDA0DD", "#98D8C8", "#F7DC6F"
        ]
        self.color_index = 0
    
    def assign_speaker_name(self, speaker_id: str, name: str):
        """Assign a custom name to a speaker"""
        self.speaker_names[speaker_id] = name
        logger.info(f"Assigned name '{name}' to speaker {speaker_id}")
    
    def get_speaker_display_name(self, speaker_id: str) -> str:
        """Get display name for speaker"""
        return self.speaker_names.get(speaker_id, f"Speaker {speaker_id}")
    
    def get_speaker_color(self, speaker_id: str) -> str:
        """Get color for speaker visualization"""
        if speaker_id not in self.speaker_colors:
            self.speaker_colors[speaker_id] = self.available_colors[
                self.color_index % len(self.available_colors)
            ]
            self.color_index += 1
        
        return self.speaker_colors[speaker_id]
    
    def get_all_speakers(self) -> List[Dict]:
        """Get all known speakers with their information"""
        speakers = []
        all_speaker_ids = set(self.speaker_names.keys()) | set(self.speaker_colors.keys())
        
        for speaker_id in all_speaker_ids:
            speakers.append({
                'id': speaker_id,
                'name': self.get_speaker_display_name(speaker_id),
                'color': self.get_speaker_color(speaker_id)
            })
        
        return speakers

# Simple fallback speaker identification for when pyannote is not available
class SimpleSpeakerIdentifier:
    """Fallback speaker identifier using voice activity detection"""
    
    def __init__(self):
        self.is_processing = False
        self.speaker_callbacks = []
        self.current_speaker = "Speaker_1"
        
    def add_speaker_callback(self, callback: Callable[[SpeakerSegment], None]):
        """Add callback for speaker detection results"""
        self.speaker_callbacks.append(callback)
    
    def add_audio(self, audio_data: np.ndarray, timestamp: float):
        """Add audio data - simple implementation just detects voice activity"""
        if not self.is_processing:
            return
        
        # Simple voice activity detection based on energy
        energy = np.sqrt(np.mean(audio_data ** 2))
        
        if energy > 0.01:  # Threshold for voice activity
            segment = SpeakerSegment(
                speaker_id=self.current_speaker,
                start_time=0.0,
                end_time=len(audio_data) / 16000,
                confidence=energy,
                timestamp=datetime.now()
            )
            
            for callback in self.speaker_callbacks:
                try:
                    callback(segment)
                except Exception as e:
                    logger.error(f"Error in speaker callback: {e}")
    
    def start_processing(self):
        """Start processing"""
        self.is_processing = True
        logger.info("Simple speaker identification started")
    
    def stop_processing(self):
        """Stop processing"""
        self.is_processing = False
        logger.info("Simple speaker identification stopped")
    
    def get_current_speaker(self) -> Optional[str]:
        """Get current speaker"""
        return self.current_speaker

if __name__ == "__main__":
    # Test speaker identification
    try:
        identifier = SpeakerIdentifier()
    except Exception as e:
        logger.warning(f"Using fallback speaker identifier: {e}")
        identifier = SimpleSpeakerIdentifier()
    
    manager = SpeakerManager()
    
    def on_speaker_detection(segment: SpeakerSegment):
        display_name = manager.get_speaker_display_name(segment.speaker_id)
        color = manager.get_speaker_color(segment.speaker_id)
        print(f"[{segment.timestamp.strftime('%H:%M:%S')}] {display_name}: "
              f"{segment.start_time:.1f}s - {segment.end_time:.1f}s")
    
    identifier.add_speaker_callback(on_speaker_detection)
    
    print("Speaker identification test...")
    identifier.start_processing()
    
    # Simulate some audio input
    try:
        time.sleep(10)
    except KeyboardInterrupt:
        pass
    
    identifier.stop_processing()
    
    # Show speaker statistics
    if hasattr(identifier, 'get_speaker_statistics'):
        stats = identifier.get_speaker_statistics()
        print(f"\nDetected {len(stats)} speakers:")
        for speaker_id, data in stats.items():
            display_name = manager.get_speaker_display_name(speaker_id)
            print(f"  {display_name}: {data['total_duration']:.1f}s total, "
                  f"{data['segment_count']} segments")
