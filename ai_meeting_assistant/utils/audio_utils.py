"""
Audio Utilities
Helper functions for audio processing and analysis
"""

import numpy as np
import librosa
import soundfile as sf
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

def normalize_audio(audio: np.ndarray, target_level: float = 0.5) -> np.ndarray:
    """
    Normalize audio to target level
    
    Args:
        audio: Input audio array
        target_level: Target RMS level (0.0 to 1.0)
    
    Returns:
        Normalized audio array
    """
    if len(audio) == 0:
        return audio
    
    # Calculate current RMS
    rms = np.sqrt(np.mean(audio ** 2))
    
    if rms > 0:
        # Scale to target level
        scaling_factor = target_level / rms
        normalized = audio * scaling_factor
        
        # Prevent clipping
        max_val = np.max(np.abs(normalized))
        if max_val > 1.0:
            normalized = normalized / max_val
        
        return normalized
    
    return audio

def remove_silence(audio: np.ndarray, 
                  sample_rate: int = 16000,
                  threshold: float = 0.01,
                  min_silence_duration: float = 0.5) -> np.ndarray:
    """
    Remove silence from audio
    
    Args:
        audio: Input audio array
        sample_rate: Audio sample rate
        threshold: Energy threshold for silence detection
        min_silence_duration: Minimum silence duration to remove (seconds)
    
    Returns:
        Audio with silence removed
    """
    # Calculate frame size for silence detection
    frame_size = int(min_silence_duration * sample_rate)
    
    # Calculate energy for each frame
    energy = []
    for i in range(0, len(audio) - frame_size, frame_size // 2):
        frame = audio[i:i + frame_size]
        frame_energy = np.sqrt(np.mean(frame ** 2))
        energy.append(frame_energy)
    
    # Find non-silent frames
    non_silent_frames = [i for i, e in enumerate(energy) if e > threshold]
    
    if not non_silent_frames:
        return audio  # Return original if all frames are silent
    
    # Extract non-silent audio
    start_frame = non_silent_frames[0]
    end_frame = non_silent_frames[-1] + 1
    
    start_sample = start_frame * (frame_size // 2)
    end_sample = min(end_frame * (frame_size // 2) + frame_size, len(audio))
    
    return audio[start_sample:end_sample]

def apply_noise_reduction(audio: np.ndarray, 
                         sample_rate: int = 16000,
                         noise_duration: float = 1.0) -> np.ndarray:
    """
    Simple noise reduction using spectral subtraction
    
    Args:
        audio: Input audio array
        sample_rate: Audio sample rate
        noise_duration: Duration of noise sample at beginning (seconds)
    
    Returns:
        Audio with reduced noise
    """
    try:
        # Use first part of audio as noise sample
        noise_samples = int(noise_duration * sample_rate)
        if len(audio) <= noise_samples:
            return audio
        
        noise_segment = audio[:noise_samples]
        
        # Calculate noise spectrum
        noise_fft = np.fft.rfft(noise_segment)
        noise_magnitude = np.abs(noise_fft)
        
        # Process audio in overlapping windows
        window_size = 2048
        hop_size = window_size // 2
        
        # Apply Hanning window
        window = np.hanning(window_size)
        
        output = np.zeros_like(audio)
        
        for i in range(0, len(audio) - window_size, hop_size):
            # Extract windowed segment
            segment = audio[i:i + window_size] * window
            
            # FFT
            segment_fft = np.fft.rfft(segment)
            segment_magnitude = np.abs(segment_fft)
            segment_phase = np.angle(segment_fft)
            
            # Spectral subtraction
            alpha = 2.0  # Over-subtraction factor
            enhanced_magnitude = segment_magnitude - alpha * noise_magnitude[:len(segment_magnitude)]
            
            # Prevent over-subtraction
            enhanced_magnitude = np.maximum(enhanced_magnitude, 0.1 * segment_magnitude)
            
            # Reconstruct signal
            enhanced_fft = enhanced_magnitude * np.exp(1j * segment_phase)
            enhanced_segment = np.fft.irfft(enhanced_fft)
            
            # Overlap-add
            output[i:i + window_size] += enhanced_segment * window
        
        return output
        
    except Exception as e:
        logger.warning(f"Noise reduction failed: {e}")
        return audio

def detect_voice_activity(audio: np.ndarray,
                         sample_rate: int = 16000,
                         frame_duration: float = 0.025,
                         energy_threshold: float = 0.01,
                         zero_crossing_threshold: float = 0.3) -> np.ndarray:
    """
    Detect voice activity in audio
    
    Args:
        audio: Input audio array
        sample_rate: Audio sample rate
        frame_duration: Frame duration in seconds
        energy_threshold: Energy threshold for voice detection
        zero_crossing_threshold: Zero crossing rate threshold
    
    Returns:
        Boolean array indicating voice activity for each frame
    """
    frame_size = int(frame_duration * sample_rate)
    num_frames = len(audio) // frame_size
    
    voice_activity = np.zeros(num_frames, dtype=bool)
    
    for i in range(num_frames):
        start = i * frame_size
        end = start + frame_size
        frame = audio[start:end]
        
        # Calculate energy
        energy = np.sqrt(np.mean(frame ** 2))
        
        # Calculate zero crossing rate
        zero_crossings = np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * len(frame))
        
        # Voice activity decision
        if energy > energy_threshold and zero_crossings < zero_crossing_threshold:
            voice_activity[i] = True
    
    return voice_activity

def resample_audio(audio: np.ndarray, 
                  original_sr: int, 
                  target_sr: int) -> np.ndarray:
    """
    Resample audio to target sample rate
    
    Args:
        audio: Input audio array
        original_sr: Original sample rate
        target_sr: Target sample rate
    
    Returns:
        Resampled audio array
    """
    if original_sr == target_sr:
        return audio
    
    try:
        resampled = librosa.resample(audio, orig_sr=original_sr, target_sr=target_sr)
        return resampled
    except Exception as e:
        logger.error(f"Resampling failed: {e}")
        return audio

def save_audio(audio: np.ndarray, 
               filename: str, 
               sample_rate: int = 16000) -> bool:
    """
    Save audio to file
    
    Args:
        audio: Audio array to save
        filename: Output filename
        sample_rate: Audio sample rate
    
    Returns:
        True if successful, False otherwise
    """
    try:
        sf.write(filename, audio, sample_rate)
        return True
    except Exception as e:
        logger.error(f"Failed to save audio: {e}")
        return False

def load_audio(filename: str, 
               target_sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """
    Load audio from file
    
    Args:
        filename: Input filename
        target_sr: Target sample rate (optional)
    
    Returns:
        Tuple of (audio_array, sample_rate)
    """
    try:
        audio, sr = sf.read(filename)
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Resample if needed
        if target_sr and sr != target_sr:
            audio = resample_audio(audio, sr, target_sr)
            sr = target_sr
        
        return audio, sr
        
    except Exception as e:
        logger.error(f"Failed to load audio: {e}")
        return np.array([]), 0

def calculate_audio_features(audio: np.ndarray, 
                           sample_rate: int = 16000) -> dict:
    """
    Calculate various audio features
    
    Args:
        audio: Input audio array
        sample_rate: Audio sample rate
    
    Returns:
        Dictionary of audio features
    """
    features = {}
    
    try:
        # Basic features
        features['duration'] = len(audio) / sample_rate
        features['rms_energy'] = np.sqrt(np.mean(audio ** 2))
        features['max_amplitude'] = np.max(np.abs(audio))
        
        # Zero crossing rate
        zero_crossings = np.sum(np.abs(np.diff(np.sign(audio))))
        features['zero_crossing_rate'] = zero_crossings / (2 * len(audio))
        
        # Spectral features using librosa
        if len(audio) > 0:
            # Spectral centroid
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)[0]
            features['spectral_centroid_mean'] = np.mean(spectral_centroids)
            features['spectral_centroid_std'] = np.std(spectral_centroids)
            
            # Spectral rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)[0]
            features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
            
            # MFCCs
            mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
            for i in range(13):
                features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
                features[f'mfcc_{i}_std'] = np.std(mfccs[i])
        
    except Exception as e:
        logger.warning(f"Feature calculation failed: {e}")
    
    return features

def split_audio_by_silence(audio: np.ndarray,
                          sample_rate: int = 16000,
                          silence_threshold: float = 0.01,
                          min_silence_duration: float = 0.5,
                          min_segment_duration: float = 1.0) -> list:
    """
    Split audio into segments based on silence
    
    Args:
        audio: Input audio array
        sample_rate: Audio sample rate
        silence_threshold: Energy threshold for silence detection
        min_silence_duration: Minimum silence duration for splitting
        min_segment_duration: Minimum segment duration
    
    Returns:
        List of audio segments
    """
    frame_size = int(0.025 * sample_rate)  # 25ms frames
    hop_size = frame_size // 2
    
    # Calculate energy for each frame
    energy = []
    for i in range(0, len(audio) - frame_size, hop_size):
        frame = audio[i:i + frame_size]
        frame_energy = np.sqrt(np.mean(frame ** 2))
        energy.append(frame_energy)
    
    # Find silence regions
    silence_frames = [i for i, e in enumerate(energy) if e < silence_threshold]
    
    # Group consecutive silence frames
    silence_regions = []
    if silence_frames:
        start = silence_frames[0]
        end = start
        
        for frame in silence_frames[1:]:
            if frame == end + 1:
                end = frame
            else:
                if (end - start + 1) * hop_size >= min_silence_duration * sample_rate:
                    silence_regions.append((start * hop_size, (end + 1) * hop_size))
                start = frame
                end = frame
        
        # Add last region
        if (end - start + 1) * hop_size >= min_silence_duration * sample_rate:
            silence_regions.append((start * hop_size, (end + 1) * hop_size))
    
    # Split audio at silence regions
    segments = []
    last_end = 0
    
    for silence_start, silence_end in silence_regions:
        if silence_start - last_end >= min_segment_duration * sample_rate:
            segment = audio[last_end:silence_start]
            segments.append(segment)
        last_end = silence_end
    
    # Add final segment
    if len(audio) - last_end >= min_segment_duration * sample_rate:
        segment = audio[last_end:]
        segments.append(segment)
    
    return segments
