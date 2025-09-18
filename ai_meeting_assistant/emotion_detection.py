"""
Emotion Detection Module
Analyzes sentiment and emotions from transcribed text using HuggingFace models
"""

import numpy as np
import threading
import queue
import time
from typing import Dict, List, Optional, Callable, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EmotionResult:
    """Represents emotion analysis result"""
    text: str
    sentiment: str  # positive, negative, neutral
    sentiment_score: float
    emotions: Dict[str, float]  # emotion -> confidence
    dominant_emotion: str
    timestamp: datetime
    speaker_id: Optional[str] = None

class EmotionDetector:
    def __init__(self, 
                 sentiment_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
                 emotion_model: str = "j-hartmann/emotion-english-distilroberta-base"):
        """
        Initialize emotion detector
        
        Args:
            sentiment_model: HuggingFace model for sentiment analysis
            emotion_model: HuggingFace model for emotion detection
        """
        self.sentiment_model_name = sentiment_model
        self.emotion_model_name = emotion_model
        
        # Initialize models
        logger.info("Loading emotion detection models...")
        try:
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model=sentiment_model,
                tokenizer=sentiment_model,
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info(f"Loaded sentiment model: {sentiment_model}")
        except Exception as e:
            logger.warning(f"Failed to load sentiment model: {e}")
            self.sentiment_analyzer = None
        
        try:
            self.emotion_analyzer = pipeline(
                "text-classification",
                model=emotion_model,
                tokenizer=emotion_model,
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info(f"Loaded emotion model: {emotion_model}")
        except Exception as e:
            logger.warning(f"Failed to load emotion model: {e}")
            self.emotion_analyzer = None
        
        # Processing queue and threading
        self.text_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.is_processing = False
        self.processing_thread = None
        
        # Callbacks
        self.emotion_callbacks = []
        
        # History and statistics
        self.emotion_history: List[EmotionResult] = []
        self.speaker_emotions: Dict[str, List[EmotionResult]] = {}
        
        # Configuration
        self.min_text_length = 10  # Minimum text length to analyze
        self.max_text_length = 512  # Maximum text length for models
        
    def add_emotion_callback(self, callback: Callable[[EmotionResult], None]):
        """Add callback for emotion detection results"""
        self.emotion_callbacks.append(callback)
    
    def _notify_emotion_callbacks(self, result: EmotionResult):
        """Notify all emotion callbacks"""
        for callback in self.emotion_callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"Error in emotion callback: {e}")
    
    def add_text(self, text: str, speaker_id: Optional[str] = None):
        """Add text for emotion analysis"""
        if not self.is_processing:
            return
        
        # Filter out very short texts
        if len(text.strip()) < self.min_text_length:
            return
        
        # Truncate very long texts
        if len(text) > self.max_text_length:
            text = text[:self.max_text_length]
        
        self.text_queue.put((text.strip(), speaker_id, datetime.now()))
    
    def _processing_worker(self):
        """Worker thread for processing text"""
        logger.info("Emotion detection processing started")
        
        while self.is_processing:
            try:
                # Get text from queue
                text, speaker_id, timestamp = self.text_queue.get(timeout=1.0)
                
                # Analyze emotions
                result = self._analyze_emotions(text, speaker_id, timestamp)
                
                if result:
                    # Store in history
                    self.emotion_history.append(result)
                    
                    # Store by speaker
                    if speaker_id:
                        if speaker_id not in self.speaker_emotions:
                            self.speaker_emotions[speaker_id] = []
                        self.speaker_emotions[speaker_id].append(result)
                    
                    # Notify callbacks
                    self._notify_emotion_callbacks(result)
                    self.result_queue.put(result)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in emotion processing: {e}")
    
    def _analyze_emotions(self, text: str, speaker_id: Optional[str], 
                         timestamp: datetime) -> Optional[EmotionResult]:
        """Analyze emotions in text"""
        try:
            sentiment = "neutral"
            sentiment_score = 0.0
            emotions = {}
            dominant_emotion = "neutral"
            
            # Sentiment analysis
            if self.sentiment_analyzer:
                try:
                    sentiment_results = self.sentiment_analyzer(text)
                    if sentiment_results:
                        result = sentiment_results[0]
                        sentiment = result['label'].lower()
                        sentiment_score = result['score']
                        
                        # Map different sentiment labels to standard format
                        if sentiment in ['positive', 'pos']:
                            sentiment = 'positive'
                        elif sentiment in ['negative', 'neg']:
                            sentiment = 'negative'
                        else:
                            sentiment = 'neutral'
                            
                except Exception as e:
                    logger.warning(f"Sentiment analysis failed: {e}")
            
            # Emotion analysis
            if self.emotion_analyzer:
                try:
                    emotion_results = self.emotion_analyzer(text)
                    if emotion_results:
                        # Convert to emotion dictionary
                        for result in emotion_results:
                            emotion_name = result['label'].lower()
                            confidence = result['score']
                            emotions[emotion_name] = confidence
                        
                        # Find dominant emotion
                        if emotions:
                            dominant_emotion = max(emotions.keys(), key=lambda k: emotions[k])
                            
                except Exception as e:
                    logger.warning(f"Emotion analysis failed: {e}")
            
            # Create result
            result = EmotionResult(
                text=text,
                sentiment=sentiment,
                sentiment_score=sentiment_score,
                emotions=emotions,
                dominant_emotion=dominant_emotion,
                timestamp=timestamp,
                speaker_id=speaker_id
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing emotions: {e}")
            return None
    
    def start_processing(self):
        """Start emotion detection processing"""
        if self.is_processing:
            logger.warning("Emotion detection already running")
            return
        
        if not self.sentiment_analyzer and not self.emotion_analyzer:
            logger.error("No emotion detection models available")
            return
        
        self.is_processing = True
        self.processing_thread = threading.Thread(target=self._processing_worker)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        logger.info("Emotion detection started")
    
    def stop_processing(self):
        """Stop emotion detection processing"""
        if not self.is_processing:
            return
        
        self.is_processing = False
        
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)
        
        logger.info("Emotion detection stopped")
    
    def get_recent_emotions(self, duration_minutes: float = 5.0) -> List[EmotionResult]:
        """Get recent emotion results"""
        cutoff_time = datetime.now().timestamp() - (duration_minutes * 60)
        
        recent = []
        for result in reversed(self.emotion_history):
            if result.timestamp.timestamp() >= cutoff_time:
                recent.append(result)
            else:
                break
        
        return list(reversed(recent))
    
    def get_speaker_emotion_summary(self, speaker_id: str, 
                                  duration_minutes: float = 10.0) -> Dict:
        """Get emotion summary for a specific speaker"""
        if speaker_id not in self.speaker_emotions:
            return {}
        
        cutoff_time = datetime.now().timestamp() - (duration_minutes * 60)
        recent_emotions = [
            result for result in self.speaker_emotions[speaker_id]
            if result.timestamp.timestamp() >= cutoff_time
        ]
        
        if not recent_emotions:
            return {}
        
        # Calculate statistics
        sentiments = [r.sentiment for r in recent_emotions]
        emotions = {}
        
        # Aggregate emotions
        for result in recent_emotions:
            for emotion, score in result.emotions.items():
                if emotion not in emotions:
                    emotions[emotion] = []
                emotions[emotion].append(score)
        
        # Calculate averages
        avg_emotions = {
            emotion: np.mean(scores) 
            for emotion, scores in emotions.items()
        }
        
        # Sentiment distribution
        sentiment_counts = {
            'positive': sentiments.count('positive'),
            'negative': sentiments.count('negative'),
            'neutral': sentiments.count('neutral')
        }
        
        total_sentiments = len(sentiments)
        sentiment_percentages = {
            sentiment: (count / total_sentiments * 100) if total_sentiments > 0 else 0
            for sentiment, count in sentiment_counts.items()
        }
        
        return {
            'total_messages': len(recent_emotions),
            'sentiment_distribution': sentiment_percentages,
            'average_emotions': avg_emotions,
            'dominant_emotion': max(avg_emotions.keys(), key=lambda k: avg_emotions[k]) if avg_emotions else 'neutral',
            'recent_trend': self._calculate_emotion_trend(recent_emotions)
        }
    
    def _calculate_emotion_trend(self, emotions: List[EmotionResult]) -> str:
        """Calculate emotion trend (improving, declining, stable)"""
        if len(emotions) < 3:
            return "stable"
        
        # Use sentiment scores to determine trend
        recent_scores = []
        for result in emotions[-5:]:  # Last 5 results
            if result.sentiment == 'positive':
                recent_scores.append(result.sentiment_score)
            elif result.sentiment == 'negative':
                recent_scores.append(-result.sentiment_score)
            else:
                recent_scores.append(0.0)
        
        if len(recent_scores) < 2:
            return "stable"
        
        # Simple trend calculation
        trend = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
        
        if trend > 0.1:
            return "improving"
        elif trend < -0.1:
            return "declining"
        else:
            return "stable"
    
    def get_meeting_emotion_summary(self) -> Dict:
        """Get overall meeting emotion summary"""
        if not self.emotion_history:
            return {}
        
        # Overall sentiment distribution
        sentiments = [r.sentiment for r in self.emotion_history]
        sentiment_counts = {
            'positive': sentiments.count('positive'),
            'negative': sentiments.count('negative'),
            'neutral': sentiments.count('neutral')
        }
        
        total = len(sentiments)
        sentiment_percentages = {
            sentiment: (count / total * 100) if total > 0 else 0
            for sentiment, count in sentiment_counts.items()
        }
        
        # Aggregate all emotions
        all_emotions = {}
        for result in self.emotion_history:
            for emotion, score in result.emotions.items():
                if emotion not in all_emotions:
                    all_emotions[emotion] = []
                all_emotions[emotion].append(score)
        
        avg_emotions = {
            emotion: np.mean(scores)
            for emotion, scores in all_emotions.items()
        }
        
        # Speaker-specific summaries
        speaker_summaries = {}
        for speaker_id in self.speaker_emotions.keys():
            speaker_summaries[speaker_id] = self.get_speaker_emotion_summary(speaker_id, 60.0)
        
        return {
            'total_analyzed_messages': len(self.emotion_history),
            'overall_sentiment_distribution': sentiment_percentages,
            'average_emotions': avg_emotions,
            'dominant_emotion': max(avg_emotions.keys(), key=lambda k: avg_emotions[k]) if avg_emotions else 'neutral',
            'speaker_summaries': speaker_summaries,
            'meeting_trend': self._calculate_emotion_trend(self.emotion_history)
        }

class EmotionVisualizer:
    """Helper class for emotion visualization"""
    
    @staticmethod
    def get_emotion_color(emotion: str) -> str:
        """Get color for emotion visualization"""
        emotion_colors = {
            'joy': '#FFD700',
            'happiness': '#FFD700',
            'sadness': '#4169E1',
            'anger': '#DC143C',
            'fear': '#800080',
            'surprise': '#FF69B4',
            'disgust': '#228B22',
            'neutral': '#808080',
            'positive': '#32CD32',
            'negative': '#FF6347'
        }
        return emotion_colors.get(emotion.lower(), '#808080')
    
    @staticmethod
    def get_emotion_emoji(emotion: str) -> str:
        """Get emoji for emotion"""
        emotion_emojis = {
            'joy': '😊',
            'happiness': '😊',
            'sadness': '😢',
            'anger': '😠',
            'fear': '😨',
            'surprise': '😲',
            'disgust': '🤢',
            'neutral': '😐',
            'positive': '👍',
            'negative': '👎'
        }
        return emotion_emojis.get(emotion.lower(), '😐')

if __name__ == "__main__":
    # Test emotion detection
    detector = EmotionDetector()
    
    def on_emotion_result(result: EmotionResult):
        emoji = EmotionVisualizer.get_emotion_emoji(result.dominant_emotion)
        print(f"[{result.timestamp.strftime('%H:%M:%S')}] {emoji} "
              f"{result.sentiment.upper()} ({result.sentiment_score:.2f}) - "
              f"{result.dominant_emotion}: {result.text[:50]}...")
    
    detector.add_emotion_callback(on_emotion_result)
    
    print("Emotion detection test...")
    detector.start_processing()
    
    # Test with sample texts
    test_texts = [
        "I'm really excited about this new project!",
        "This is quite frustrating and disappointing.",
        "The meeting is going well, everyone seems engaged.",
        "I'm not sure about this approach, it seems risky.",
        "Great job everyone, we're making excellent progress!"
    ]
    
    for i, text in enumerate(test_texts):
        detector.add_text(text, f"Speaker_{i % 2 + 1}")
        time.sleep(1)
    
    time.sleep(3)  # Wait for processing
    
    detector.stop_processing()
    
    # Show summary
    summary = detector.get_meeting_emotion_summary()
    print(f"\nMeeting Summary:")
    print(f"Total messages analyzed: {summary.get('total_analyzed_messages', 0)}")
    print(f"Sentiment distribution: {summary.get('overall_sentiment_distribution', {})}")
    print(f"Dominant emotion: {summary.get('dominant_emotion', 'neutral')}")
    print(f"Meeting trend: {summary.get('meeting_trend', 'stable')}")
