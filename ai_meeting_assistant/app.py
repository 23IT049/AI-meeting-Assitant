"""
AI Meeting Assistant - Main Streamlit Application
Live dashboard for real-time meeting insights
"""

import streamlit as st
import threading
import time
import queue
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, List, Optional

# Import our modules
from audio_capture import AudioCapture, AudioBuffer
from speech_recognition import SpeechRecognizer, TranscriptionManager
from speaker_identification import SpeakerIdentifier, SpeakerManager, SimpleSpeakerIdentifier
from emotion_detection import EmotionDetector, EmotionVisualizer
from jargon_detection import JargonDetector
from meeting_summary import MeetingSummarizer

# Configure Streamlit page
st.set_page_config(
    page_title="AI Meeting Assistant",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .speaker-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin-bottom: 0.5rem;
    }
    .transcript-entry {
        background-color: #f8f9fa;
        padding: 0.8rem;
        border-radius: 0.3rem;
        margin-bottom: 0.5rem;
        border-left: 3px solid #28a745;
    }
    .jargon-term {
        background-color: #fff3cd;
        padding: 0.3rem 0.6rem;
        border-radius: 0.2rem;
        border: 1px solid #ffeaa7;
        display: inline-block;
        margin: 0.2rem;
    }
    .emotion-indicator {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        border-radius: 1rem;
        font-size: 0.8rem;
        font-weight: bold;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

class MeetingAssistantApp:
    def __init__(self):
        self.initialize_session_state()
        self.initialize_components()
    
    def initialize_session_state(self):
        """Initialize Streamlit session state variables"""
        if 'meeting_active' not in st.session_state:
            st.session_state.meeting_active = False
        
        if 'meeting_start_time' not in st.session_state:
            st.session_state.meeting_start_time = None
        
        if 'transcription_history' not in st.session_state:
            st.session_state.transcription_history = []
        
        if 'emotion_history' not in st.session_state:
            st.session_state.emotion_history = []
        
        if 'jargon_history' not in st.session_state:
            st.session_state.jargon_history = []
        
        if 'speaker_stats' not in st.session_state:
            st.session_state.speaker_stats = {}
        
        if 'audio_level' not in st.session_state:
            st.session_state.audio_level = 0.0
    
    def initialize_components(self):
        """Initialize AI components"""
        try:
            # Audio capture
            self.audio_capture = AudioCapture()
            self.audio_buffer = AudioBuffer()
            
            # Speech recognition with auto-detection
            self.speech_recognizer = SpeechRecognizer(
                model_size="base", 
                device="auto", 
                compute_type="auto"
            )
            self.transcription_manager = TranscriptionManager()
            
            # Speaker identification
            try:
                self.speaker_identifier = SpeakerIdentifier()
            except Exception as e:
                st.warning(f"Using fallback speaker identification: {e}")
                self.speaker_identifier = SimpleSpeakerIdentifier()
            
            self.speaker_manager = SpeakerManager()
            
            # Emotion detection
            self.emotion_detector = EmotionDetector()
            
            # Jargon detection
            self.jargon_detector = JargonDetector()
            
            # Meeting summarizer
            self.meeting_summarizer = MeetingSummarizer()
            
            # Set up callbacks
            self.setup_callbacks()
            
        except Exception as e:
            st.error(f"Error initializing components: {e}")
            st.stop()
    
    def setup_callbacks(self):
        """Set up callbacks between components"""
        # Transcription callback
        def on_transcription(segment):
            st.session_state.transcription_history.append({
                'timestamp': segment.timestamp,
                'text': segment.text,
                'speaker': getattr(segment, 'speaker_id', 'Unknown'),
                'confidence': segment.confidence
            })
            
            # Add to emotion and jargon detection
            self.emotion_detector.add_text(segment.text, getattr(segment, 'speaker_id', None))
            self.jargon_detector.add_text(segment.text, getattr(segment, 'speaker_id', None))
            
            # Add to meeting summarizer
            self.meeting_summarizer.add_transcript_segment(
                segment.text, 
                getattr(segment, 'speaker_id', None)
            )
        
        # Emotion callback
        def on_emotion(result):
            st.session_state.emotion_history.append({
                'timestamp': result.timestamp,
                'text': result.text,
                'sentiment': result.sentiment,
                'sentiment_score': result.sentiment_score,
                'emotions': result.emotions,
                'dominant_emotion': result.dominant_emotion,
                'speaker': result.speaker_id
            })
        
        # Jargon callback
        def on_jargon(term):
            st.session_state.jargon_history.append({
                'timestamp': term.timestamp,
                'term': term.term,
                'definition': term.definition,
                'source': term.source,
                'confidence': term.confidence,
                'context': term.context,
                'speaker': term.speaker_id
            })
        
        # Speaker callback
        def on_speaker(segment):
            if segment.speaker_id not in st.session_state.speaker_stats:
                st.session_state.speaker_stats[segment.speaker_id] = {
                    'total_time': 0.0,
                    'segments': 0,
                    'last_active': segment.timestamp
                }
            
            duration = segment.end_time - segment.start_time
            st.session_state.speaker_stats[segment.speaker_id]['total_time'] += duration
            st.session_state.speaker_stats[segment.speaker_id]['segments'] += 1
            st.session_state.speaker_stats[segment.speaker_id]['last_active'] = segment.timestamp
        
        # Audio callback
        def on_audio(audio_data):
            self.audio_buffer.add_audio(audio_data)
            
            # Calculate volume level
            volume = np.sqrt(np.mean(audio_data ** 2))
            st.session_state.audio_level = float(volume)
            
            # Add to speech recognition
            self.speech_recognizer.add_audio(audio_data)
            
            # Add to speaker identification
            self.speaker_identifier.add_audio(audio_data, time.time())
        
        # Register callbacks
        self.speech_recognizer.add_transcription_callback(on_transcription)
        self.emotion_detector.add_emotion_callback(on_emotion)
        self.jargon_detector.add_jargon_callback(on_jargon)
        self.speaker_identifier.add_speaker_callback(on_speaker)
        self.audio_capture.add_callback(on_audio)
    
    def render_header(self):
        """Render application header"""
        st.markdown('<h1 class="main-header">🎤 AI Meeting Assistant</h1>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            if st.session_state.meeting_active:
                duration = datetime.now() - st.session_state.meeting_start_time
                st.success(f"🔴 Meeting Active - Duration: {str(duration).split('.')[0]}")
            else:
                st.info("⚪ Meeting Not Active")
        
        with col2:
            # Audio level indicator
            volume_percentage = min(st.session_state.audio_level * 100, 100)
            st.metric("Audio Level", f"{volume_percentage:.0f}%")
        
        with col3:
            # Participant count
            participant_count = len(st.session_state.speaker_stats)
            st.metric("Participants", participant_count)
    
    def render_sidebar(self):
        """Render sidebar controls"""
        st.sidebar.title("Meeting Controls")
        
        # Meeting control buttons
        if not st.session_state.meeting_active:
            if st.sidebar.button("🎯 Start Meeting", type="primary"):
                self.start_meeting()
        else:
            if st.sidebar.button("⏹️ Stop Meeting", type="secondary"):
                self.stop_meeting()
        
        st.sidebar.divider()
        
        # Audio device selection
        st.sidebar.subheader("Audio Settings")
        devices = self.audio_capture.list_audio_devices()
        device_options = {f"{d['name']} (Index: {d['index']})": d['index'] for d in devices}
        
        selected_device = st.sidebar.selectbox(
            "Audio Input Device",
            options=list(device_options.keys()),
            index=0 if device_options else None
        )
        
        # Model settings
        st.sidebar.subheader("AI Settings")
        
        # Industry selection for jargon detection
        industries = ["Technology", "Finance", "Manufacturing", "Healthcare", "General"]
        selected_industries = st.sidebar.multiselect(
            "Industry Focus",
            industries,
            default=["Technology"]
        )
        
        # Real-time settings
        st.sidebar.subheader("Display Settings")
        
        auto_scroll = st.sidebar.checkbox("Auto-scroll Transcript", value=True)
        show_confidence = st.sidebar.checkbox("Show Confidence Scores", value=False)
        max_transcript_items = st.sidebar.slider("Max Transcript Items", 10, 100, 50)
        
        return {
            'selected_device': device_options.get(selected_device) if selected_device else None,
            'industries': selected_industries,
            'auto_scroll': auto_scroll,
            'show_confidence': show_confidence,
            'max_transcript_items': max_transcript_items
        }
    
    def start_meeting(self):
        """Start meeting and all AI components"""
        try:
            st.session_state.meeting_active = True
            st.session_state.meeting_start_time = datetime.now()
            
            # Start all components
            self.audio_capture.start_recording()
            self.speech_recognizer.start_processing()
            self.speaker_identifier.start_processing()
            self.emotion_detector.start_processing()
            self.jargon_detector.start_processing()
            
            # Start meeting summarizer
            self.meeting_summarizer.start_meeting("Live Meeting")
            
            st.sidebar.success("Meeting started successfully!")
            
        except Exception as e:
            st.sidebar.error(f"Error starting meeting: {e}")
            st.session_state.meeting_active = False
    
    def stop_meeting(self):
        """Stop meeting and all AI components"""
        try:
            st.session_state.meeting_active = False
            
            # Stop all components
            self.audio_capture.stop_recording()
            self.speech_recognizer.stop_processing()
            self.speaker_identifier.stop_processing()
            self.emotion_detector.stop_processing()
            self.jargon_detector.stop_processing()
            
            # End meeting summarizer
            self.meeting_summarizer.end_meeting()
            
            st.sidebar.success("Meeting stopped successfully!")
            
        except Exception as e:
            st.sidebar.error(f"Error stopping meeting: {e}")
    
    def render_main_dashboard(self, settings):
        """Render main dashboard with tabs"""
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📝 Live Transcript", 
            "😊 Emotions", 
            "🔍 Jargon", 
            "👥 Speakers", 
            "📋 Summary"
        ])
        
        with tab1:
            self.render_transcript_tab(settings)
        
        with tab2:
            self.render_emotions_tab()
        
        with tab3:
            self.render_jargon_tab()
        
        with tab4:
            self.render_speakers_tab()
        
        with tab5:
            self.render_summary_tab()
    
    def render_transcript_tab(self, settings):
        """Render live transcript tab"""
        st.subheader("Live Transcript")
        
        # Get recent transcriptions
        recent_transcripts = st.session_state.transcription_history[-settings['max_transcript_items']:]
        
        if not recent_transcripts:
            st.info("No transcription data yet. Start speaking to see live transcript.")
            return
        
        # Display transcripts
        transcript_container = st.container()
        
        with transcript_container:
            for entry in reversed(recent_transcripts):
                timestamp = entry['timestamp'].strftime("%H:%M:%S")
                speaker = entry['speaker']
                text = entry['text']
                confidence = entry['confidence']
                
                # Get speaker color
                color = self.speaker_manager.get_speaker_color(speaker)
                
                # Create transcript entry
                confidence_text = f" (confidence: {confidence:.2f})" if settings['show_confidence'] else ""
                
                st.markdown(f"""
                <div class="transcript-entry">
                    <strong style="color: {color}">[{timestamp}] {speaker}:</strong>{confidence_text}<br>
                    {text}
                </div>
                """, unsafe_allow_html=True)
        
        # Auto-refresh
        if st.session_state.meeting_active:
            time.sleep(1)
            st.rerun()
    
    def render_emotions_tab(self):
        """Render emotions analysis tab"""
        st.subheader("Emotion Analysis")
        
        if not st.session_state.emotion_history:
            st.info("No emotion data yet. Emotions will appear as speech is transcribed.")
            return
        
        # Recent emotions summary
        col1, col2, col3 = st.columns(3)
        
        recent_emotions = st.session_state.emotion_history[-10:]
        
        with col1:
            # Sentiment distribution
            sentiments = [e['sentiment'] for e in recent_emotions]
            sentiment_counts = {s: sentiments.count(s) for s in set(sentiments)}
            
            fig_sentiment = px.pie(
                values=list(sentiment_counts.values()),
                names=list(sentiment_counts.keys()),
                title="Recent Sentiment Distribution"
            )
            st.plotly_chart(fig_sentiment, use_container_width=True)
        
        with col2:
            # Emotion timeline
            if len(recent_emotions) > 1:
                emotion_df = pd.DataFrame([
                    {
                        'timestamp': e['timestamp'],
                        'sentiment_score': e['sentiment_score'] if e['sentiment'] == 'positive' else -e['sentiment_score']
                    }
                    for e in recent_emotions
                ])
                
                fig_timeline = px.line(
                    emotion_df,
                    x='timestamp',
                    y='sentiment_score',
                    title="Sentiment Timeline"
                )
                st.plotly_chart(fig_timeline, use_container_width=True)
        
        with col3:
            # Current mood indicator
            if recent_emotions:
                latest_emotion = recent_emotions[-1]
                emoji = EmotionVisualizer.get_emotion_emoji(latest_emotion['dominant_emotion'])
                
                st.markdown(f"""
                <div style="text-align: center; padding: 2rem;">
                    <div style="font-size: 4rem;">{emoji}</div>
                    <div style="font-size: 1.5rem; font-weight: bold;">
                        {latest_emotion['dominant_emotion'].title()}
                    </div>
                    <div style="color: #666;">Current Mood</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Recent emotion entries
        st.subheader("Recent Emotion Detections")
        
        for entry in reversed(recent_emotions[-5:]):
            timestamp = entry['timestamp'].strftime("%H:%M:%S")
            emoji = EmotionVisualizer.get_emotion_emoji(entry['dominant_emotion'])
            
            st.markdown(f"""
            <div class="transcript-entry">
                <strong>[{timestamp}] {emoji} {entry['sentiment'].upper()}</strong> 
                (Score: {entry['sentiment_score']:.2f})<br>
                <em>Dominant emotion: {entry['dominant_emotion']}</em><br>
                {entry['text'][:100]}...
            </div>
            """, unsafe_allow_html=True)
    
    def render_jargon_tab(self):
        """Render jargon detection tab"""
        st.subheader("Jargon & Technical Terms")
        
        if not st.session_state.jargon_history:
            st.info("No jargon detected yet. Technical terms will appear here as they're identified.")
            return
        
        # Jargon statistics
        col1, col2 = st.columns(2)
        
        with col1:
            # Most common terms
            terms = [j['term'] for j in st.session_state.jargon_history]
            term_counts = pd.Series(terms).value_counts().head(10)
            
            if not term_counts.empty:
                fig_terms = px.bar(
                    x=term_counts.values,
                    y=term_counts.index,
                    orientation='h',
                    title="Most Common Jargon Terms"
                )
                st.plotly_chart(fig_terms, use_container_width=True)
        
        with col2:
            # Source breakdown
            sources = [j['source'] for j in st.session_state.jargon_history]
            source_counts = pd.Series(sources).value_counts()
            
            if not source_counts.empty:
                fig_sources = px.pie(
                    values=source_counts.values,
                    names=source_counts.index,
                    title="Jargon Detection Sources"
                )
                st.plotly_chart(fig_sources, use_container_width=True)
        
        # Recent jargon terms with definitions
        st.subheader("Recent Jargon Detections")
        
        recent_jargon = st.session_state.jargon_history[-10:]
        
        for entry in reversed(recent_jargon):
            timestamp = entry['timestamp'].strftime("%H:%M:%S")
            
            with st.expander(f"[{timestamp}] {entry['term']} (confidence: {entry['confidence']:.2f})"):
                if entry['definition']:
                    st.write(f"**Definition:** {entry['definition']}")
                else:
                    st.write("*No definition available*")
                
                st.write(f"**Source:** {entry['source']}")
                st.write(f"**Context:** {entry['context'][:200]}...")
    
    def render_speakers_tab(self):
        """Render speaker analysis tab"""
        st.subheader("Speaker Analysis")
        
        if not st.session_state.speaker_stats:
            st.info("No speaker data yet. Speakers will be identified as they speak.")
            return
        
        # Speaker statistics
        col1, col2 = st.columns(2)
        
        with col1:
            # Speaking time distribution
            speakers = list(st.session_state.speaker_stats.keys())
            speaking_times = [st.session_state.speaker_stats[s]['total_time'] for s in speakers]
            
            fig_speaking = px.pie(
                values=speaking_times,
                names=speakers,
                title="Speaking Time Distribution"
            )
            st.plotly_chart(fig_speaking, use_container_width=True)
        
        with col2:
            # Speaker activity
            speaker_data = []
            for speaker, stats in st.session_state.speaker_stats.items():
                speaker_data.append({
                    'Speaker': speaker,
                    'Total Time (s)': f"{stats['total_time']:.1f}",
                    'Segments': stats['segments'],
                    'Last Active': stats['last_active'].strftime("%H:%M:%S")
                })
            
            if speaker_data:
                df_speakers = pd.DataFrame(speaker_data)
                st.dataframe(df_speakers, use_container_width=True)
        
        # Speaker management
        st.subheader("Speaker Management")
        
        for speaker_id in st.session_state.speaker_stats.keys():
            col_name, col_color = st.columns([3, 1])
            
            with col_name:
                current_name = self.speaker_manager.get_speaker_display_name(speaker_id)
                new_name = st.text_input(f"Name for {speaker_id}", value=current_name, key=f"name_{speaker_id}")
                
                if new_name != current_name:
                    self.speaker_manager.assign_speaker_name(speaker_id, new_name)
            
            with col_color:
                color = self.speaker_manager.get_speaker_color(speaker_id)
                st.color_picker(f"Color", value=color, key=f"color_{speaker_id}")
    
    def render_summary_tab(self):
        """Render meeting summary tab"""
        st.subheader("Meeting Summary")
        
        if not st.session_state.meeting_active and st.session_state.transcription_history:
            # Generate summary button
            if st.button("📋 Generate Meeting Summary", type="primary"):
                with st.spinner("Generating comprehensive meeting summary..."):
                    try:
                        summary = self.meeting_summarizer.generate_summary()
                        
                        # Display summary
                        st.success("Summary generated successfully!")
                        
                        # Export options
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.markdown(f"### {summary.title}")
                            st.write(f"**Duration:** {summary.duration}")
                            st.write(f"**Participants:** {', '.join(summary.participants)}")
                        
                        with col2:
                            # Download button
                            summary_text = self.meeting_summarizer.export_summary_to_text(summary)
                            st.download_button(
                                "📥 Download Summary",
                                summary_text,
                                file_name=f"meeting_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                                mime="text/markdown"
                            )
                        
                        # Summary sections
                        st.markdown("### Summary")
                        st.write(summary.summary)
                        
                        if summary.key_points:
                            st.markdown("### Key Points")
                            for i, point in enumerate(summary.key_points, 1):
                                st.write(f"{i}. {point}")
                        
                        if summary.decisions:
                            st.markdown("### Decisions Made")
                            for i, decision in enumerate(summary.decisions, 1):
                                st.write(f"{i}. {decision}")
                        
                        if summary.action_items:
                            st.markdown("### Action Items")
                            for i, action in enumerate(summary.action_items, 1):
                                st.write(f"{i}. {action}")
                        
                        st.info(f"Confidence Score: {summary.confidence_score:.2f}")
                        
                    except Exception as e:
                        st.error(f"Error generating summary: {e}")
        
        elif st.session_state.meeting_active:
            st.info("Meeting is currently active. Stop the meeting to generate a summary.")
        
        else:
            st.info("No meeting data available. Start and conduct a meeting to generate a summary.")
        
        # Live statistics during meeting
        if st.session_state.meeting_active and st.session_state.transcription_history:
            st.subheader("Live Meeting Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Transcripts", len(st.session_state.transcription_history))
            
            with col2:
                st.metric("Emotions Detected", len(st.session_state.emotion_history))
            
            with col3:
                st.metric("Jargon Terms", len(st.session_state.jargon_history))
            
            with col4:
                if st.session_state.meeting_start_time:
                    duration = datetime.now() - st.session_state.meeting_start_time
                    st.metric("Duration", str(duration).split('.')[0])
    
    def run(self):
        """Run the Streamlit application"""
        # Render header
        self.render_header()
        
        # Render sidebar and get settings
        settings = self.render_sidebar()
        
        # Render main dashboard
        self.render_main_dashboard(settings)
        
        # Auto-refresh for live updates
        if st.session_state.meeting_active:
            time.sleep(2)
            st.rerun()

def main():
    """Main application entry point"""
    app = MeetingAssistantApp()
    app.run()

if __name__ == "__main__":
    main()
