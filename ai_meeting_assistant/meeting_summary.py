"""
Meeting Summary Module
Generates actionable meeting summaries using HuggingFace BART/T5 models
"""

import numpy as np
import threading
import queue
import time
from typing import Dict, List, Optional, Callable, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MeetingSummary:
    """Represents a meeting summary"""
    title: str
    summary: str
    key_points: List[str]
    action_items: List[str]
    decisions: List[str]
    participants: List[str]
    duration: timedelta
    start_time: datetime
    end_time: datetime
    transcript_length: int
    confidence_score: float

@dataclass
class ActionItem:
    """Represents an action item"""
    description: str
    assignee: Optional[str]
    due_date: Optional[datetime]
    priority: str  # high, medium, low
    status: str  # pending, in_progress, completed
    timestamp: datetime

class MeetingSummarizer:
    def __init__(self, 
                 summarization_model: str = "facebook/bart-large-cnn",
                 max_input_length: int = 1024,
                 max_output_length: int = 512):
        """
        Initialize meeting summarizer
        
        Args:
            summarization_model: HuggingFace model for summarization
            max_input_length: Maximum input text length
            max_output_length: Maximum output summary length
        """
        self.summarization_model_name = summarization_model
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        
        # Initialize model
        logger.info(f"Loading summarization model: {summarization_model}")
        try:
            self.summarizer = pipeline(
                "summarization",
                model=summarization_model,
                tokenizer=summarization_model,
                device=0 if torch.cuda.is_available() else -1,
                max_length=max_output_length,
                min_length=50,
                do_sample=False
            )
            logger.info(f"Loaded summarization model: {summarization_model}")
        except Exception as e:
            logger.error(f"Failed to load summarization model: {e}")
            self.summarizer = None
        
        # Meeting data storage
        self.meeting_transcripts: List[str] = []
        self.meeting_metadata: Dict = {}
        self.participants: Set[str] = set()
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        
        # Action item patterns
        self.action_patterns = [
            r"(?:action|todo|task|follow[- ]?up|next step)s?:?\s*(.+)",
            r"(.+)\s+(?:will|should|needs? to|must)\s+(.+)",
            r"(?:assign|delegate)\s+(.+)\s+to\s+(\w+)",
            r"(\w+)\s+(?:will|should)\s+(.+)\s+by\s+(.+)",
            r"we need to\s+(.+)",
            r"let'?s\s+(.+)",
            r"i'?ll\s+(.+)",
            r"(\w+)\s+(?:is|are)\s+responsible for\s+(.+)"
        ]
        
        # Decision patterns
        self.decision_patterns = [
            r"(?:decision|decided|agree[d]?|conclude[d]?):?\s*(.+)",
            r"we (?:decided|agreed|concluded) (?:to|that)\s+(.+)",
            r"(?:final|ultimate) decision:?\s*(.+)",
            r"it was (?:decided|agreed|concluded) that\s+(.+)",
            r"the team (?:decided|agreed) (?:to|that)\s+(.+)"
        ]
        
        # Key point patterns
        self.key_point_patterns = [
            r"(?:key|main|important) point:?\s*(.+)",
            r"(?:highlight|emphasis|focus):?\s*(.+)",
            r"(?:critical|crucial|essential):?\s*(.+)",
            r"(?:note|remember|keep in mind):?\s*(.+)"
        ]
    
    def start_meeting(self, title: str = "Meeting"):
        """Start a new meeting session"""
        self.meeting_transcripts = []
        self.meeting_metadata = {"title": title}
        self.participants = set()
        self.start_time = datetime.now()
        self.end_time = None
        logger.info(f"Started meeting: {title}")
    
    def add_transcript_segment(self, text: str, speaker_id: Optional[str] = None):
        """Add a transcript segment to the meeting"""
        if speaker_id:
            self.participants.add(speaker_id)
            formatted_text = f"[{speaker_id}]: {text}"
        else:
            formatted_text = text
        
        self.meeting_transcripts.append(formatted_text)
    
    def end_meeting(self):
        """End the current meeting session"""
        self.end_time = datetime.now()
        logger.info("Meeting ended")
    
    def generate_summary(self, 
                        include_action_items: bool = True,
                        include_decisions: bool = True,
                        include_key_points: bool = True) -> MeetingSummary:
        """Generate comprehensive meeting summary"""
        if not self.meeting_transcripts:
            raise ValueError("No transcript data available for summarization")
        
        # Combine all transcript segments
        full_transcript = " ".join(self.meeting_transcripts)
        
        # Generate main summary
        main_summary = self._generate_main_summary(full_transcript)
        
        # Extract structured information
        action_items = []
        decisions = []
        key_points = []
        
        if include_action_items:
            action_items = self._extract_action_items(full_transcript)
        
        if include_decisions:
            decisions = self._extract_decisions(full_transcript)
        
        if include_key_points:
            key_points = self._extract_key_points(full_transcript)
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(
            full_transcript, main_summary, action_items, decisions
        )
        
        # Create summary object
        summary = MeetingSummary(
            title=self.meeting_metadata.get("title", "Meeting Summary"),
            summary=main_summary,
            key_points=key_points,
            action_items=[item.description for item in action_items],
            decisions=decisions,
            participants=list(self.participants),
            duration=self.end_time - self.start_time if self.end_time else timedelta(0),
            start_time=self.start_time or datetime.now(),
            end_time=self.end_time or datetime.now(),
            transcript_length=len(full_transcript),
            confidence_score=confidence_score
        )
        
        return summary
    
    def _generate_main_summary(self, transcript: str) -> str:
        """Generate main summary using the summarization model"""
        if not self.summarizer:
            return self._generate_extractive_summary(transcript)
        
        try:
            # Split long transcripts into chunks
            chunks = self._split_text_into_chunks(transcript, self.max_input_length)
            
            if len(chunks) == 1:
                # Single chunk - direct summarization
                result = self.summarizer(chunks[0])
                return result[0]['summary_text']
            else:
                # Multiple chunks - summarize each then combine
                chunk_summaries = []
                for chunk in chunks:
                    result = self.summarizer(chunk)
                    chunk_summaries.append(result[0]['summary_text'])
                
                # Combine and summarize again if needed
                combined = " ".join(chunk_summaries)
                if len(combined) > self.max_input_length:
                    result = self.summarizer(combined)
                    return result[0]['summary_text']
                else:
                    return combined
        
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return self._generate_extractive_summary(transcript)
    
    def _generate_extractive_summary(self, transcript: str) -> str:
        """Generate extractive summary as fallback"""
        sentences = re.split(r'[.!?]+', transcript)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        # Simple extractive approach - take first and last few sentences
        if len(sentences) <= 5:
            return ". ".join(sentences) + "."
        
        summary_sentences = sentences[:2] + sentences[-2:]
        return ". ".join(summary_sentences) + "."
    
    def _split_text_into_chunks(self, text: str, max_length: int) -> List[str]:
        """Split text into chunks that fit model constraints"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + 1  # +1 for space
            
            if current_length + word_length > max_length and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = word_length
            else:
                current_chunk.append(word)
                current_length += word_length
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def _extract_action_items(self, transcript: str) -> List[ActionItem]:
        """Extract action items from transcript"""
        action_items = []
        
        for pattern in self.action_patterns:
            matches = re.finditer(pattern, transcript, re.IGNORECASE)
            for match in matches:
                description = match.group(1).strip()
                
                # Try to extract assignee and due date
                assignee = self._extract_assignee(description)
                due_date = self._extract_due_date(description)
                priority = self._determine_priority(description)
                
                action_item = ActionItem(
                    description=description,
                    assignee=assignee,
                    due_date=due_date,
                    priority=priority,
                    status="pending",
                    timestamp=datetime.now()
                )
                action_items.append(action_item)
        
        # Remove duplicates
        unique_actions = []
        seen_descriptions = set()
        
        for item in action_items:
            desc_lower = item.description.lower()
            if desc_lower not in seen_descriptions:
                seen_descriptions.add(desc_lower)
                unique_actions.append(item)
        
        return unique_actions
    
    def _extract_assignee(self, text: str) -> Optional[str]:
        """Extract assignee from action item text"""
        # Look for names or pronouns
        assignee_patterns = [
            r"(\w+)\s+(?:will|should|needs? to)",
            r"assign(?:ed)?\s+to\s+(\w+)",
            r"(\w+)\s+(?:is|are)\s+responsible"
        ]
        
        for pattern in assignee_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
    
    def _extract_due_date(self, text: str) -> Optional[datetime]:
        """Extract due date from action item text"""
        # Simple date extraction patterns
        date_patterns = [
            r"by\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
            r"by\s+(tomorrow|next week|end of week)",
            r"by\s+(\d{1,2}[/-]\d{1,2})",
            r"due\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
            r"deadline\s+(\w+)"
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Simple mapping - in real implementation, use proper date parsing
                date_text = match.group(1).lower()
                if "tomorrow" in date_text:
                    return datetime.now() + timedelta(days=1)
                elif "next week" in date_text:
                    return datetime.now() + timedelta(weeks=1)
                elif "end of week" in date_text:
                    days_until_friday = (4 - datetime.now().weekday()) % 7
                    return datetime.now() + timedelta(days=days_until_friday)
        
        return None
    
    def _determine_priority(self, text: str) -> str:
        """Determine priority of action item"""
        text_lower = text.lower()
        
        high_priority_keywords = ["urgent", "asap", "immediately", "critical", "important"]
        low_priority_keywords = ["when possible", "eventually", "nice to have", "optional"]
        
        if any(keyword in text_lower for keyword in high_priority_keywords):
            return "high"
        elif any(keyword in text_lower for keyword in low_priority_keywords):
            return "low"
        else:
            return "medium"
    
    def _extract_decisions(self, transcript: str) -> List[str]:
        """Extract decisions from transcript"""
        decisions = []
        
        for pattern in self.decision_patterns:
            matches = re.finditer(pattern, transcript, re.IGNORECASE)
            for match in matches:
                decision = match.group(1).strip()
                if len(decision) > 10:  # Filter out very short matches
                    decisions.append(decision)
        
        # Remove duplicates
        unique_decisions = list(set(decisions))
        return unique_decisions[:10]  # Limit to top 10
    
    def _extract_key_points(self, transcript: str) -> List[str]:
        """Extract key points from transcript"""
        key_points = []
        
        for pattern in self.key_point_patterns:
            matches = re.finditer(pattern, transcript, re.IGNORECASE)
            for match in matches:
                point = match.group(1).strip()
                if len(point) > 10:
                    key_points.append(point)
        
        # Also extract sentences with high importance indicators
        sentences = re.split(r'[.!?]+', transcript)
        importance_keywords = ["important", "critical", "key", "main", "significant", "crucial"]
        
        for sentence in sentences:
            sentence = sentence.strip()
            if (len(sentence) > 20 and 
                any(keyword in sentence.lower() for keyword in importance_keywords)):
                key_points.append(sentence)
        
        # Remove duplicates and limit
        unique_points = list(set(key_points))
        return unique_points[:8]  # Limit to top 8
    
    def _calculate_confidence_score(self, transcript: str, summary: str, 
                                  action_items: List[ActionItem], 
                                  decisions: List[str]) -> float:
        """Calculate confidence score for the summary"""
        score = 0.0
        
        # Base score from transcript length
        if len(transcript) > 100:
            score += 0.3
        
        # Score from summary quality
        if len(summary) > 50:
            score += 0.2
        
        # Score from extracted items
        if action_items:
            score += 0.2
        
        if decisions:
            score += 0.2
        
        # Score from model availability
        if self.summarizer:
            score += 0.1
        
        return min(score, 1.0)
    
    def export_summary_to_text(self, summary: MeetingSummary) -> str:
        """Export summary to formatted text"""
        text_parts = []
        
        # Header
        text_parts.append(f"# {summary.title}")
        text_parts.append(f"**Date:** {summary.start_time.strftime('%Y-%m-%d %H:%M')}")
        text_parts.append(f"**Duration:** {summary.duration}")
        text_parts.append(f"**Participants:** {', '.join(summary.participants)}")
        text_parts.append("")
        
        # Main summary
        text_parts.append("## Summary")
        text_parts.append(summary.summary)
        text_parts.append("")
        
        # Key points
        if summary.key_points:
            text_parts.append("## Key Points")
            for i, point in enumerate(summary.key_points, 1):
                text_parts.append(f"{i}. {point}")
            text_parts.append("")
        
        # Decisions
        if summary.decisions:
            text_parts.append("## Decisions Made")
            for i, decision in enumerate(summary.decisions, 1):
                text_parts.append(f"{i}. {decision}")
            text_parts.append("")
        
        # Action items
        if summary.action_items:
            text_parts.append("## Action Items")
            for i, action in enumerate(summary.action_items, 1):
                text_parts.append(f"{i}. {action}")
            text_parts.append("")
        
        # Footer
        text_parts.append("---")
        text_parts.append(f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}*")
        text_parts.append(f"*Confidence Score: {summary.confidence_score:.2f}*")
        
        return "\n".join(text_parts)

if __name__ == "__main__":
    # Test meeting summarizer
    summarizer = MeetingSummarizer()
    
    # Simulate a meeting
    summarizer.start_meeting("Project Planning Meeting")
    
    # Add sample transcript segments
    sample_segments = [
        ("Alice", "Welcome everyone to our project planning meeting. Today we need to discuss the timeline and resource allocation."),
        ("Bob", "I think we should focus on the API development first. It's critical for the frontend team."),
        ("Alice", "Good point. Bob, can you take ownership of the API design? We need it completed by Friday."),
        ("Charlie", "I'll handle the database schema. Should have it ready by Wednesday."),
        ("Bob", "Agreed. We also decided that we'll use REST architecture for the API."),
        ("Alice", "Perfect. Let's also remember that the client presentation is next Monday, so we need to have a working demo."),
        ("Charlie", "Action item: I'll prepare the demo environment by Sunday."),
        ("Alice", "Great. Any other important points before we wrap up?"),
        ("Bob", "We should schedule a follow-up meeting for Thursday to review progress."),
        ("Alice", "Excellent. Meeting adjourned.")
    ]
    
    for speaker, text in sample_segments:
        summarizer.add_transcript_segment(text, speaker)
    
    summarizer.end_meeting()
    
    # Generate summary
    print("Generating meeting summary...")
    summary = summarizer.generate_summary()
    
    # Display results
    print("\n" + "="*50)
    print("MEETING SUMMARY")
    print("="*50)
    
    formatted_summary = summarizer.export_summary_to_text(summary)
    print(formatted_summary)
