"""
Text Processing Utilities
Helper functions for text processing and analysis
"""

import re
import string
from typing import List, Dict, Set, Tuple, Optional
import logging
from collections import Counter
import unicodedata

logger = logging.getLogger(__name__)

def clean_text(text: str, 
               remove_punctuation: bool = False,
               remove_numbers: bool = False,
               lowercase: bool = False,
               remove_extra_whitespace: bool = True) -> str:
    """
    Clean and normalize text
    
    Args:
        text: Input text
        remove_punctuation: Remove punctuation marks
        remove_numbers: Remove numeric characters
        lowercase: Convert to lowercase
        remove_extra_whitespace: Remove extra whitespace
    
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Normalize unicode characters
    text = unicodedata.normalize('NFKD', text)
    
    # Remove extra whitespace
    if remove_extra_whitespace:
        text = re.sub(r'\s+', ' ', text).strip()
    
    # Convert to lowercase
    if lowercase:
        text = text.lower()
    
    # Remove punctuation
    if remove_punctuation:
        text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove numbers
    if remove_numbers:
        text = re.sub(r'\d+', '', text)
    
    return text

def extract_sentences(text: str) -> List[str]:
    """
    Extract sentences from text
    
    Args:
        text: Input text
    
    Returns:
        List of sentences
    """
    # Simple sentence splitting
    sentences = re.split(r'[.!?]+', text)
    
    # Clean and filter sentences
    cleaned_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > 5:  # Filter very short sentences
            cleaned_sentences.append(sentence)
    
    return cleaned_sentences

def extract_keywords(text: str, 
                    min_length: int = 3,
                    max_length: int = 20,
                    top_k: int = 10) -> List[Tuple[str, int]]:
    """
    Extract keywords from text using frequency analysis
    
    Args:
        text: Input text
        min_length: Minimum keyword length
        max_length: Maximum keyword length
        top_k: Number of top keywords to return
    
    Returns:
        List of (keyword, frequency) tuples
    """
    # Common stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
        'before', 'after', 'above', 'below', 'between', 'among', 'is', 'are',
        'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
        'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can',
        'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we',
        'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her',
        'its', 'our', 'their'
    }
    
    # Clean text and extract words
    cleaned_text = clean_text(text, remove_punctuation=True, lowercase=True)
    words = cleaned_text.split()
    
    # Filter words
    filtered_words = []
    for word in words:
        if (min_length <= len(word) <= max_length and 
            word not in stop_words and 
            word.isalpha()):
            filtered_words.append(word)
    
    # Count frequencies
    word_counts = Counter(filtered_words)
    
    return word_counts.most_common(top_k)

def extract_phrases(text: str, 
                   phrase_length: int = 2,
                   min_frequency: int = 2) -> List[Tuple[str, int]]:
    """
    Extract common phrases from text
    
    Args:
        text: Input text
        phrase_length: Length of phrases (number of words)
        min_frequency: Minimum frequency for inclusion
    
    Returns:
        List of (phrase, frequency) tuples
    """
    # Clean text
    cleaned_text = clean_text(text, remove_punctuation=True, lowercase=True)
    words = cleaned_text.split()
    
    # Extract phrases
    phrases = []
    for i in range(len(words) - phrase_length + 1):
        phrase = ' '.join(words[i:i + phrase_length])
        phrases.append(phrase)
    
    # Count frequencies
    phrase_counts = Counter(phrases)
    
    # Filter by minimum frequency
    filtered_phrases = [(phrase, count) for phrase, count in phrase_counts.items() 
                       if count >= min_frequency]
    
    return sorted(filtered_phrases, key=lambda x: x[1], reverse=True)

def detect_questions(text: str) -> List[str]:
    """
    Detect questions in text
    
    Args:
        text: Input text
    
    Returns:
        List of detected questions
    """
    questions = []
    
    # Split into sentences
    sentences = extract_sentences(text + '.')  # Add period to handle last sentence
    
    for sentence in sentences:
        sentence = sentence.strip()
        
        # Check for question marks
        if sentence.endswith('?'):
            questions.append(sentence)
        
        # Check for question words at the beginning
        question_words = ['what', 'when', 'where', 'who', 'why', 'how', 'which', 'whose']
        first_word = sentence.split()[0].lower() if sentence.split() else ''
        
        if first_word in question_words:
            questions.append(sentence + '?')
    
    return questions

def detect_action_items(text: str) -> List[str]:
    """
    Detect potential action items in text
    
    Args:
        text: Input text
    
    Returns:
        List of detected action items
    """
    action_items = []
    
    # Action item patterns
    patterns = [
        r'(?:action|todo|task|follow[- ]?up):?\s*(.+)',
        r'(.+)\s+(?:will|should|needs? to|must)\s+(.+)',
        r'(?:assign|delegate)\s+(.+)\s+to\s+(\w+)',
        r'we need to\s+(.+)',
        r'let\'?s\s+(.+)',
        r'i\'?ll\s+(.+)',
        r'(\w+)\s+(?:is|are)\s+responsible for\s+(.+)'
    ]
    
    for pattern in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            action_item = match.group(0).strip()
            if len(action_item) > 10:  # Filter very short matches
                action_items.append(action_item)
    
    return action_items

def detect_decisions(text: str) -> List[str]:
    """
    Detect decisions in text
    
    Args:
        text: Input text
    
    Returns:
        List of detected decisions
    """
    decisions = []
    
    # Decision patterns
    patterns = [
        r'(?:decision|decided|agree[d]?|conclude[d]?):?\s*(.+)',
        r'we (?:decided|agreed|concluded) (?:to|that)\s+(.+)',
        r'(?:final|ultimate) decision:?\s*(.+)',
        r'it was (?:decided|agreed|concluded) that\s+(.+)'
    ]
    
    for pattern in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            decision = match.group(1).strip() if len(match.groups()) > 0 else match.group(0).strip()
            if len(decision) > 10:
                decisions.append(decision)
    
    return decisions

def calculate_readability_score(text: str) -> Dict[str, float]:
    """
    Calculate readability scores for text
    
    Args:
        text: Input text
    
    Returns:
        Dictionary of readability metrics
    """
    if not text:
        return {}
    
    # Basic metrics
    sentences = extract_sentences(text)
    words = clean_text(text, remove_punctuation=True).split()
    
    if not sentences or not words:
        return {}
    
    # Count syllables (simple approximation)
    def count_syllables(word):
        word = word.lower()
        vowels = 'aeiouy'
        syllable_count = 0
        prev_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                syllable_count += 1
            prev_was_vowel = is_vowel
        
        # Handle silent e
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
        
        return max(1, syllable_count)
    
    total_syllables = sum(count_syllables(word) for word in words)
    
    # Calculate metrics
    avg_sentence_length = len(words) / len(sentences)
    avg_syllables_per_word = total_syllables / len(words)
    
    # Flesch Reading Ease Score
    flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
    
    # Flesch-Kincaid Grade Level
    fk_grade = (0.39 * avg_sentence_length) + (11.8 * avg_syllables_per_word) - 15.59
    
    return {
        'flesch_reading_ease': max(0, min(100, flesch_score)),
        'flesch_kincaid_grade': max(0, fk_grade),
        'avg_sentence_length': avg_sentence_length,
        'avg_syllables_per_word': avg_syllables_per_word,
        'total_words': len(words),
        'total_sentences': len(sentences),
        'total_syllables': total_syllables
    }

def extract_entities(text: str) -> Dict[str, List[str]]:
    """
    Extract named entities using simple pattern matching
    
    Args:
        text: Input text
    
    Returns:
        Dictionary of entity types and their values
    """
    entities = {
        'emails': [],
        'urls': [],
        'phone_numbers': [],
        'dates': [],
        'times': [],
        'money': [],
        'percentages': []
    }
    
    # Email pattern
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    entities['emails'] = re.findall(email_pattern, text)
    
    # URL pattern
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    entities['urls'] = re.findall(url_pattern, text)
    
    # Phone number pattern (simple)
    phone_pattern = r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b'
    entities['phone_numbers'] = re.findall(phone_pattern, text)
    
    # Date pattern (simple)
    date_pattern = r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b'
    entities['dates'] = re.findall(date_pattern, text)
    
    # Time pattern
    time_pattern = r'\b(?:[01]?[0-9]|2[0-3]):[0-5][0-9](?:\s?[AaPp][Mm])?\b'
    entities['times'] = re.findall(time_pattern, text)
    
    # Money pattern
    money_pattern = r'\$\s?[0-9,]+(?:\.[0-9]{2})?'
    entities['money'] = re.findall(money_pattern, text)
    
    # Percentage pattern
    percentage_pattern = r'\b[0-9]+(?:\.[0-9]+)?%'
    entities['percentages'] = re.findall(percentage_pattern, text)
    
    return entities

def summarize_text_extractive(text: str, 
                             num_sentences: int = 3,
                             min_sentence_length: int = 10) -> str:
    """
    Create extractive summary by selecting important sentences
    
    Args:
        text: Input text
        num_sentences: Number of sentences in summary
        min_sentence_length: Minimum sentence length
    
    Returns:
        Extractive summary
    """
    sentences = extract_sentences(text)
    
    # Filter sentences by length
    filtered_sentences = [s for s in sentences if len(s) >= min_sentence_length]
    
    if len(filtered_sentences) <= num_sentences:
        return '. '.join(filtered_sentences) + '.'
    
    # Score sentences based on word frequency
    word_freq = {}
    for sentence in filtered_sentences:
        words = clean_text(sentence, remove_punctuation=True, lowercase=True).split()
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Calculate sentence scores
    sentence_scores = []
    for sentence in filtered_sentences:
        words = clean_text(sentence, remove_punctuation=True, lowercase=True).split()
        score = sum(word_freq.get(word, 0) for word in words) / len(words) if words else 0
        sentence_scores.append((sentence, score))
    
    # Select top sentences
    top_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)[:num_sentences]
    
    # Maintain original order
    selected_sentences = []
    for sentence in filtered_sentences:
        if any(sentence == s[0] for s in top_sentences):
            selected_sentences.append(sentence)
            if len(selected_sentences) >= num_sentences:
                break
    
    return '. '.join(selected_sentences) + '.'

def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity between two texts using Jaccard similarity
    
    Args:
        text1: First text
        text2: Second text
    
    Returns:
        Similarity score (0.0 to 1.0)
    """
    # Clean and tokenize texts
    words1 = set(clean_text(text1, remove_punctuation=True, lowercase=True).split())
    words2 = set(clean_text(text2, remove_punctuation=True, lowercase=True).split())
    
    if not words1 and not words2:
        return 1.0
    
    if not words1 or not words2:
        return 0.0
    
    # Calculate Jaccard similarity
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union if union > 0 else 0.0
