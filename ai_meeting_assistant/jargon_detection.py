"""
Jargon Detection Module
Identifies technical terms and provides explanations using spaCy, KeyBERT, and Wikipedia
"""

import spacy
import numpy as np
import threading
import queue
import time
import re
from typing import Dict, List, Optional, Callable, Set, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime
from keybert import KeyBERT
import wikipedia
import requests
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class JargonTerm:
    """Represents a detected jargon term"""
    term: str
    context: str
    definition: Optional[str]
    source: str  # 'glossary', 'wikipedia', 'keybert'
    confidence: float
    timestamp: datetime
    speaker_id: Optional[str] = None

@dataclass
class IndustryGlossary:
    """Industry-specific glossary"""
    name: str
    terms: Dict[str, str]  # term -> definition
    keywords: Set[str]  # keywords that indicate this industry

class JargonDetector:
    def __init__(self, 
                 spacy_model: str = "en_core_web_sm",
                 industries: Optional[List[str]] = None):
        """
        Initialize jargon detector
        
        Args:
            spacy_model: spaCy model to use
            industries: List of industries to focus on
        """
        self.spacy_model_name = spacy_model
        self.industries = industries or ["general", "technology", "finance", "manufacturing"]
        
        # Initialize models
        logger.info("Loading jargon detection models...")
        try:
            self.nlp = spacy.load(spacy_model)
            logger.info(f"Loaded spaCy model: {spacy_model}")
        except Exception as e:
            logger.error(f"Failed to load spaCy model: {e}")
            self.nlp = None
        
        try:
            self.keybert = KeyBERT()
            logger.info("Loaded KeyBERT model")
        except Exception as e:
            logger.warning(f"Failed to load KeyBERT: {e}")
            self.keybert = None
        
        # Processing queue and threading
        self.text_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.is_processing = False
        self.processing_thread = None
        
        # Callbacks
        self.jargon_callbacks = []
        
        # Jargon tracking
        self.detected_jargon: List[JargonTerm] = []
        self.term_cache: Dict[str, JargonTerm] = {}  # Cache for definitions
        self.speaker_jargon: Dict[str, List[JargonTerm]] = {}
        
        # Industry glossaries
        self.glossaries = self._load_industry_glossaries()
        
        # Configuration
        self.min_term_length = 3
        self.max_term_length = 50
        self.confidence_threshold = 0.3
        
    def _load_industry_glossaries(self) -> Dict[str, IndustryGlossary]:
        """Load industry-specific glossaries"""
        glossaries = {}
        
        # Technology glossary
        tech_terms = {
            "API": "Application Programming Interface - a set of protocols and tools for building software applications",
            "ML": "Machine Learning - a type of artificial intelligence that enables computers to learn without being explicitly programmed",
            "AI": "Artificial Intelligence - computer systems able to perform tasks that typically require human intelligence",
            "DevOps": "Development Operations - practices that combine software development and IT operations",
            "CI/CD": "Continuous Integration/Continuous Deployment - practices for automating software delivery",
            "microservices": "Architectural approach where applications are built as a collection of small, independent services",
            "containerization": "Method of packaging applications with their dependencies for consistent deployment",
            "kubernetes": "Open-source platform for automating deployment, scaling, and management of containerized applications",
            "docker": "Platform for developing, shipping, and running applications in containers",
            "REST": "Representational State Transfer - architectural style for designing networked applications",
            "GraphQL": "Query language and runtime for APIs that allows clients to request specific data",
            "blockchain": "Distributed ledger technology that maintains a continuously growing list of records",
            "IoT": "Internet of Things - network of physical devices connected to the internet",
            "SaaS": "Software as a Service - software licensing and delivery model",
            "PaaS": "Platform as a Service - cloud computing service that provides a platform for development",
            "IaaS": "Infrastructure as a Service - cloud computing service that provides virtualized computing resources"
        }
        
        tech_keywords = {"technology", "software", "development", "programming", "coding", "tech", "digital"}
        
        glossaries["technology"] = IndustryGlossary(
            name="Technology",
            terms=tech_terms,
            keywords=tech_keywords
        )
        
        # Finance glossary
        finance_terms = {
            "ROI": "Return on Investment - measure of investment efficiency",
            "KPI": "Key Performance Indicator - measurable value that demonstrates effectiveness",
            "EBITDA": "Earnings Before Interest, Taxes, Depreciation, and Amortization",
            "P&L": "Profit and Loss statement - financial statement summarizing revenues and expenses",
            "CAPEX": "Capital Expenditure - funds used to acquire or upgrade physical assets",
            "OPEX": "Operating Expense - ongoing costs for running a business",
            "NPV": "Net Present Value - difference between present value of cash inflows and outflows",
            "IRR": "Internal Rate of Return - metric used to estimate profitability of investments",
            "DCF": "Discounted Cash Flow - valuation method used to estimate investment attractiveness",
            "WACC": "Weighted Average Cost of Capital - calculation of firm's cost of capital",
            "leverage": "Use of borrowed capital to increase potential return on investment",
            "liquidity": "Ease with which an asset can be converted into cash",
            "volatility": "Degree of variation in trading price over time",
            "hedge": "Investment position intended to offset potential losses",
            "arbitrage": "Practice of taking advantage of price differences in different markets"
        }
        
        finance_keywords = {"finance", "financial", "investment", "banking", "money", "capital", "revenue"}
        
        glossaries["finance"] = IndustryGlossary(
            name="Finance",
            terms=finance_terms,
            keywords=finance_keywords
        )
        
        # Manufacturing glossary
        manufacturing_terms = {
            "lean": "Manufacturing methodology focused on minimizing waste while maximizing productivity",
            "six sigma": "Set of techniques and tools for process improvement",
            "kaizen": "Japanese business philosophy of continuous improvement",
            "JIT": "Just-In-Time - production strategy that reduces inventory costs",
            "OEE": "Overall Equipment Effectiveness - measure of manufacturing productivity",
            "TPM": "Total Productive Maintenance - approach to equipment maintenance",
            "5S": "Workplace organization method: Sort, Set in order, Shine, Standardize, Sustain",
            "SMED": "Single-Minute Exchange of Die - rapid changeover methodology",
            "poka-yoke": "Japanese term for mistake-proofing or error prevention",
            "takt time": "Rate at which products must be produced to meet customer demand",
            "cycle time": "Total time from beginning to end of a process",
            "bottleneck": "Point in production where flow is limited or restricted",
            "throughput": "Rate of production or amount processed in given time",
            "yield": "Percentage of products that meet quality standards",
            "scrap rate": "Percentage of materials that become waste during production"
        }
        
        manufacturing_keywords = {"manufacturing", "production", "factory", "assembly", "quality", "process"}
        
        glossaries["manufacturing"] = IndustryGlossary(
            name="Manufacturing",
            terms=manufacturing_terms,
            keywords=manufacturing_keywords
        )
        
        return glossaries
    
    def add_jargon_callback(self, callback: Callable[[JargonTerm], None]):
        """Add callback for jargon detection results"""
        self.jargon_callbacks.append(callback)
    
    def _notify_jargon_callbacks(self, term: JargonTerm):
        """Notify all jargon callbacks"""
        for callback in self.jargon_callbacks:
            try:
                callback(term)
            except Exception as e:
                logger.error(f"Error in jargon callback: {e}")
    
    def add_text(self, text: str, speaker_id: Optional[str] = None):
        """Add text for jargon analysis"""
        if not self.is_processing:
            return
        
        self.text_queue.put((text.strip(), speaker_id, datetime.now()))
    
    def _processing_worker(self):
        """Worker thread for processing text"""
        logger.info("Jargon detection processing started")
        
        while self.is_processing:
            try:
                # Get text from queue
                text, speaker_id, timestamp = self.text_queue.get(timeout=1.0)
                
                # Detect jargon
                jargon_terms = self._detect_jargon(text, speaker_id, timestamp)
                
                # Process each term
                for term in jargon_terms:
                    # Store in history
                    self.detected_jargon.append(term)
                    
                    # Store by speaker
                    if speaker_id:
                        if speaker_id not in self.speaker_jargon:
                            self.speaker_jargon[speaker_id] = []
                        self.speaker_jargon[speaker_id].append(term)
                    
                    # Notify callbacks
                    self._notify_jargon_callbacks(term)
                    self.result_queue.put(term)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in jargon processing: {e}")
    
    def _detect_jargon(self, text: str, speaker_id: Optional[str], 
                      timestamp: datetime) -> List[JargonTerm]:
        """Detect jargon terms in text"""
        detected_terms = []
        
        # Method 1: Check against industry glossaries
        glossary_terms = self._check_glossaries(text, timestamp, speaker_id)
        detected_terms.extend(glossary_terms)
        
        # Method 2: Use KeyBERT for keyword extraction
        if self.keybert:
            keybert_terms = self._extract_keywords(text, timestamp, speaker_id)
            detected_terms.extend(keybert_terms)
        
        # Method 3: Use spaCy for named entity recognition
        if self.nlp:
            ner_terms = self._extract_entities(text, timestamp, speaker_id)
            detected_terms.extend(ner_terms)
        
        # Remove duplicates and filter by confidence
        unique_terms = {}
        for term in detected_terms:
            if term.confidence >= self.confidence_threshold:
                key = term.term.lower()
                if key not in unique_terms or term.confidence > unique_terms[key].confidence:
                    unique_terms[key] = term
        
        return list(unique_terms.values())
    
    def _check_glossaries(self, text: str, timestamp: datetime, 
                         speaker_id: Optional[str]) -> List[JargonTerm]:
        """Check text against industry glossaries"""
        terms = []
        text_lower = text.lower()
        
        # Determine relevant industries based on context
        relevant_industries = self._identify_relevant_industries(text)
        
        for industry_name in relevant_industries:
            if industry_name in self.glossaries:
                glossary = self.glossaries[industry_name]
                
                for term, definition in glossary.terms.items():
                    # Check for exact matches (case insensitive)
                    pattern = r'\b' + re.escape(term.lower()) + r'\b'
                    if re.search(pattern, text_lower):
                        jargon_term = JargonTerm(
                            term=term,
                            context=text,
                            definition=definition,
                            source=f"glossary_{industry_name}",
                            confidence=0.9,  # High confidence for glossary matches
                            timestamp=timestamp,
                            speaker_id=speaker_id
                        )
                        terms.append(jargon_term)
        
        return terms
    
    def _identify_relevant_industries(self, text: str) -> List[str]:
        """Identify relevant industries based on text content"""
        text_lower = text.lower()
        relevant = []
        
        for industry_name, glossary in self.glossaries.items():
            # Check if any industry keywords appear in text
            keyword_matches = sum(1 for keyword in glossary.keywords 
                                if keyword in text_lower)
            
            if keyword_matches > 0:
                relevant.append(industry_name)
        
        # If no specific industry detected, include general terms
        if not relevant:
            relevant = ["technology"]  # Default to technology
        
        return relevant
    
    def _extract_keywords(self, text: str, timestamp: datetime, 
                         speaker_id: Optional[str]) -> List[JargonTerm]:
        """Extract keywords using KeyBERT"""
        terms = []
        
        try:
            # Extract keywords
            keywords = self.keybert.extract_keywords(
                text, 
                keyphrase_ngram_range=(1, 3),
                stop_words='english',
                top_k=5
            )
            
            for keyword, score in keywords:
                if (len(keyword) >= self.min_term_length and 
                    len(keyword) <= self.max_term_length and
                    score >= self.confidence_threshold):
                    
                    # Try to get definition from Wikipedia
                    definition = self._get_wikipedia_definition(keyword)
                    
                    jargon_term = JargonTerm(
                        term=keyword,
                        context=text,
                        definition=definition,
                        source="keybert",
                        confidence=score,
                        timestamp=timestamp,
                        speaker_id=speaker_id
                    )
                    terms.append(jargon_term)
        
        except Exception as e:
            logger.warning(f"KeyBERT extraction failed: {e}")
        
        return terms
    
    def _extract_entities(self, text: str, timestamp: datetime, 
                         speaker_id: Optional[str]) -> List[JargonTerm]:
        """Extract named entities using spaCy"""
        terms = []
        
        try:
            doc = self.nlp(text)
            
            for ent in doc.ents:
                # Focus on technical entities
                if ent.label_ in ["ORG", "PRODUCT", "EVENT", "WORK_OF_ART"]:
                    entity_text = ent.text.strip()
                    
                    if (len(entity_text) >= self.min_term_length and 
                        len(entity_text) <= self.max_term_length):
                        
                        # Try to get definition
                        definition = self._get_wikipedia_definition(entity_text)
                        
                        jargon_term = JargonTerm(
                            term=entity_text,
                            context=text,
                            definition=definition,
                            source=f"spacy_{ent.label_}",
                            confidence=0.7,  # Medium confidence for NER
                            timestamp=timestamp,
                            speaker_id=speaker_id
                        )
                        terms.append(jargon_term)
        
        except Exception as e:
            logger.warning(f"spaCy entity extraction failed: {e}")
        
        return terms
    
    def _get_wikipedia_definition(self, term: str) -> Optional[str]:
        """Get definition from Wikipedia"""
        # Check cache first
        cache_key = term.lower()
        if cache_key in self.term_cache:
            return self.term_cache[cache_key].definition
        
        try:
            # Search Wikipedia
            wikipedia.set_lang("en")
            page = wikipedia.page(term, auto_suggest=True)
            
            # Get first sentence as definition
            summary = wikipedia.summary(term, sentences=2)
            definition = summary.split('.')[0] + '.'
            
            # Cache the result
            self.term_cache[cache_key] = JargonTerm(
                term=term,
                context="",
                definition=definition,
                source="wikipedia",
                confidence=0.8,
                timestamp=datetime.now()
            )
            
            return definition
            
        except Exception as e:
            logger.debug(f"Wikipedia lookup failed for '{term}': {e}")
            return None
    
    def start_processing(self):
        """Start jargon detection processing"""
        if self.is_processing:
            logger.warning("Jargon detection already running")
            return
        
        self.is_processing = True
        self.processing_thread = threading.Thread(target=self._processing_worker)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        logger.info("Jargon detection started")
    
    def stop_processing(self):
        """Stop jargon detection processing"""
        if not self.is_processing:
            return
        
        self.is_processing = False
        
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)
        
        logger.info("Jargon detection stopped")
    
    def get_recent_jargon(self, duration_minutes: float = 5.0) -> List[JargonTerm]:
        """Get recently detected jargon"""
        cutoff_time = datetime.now().timestamp() - (duration_minutes * 60)
        
        recent = []
        for term in reversed(self.detected_jargon):
            if term.timestamp.timestamp() >= cutoff_time:
                recent.append(term)
            else:
                break
        
        return list(reversed(recent))
    
    def get_jargon_summary(self) -> Dict:
        """Get summary of detected jargon"""
        if not self.detected_jargon:
            return {}
        
        # Count terms by source
        source_counts = defaultdict(int)
        for term in self.detected_jargon:
            source_counts[term.source] += 1
        
        # Most common terms
        term_counts = defaultdict(int)
        for term in self.detected_jargon:
            term_counts[term.term.lower()] += 1
        
        most_common = sorted(term_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Terms with definitions
        defined_terms = [term for term in self.detected_jargon if term.definition]
        
        return {
            'total_terms_detected': len(self.detected_jargon),
            'unique_terms': len(term_counts),
            'terms_with_definitions': len(defined_terms),
            'source_breakdown': dict(source_counts),
            'most_common_terms': most_common,
            'recent_terms': [term.term for term in self.get_recent_jargon(5.0)]
        }
    
    def add_custom_glossary(self, name: str, terms: Dict[str, str], 
                           keywords: Set[str]):
        """Add custom industry glossary"""
        self.glossaries[name] = IndustryGlossary(
            name=name,
            terms=terms,
            keywords=keywords
        )
        logger.info(f"Added custom glossary: {name} with {len(terms)} terms")

if __name__ == "__main__":
    # Test jargon detection
    detector = JargonDetector()
    
    def on_jargon_detected(term: JargonTerm):
        print(f"[{term.timestamp.strftime('%H:%M:%S')}] Jargon detected: '{term.term}'")
        print(f"  Source: {term.source} (confidence: {term.confidence:.2f})")
        if term.definition:
            print(f"  Definition: {term.definition[:100]}...")
        print(f"  Context: {term.context[:80]}...")
        print()
    
    detector.add_jargon_callback(on_jargon_detected)
    
    print("Jargon detection test...")
    detector.start_processing()
    
    # Test with sample texts
    test_texts = [
        "We need to implement a REST API with proper CI/CD pipeline for our microservices architecture.",
        "The ROI on this CAPEX investment should improve our EBITDA significantly.",
        "Our lean manufacturing process uses kaizen principles to reduce waste and improve OEE.",
        "The machine learning model uses neural networks and deep learning algorithms.",
        "We should leverage blockchain technology for our IoT devices in the cloud infrastructure."
    ]
    
    for i, text in enumerate(test_texts):
        detector.add_text(text, f"Speaker_{i % 2 + 1}")
        time.sleep(1)
    
    time.sleep(5)  # Wait for processing
    
    detector.stop_processing()
    
    # Show summary
    summary = detector.get_jargon_summary()
    print(f"Jargon Detection Summary:")
    print(f"Total terms detected: {summary.get('total_terms_detected', 0)}")
    print(f"Unique terms: {summary.get('unique_terms', 0)}")
    print(f"Terms with definitions: {summary.get('terms_with_definitions', 0)}")
    print(f"Source breakdown: {summary.get('source_breakdown', {})}")
    print(f"Most common terms: {summary.get('most_common_terms', [])[:5]}")
