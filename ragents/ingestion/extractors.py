"""Content extraction and analysis utilities."""

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import hashlib

from ..rag.types import Document


class MetadataExtractor:
    """Extract metadata from files and documents."""

    async def extract(self, file_path: Path, document: Document) -> Dict[str, Any]:
        """Extract comprehensive metadata from file and document."""
        metadata = {}

        # File system metadata
        file_stat = file_path.stat()
        metadata.update({
            "file_name": file_path.name,
            "file_stem": file_path.stem,
            "file_extension": file_path.suffix.lower(),
            "file_size_bytes": file_stat.st_size,
            "file_size_mb": round(file_stat.st_size / (1024 * 1024), 2),
            "created_timestamp": file_stat.st_ctime,
            "modified_timestamp": file_stat.st_mtime,
            "accessed_timestamp": file_stat.st_atime,
        })

        # Content hash for deduplication
        content_hash = hashlib.sha256(document.content.encode('utf-8')).hexdigest()
        metadata["content_hash"] = content_hash
        metadata["content_hash_short"] = content_hash[:16]

        # Content statistics
        content_stats = self._analyze_content_stats(document.content)
        metadata.update(content_stats)

        # Language detection
        language_info = await self._detect_language(document.content)
        metadata.update(language_info)

        # Content structure analysis
        structure_info = self._analyze_content_structure(document.content)
        metadata.update(structure_info)

        # Extract key phrases and topics
        if len(document.content) > 100:
            key_info = await self._extract_key_information(document.content)
            metadata.update(key_info)

        return metadata

    def _analyze_content_stats(self, content: str) -> Dict[str, Any]:
        """Analyze basic content statistics."""
        lines = content.split('\n')
        words = content.split()
        sentences = re.split(r'[.!?]+', content)

        return {
            "content_length": len(content),
            "line_count": len(lines),
            "word_count": len(words),
            "sentence_count": len([s for s in sentences if s.strip()]),
            "paragraph_count": len([line for line in lines if line.strip()]),
            "avg_words_per_sentence": len(words) / max(len(sentences), 1),
            "avg_chars_per_word": len(content) / max(len(words), 1),
            "whitespace_ratio": (content.count(' ') + content.count('\n') + content.count('\t')) / max(len(content), 1)
        }

    async def _detect_language(self, content: str) -> Dict[str, Any]:
        """Detect the language of the content."""
        try:
            from langdetect import detect, detect_langs
        except ImportError:
            return {"language": "unknown", "language_confidence": 0.0}

        try:
            # Get primary language
            primary_lang = detect(content[:1000])  # Use first 1000 chars for speed

            # Get confidence scores for multiple languages
            lang_probs = detect_langs(content[:1000])
            lang_scores = {lang.lang: lang.prob for lang in lang_probs}

            return {
                "language": primary_lang,
                "language_confidence": lang_scores.get(primary_lang, 0.0),
                "language_alternatives": dict(list(lang_scores.items())[:3])
            }
        except Exception:
            return {"language": "unknown", "language_confidence": 0.0}

    def _analyze_content_structure(self, content: str) -> Dict[str, Any]:
        """Analyze the structure of the content."""
        structure_info = {}

        # Check for common structural elements
        structure_info["has_headers"] = bool(re.search(r'^#+\s', content, re.MULTILINE))
        structure_info["has_bullet_points"] = bool(re.search(r'^\s*[-*•]\s', content, re.MULTILINE))
        structure_info["has_numbered_lists"] = bool(re.search(r'^\s*\d+\.\s', content, re.MULTILINE))
        structure_info["has_urls"] = bool(re.search(r'https?://\S+', content))
        structure_info["has_emails"] = bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', content))
        structure_info["has_phone_numbers"] = bool(re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', content))
        structure_info["has_dates"] = bool(re.search(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', content))

        # Count specific elements
        structure_info["url_count"] = len(re.findall(r'https?://\S+', content))
        structure_info["email_count"] = len(re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', content))
        structure_info["header_count"] = len(re.findall(r'^#+\s', content, re.MULTILINE))

        # Analyze formatting
        structure_info["has_bold_text"] = '**' in content or '__' in content
        structure_info["has_italic_text"] = '*' in content and '**' not in content
        structure_info["has_code_blocks"] = '```' in content or '    ' in content

        return structure_info

    async def _extract_key_information(self, content: str) -> Dict[str, Any]:
        """Extract key information like keywords and topics."""
        key_info = {}

        # Extract potential keywords (simple approach)
        words = re.findall(r'\b[a-zA-Z]{4,}\b', content.lower())
        word_freq = {}
        for word in words:
            if word not in ['this', 'that', 'with', 'have', 'will', 'from', 'they', 'been', 'were', 'said', 'each', 'which']:
                word_freq[word] = word_freq.get(word, 0) + 1

        # Get top keywords
        top_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        key_info["top_keywords"] = [word for word, count in top_keywords]
        key_info["keyword_frequencies"] = dict(top_keywords)

        # Extract entities (simple pattern matching)
        entities = self._extract_simple_entities(content)
        key_info.update(entities)

        return key_info

    def _extract_simple_entities(self, content: str) -> Dict[str, List[str]]:
        """Extract simple entities using pattern matching."""
        entities = {
            "organizations": [],
            "locations": [],
            "persons": [],
            "dates": [],
            "numbers": []
        }

        # Common organization indicators
        org_patterns = [
            r'\b[A-Z][a-z]+ (?:Inc|Corp|LLC|Ltd|Company|Corporation)\b',
            r'\b[A-Z]{2,}\b',  # Acronyms
        ]
        for pattern in org_patterns:
            entities["organizations"].extend(re.findall(pattern, content))

        # Dates
        date_patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2},? \d{4}\b'
        ]
        for pattern in date_patterns:
            entities["dates"].extend(re.findall(pattern, content))

        # Numbers
        number_patterns = [
            r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b',
            r'\b\d+%\b'
        ]
        for pattern in number_patterns:
            entities["numbers"].extend(re.findall(pattern, content))

        # Clean up duplicates and limit results
        for key in entities:
            entities[key] = list(set(entities[key]))[:10]

        return entities


class ContentAnalyzer:
    """Analyze document content for quality and characteristics."""

    async def analyze(self, document: Document) -> Dict[str, Any]:
        """Perform comprehensive content analysis."""
        analysis = {}

        # Quality metrics
        quality_metrics = self._analyze_quality(document.content)
        analysis.update(quality_metrics)

        # Content classification
        content_type = self._classify_content_type(document.content)
        analysis["content_type"] = content_type

        # Readability analysis
        readability = self._analyze_readability(document.content)
        analysis.update(readability)

        # Topic analysis
        topics = await self._analyze_topics(document.content)
        analysis.update(topics)

        return analysis

    def _analyze_quality(self, content: str) -> Dict[str, Any]:
        """Analyze content quality metrics."""
        quality = {}

        # Basic quality indicators
        total_chars = len(content)
        if total_chars == 0:
            return {"content_quality_score": 0.0}

        # Calculate various quality metrics
        uppercase_ratio = sum(1 for c in content if c.isupper()) / total_chars
        lowercase_ratio = sum(1 for c in content if c.islower()) / total_chars
        digit_ratio = sum(1 for c in content if c.isdigit()) / total_chars
        punctuation_ratio = sum(1 for c in content if c in '.,;:!?') / total_chars

        # Repetitive content detection
        words = content.split()
        unique_words = set(words)
        word_diversity = len(unique_words) / max(len(words), 1)

        # Sentence structure
        sentences = re.split(r'[.!?]+', content)
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)

        # Overall quality score (0-1)
        quality_score = 0.0

        # Penalize extreme ratios
        if 0.1 <= uppercase_ratio <= 0.3:
            quality_score += 0.2
        if lowercase_ratio >= 0.5:
            quality_score += 0.2
        if 0.05 <= punctuation_ratio <= 0.15:
            quality_score += 0.2
        if word_diversity >= 0.5:
            quality_score += 0.2
        if 10 <= avg_sentence_length <= 25:
            quality_score += 0.2

        quality.update({
            "content_quality_score": quality_score,
            "uppercase_ratio": uppercase_ratio,
            "lowercase_ratio": lowercase_ratio,
            "digit_ratio": digit_ratio,
            "punctuation_ratio": punctuation_ratio,
            "word_diversity": word_diversity,
            "avg_sentence_length": avg_sentence_length
        })

        return quality

    def _classify_content_type(self, content: str) -> str:
        """Classify the type of content."""
        content_lower = content.lower()

        # Check for specific content patterns
        if any(word in content_lower for word in ['abstract', 'introduction', 'methodology', 'conclusion', 'references']):
            return "academic_paper"
        elif any(word in content_lower for word in ['recipe', 'ingredients', 'instructions', 'cooking', 'bake']):
            return "recipe"
        elif any(word in content_lower for word in ['dear', 'sincerely', 'best regards', 'yours truly']):
            return "letter"
        elif content.count('\n') / max(len(content), 1) > 0.1:
            return "structured_document"
        elif re.search(r'^\d+\.\s', content, re.MULTILINE):
            return "numbered_list"
        elif re.search(r'^[-*•]\s', content, re.MULTILINE):
            return "bulleted_list"
        elif content.count('?') > content.count('.'):
            return "faq"
        else:
            return "general_text"

    def _analyze_readability(self, content: str) -> Dict[str, Any]:
        """Analyze readability metrics."""
        sentences = re.split(r'[.!?]+', content)
        words = content.split()
        syllables = self._count_syllables(content)

        if not sentences or not words:
            return {"readability_score": 0.0}

        avg_sentence_length = len(words) / len(sentences)
        avg_syllables_per_word = syllables / len(words)

        # Simplified Flesch Reading Ease
        flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        flesch_score = max(0, min(100, flesch_score))

        # Reading level
        if flesch_score >= 90:
            reading_level = "very_easy"
        elif flesch_score >= 80:
            reading_level = "easy"
        elif flesch_score >= 70:
            reading_level = "fairly_easy"
        elif flesch_score >= 60:
            reading_level = "standard"
        elif flesch_score >= 50:
            reading_level = "fairly_difficult"
        elif flesch_score >= 30:
            reading_level = "difficult"
        else:
            reading_level = "very_difficult"

        return {
            "readability_score": flesch_score,
            "reading_level": reading_level,
            "avg_sentence_length": avg_sentence_length,
            "avg_syllables_per_word": avg_syllables_per_word
        }

    def _count_syllables(self, text: str) -> int:
        """Estimate syllable count in text."""
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        syllable_count = 0

        for word in words:
            # Simple syllable counting heuristic
            vowels = 'aeiouy'
            count = 0
            prev_was_vowel = False

            for char in word:
                is_vowel = char in vowels
                if is_vowel and not prev_was_vowel:
                    count += 1
                prev_was_vowel = is_vowel

            # Handle silent 'e'
            if word.endswith('e') and count > 1:
                count -= 1

            # Every word has at least 1 syllable
            syllable_count += max(1, count)

        return syllable_count

    async def _analyze_topics(self, content: str) -> Dict[str, Any]:
        """Analyze topics in the content."""
        topics = {}

        # Simple topic detection based on keywords
        topic_keywords = {
            "technology": ["software", "computer", "digital", "internet", "algorithm", "data", "programming"],
            "science": ["research", "study", "analysis", "experiment", "hypothesis", "theory", "scientific"],
            "business": ["company", "market", "revenue", "profit", "customer", "strategy", "business"],
            "education": ["student", "learning", "course", "education", "teacher", "curriculum", "academic"],
            "health": ["medical", "health", "treatment", "patient", "disease", "medicine", "clinical"],
            "finance": ["investment", "financial", "money", "bank", "economic", "budget", "cost"]
        }

        content_lower = content.lower()
        topic_scores = {}

        for topic, keywords in topic_keywords.items():
            score = sum(content_lower.count(keyword) for keyword in keywords)
            if score > 0:
                topic_scores[topic] = score

        # Determine primary topic
        if topic_scores:
            primary_topic = max(topic_scores, key=topic_scores.get)
            topics["primary_topic"] = primary_topic
            topics["topic_scores"] = topic_scores
        else:
            topics["primary_topic"] = "general"
            topics["topic_scores"] = {}

        return topics


class TextExtractor:
    """Extract clean text from various content types."""

    def extract_clean_text(self, content: str, preserve_structure: bool = True) -> str:
        """Extract clean, readable text from content."""
        # Remove excessive whitespace
        content = re.sub(r'\s+', ' ', content)

        # Remove special characters but preserve basic punctuation
        if not preserve_structure:
            content = re.sub(r'[^\w\s.,;:!?-]', '', content)

        # Normalize line breaks
        content = re.sub(r'\n\s*\n', '\n\n', content)

        return content.strip()

    def extract_sentences(self, content: str) -> List[str]:
        """Extract individual sentences from content."""
        sentences = re.split(r'[.!?]+', content)
        return [s.strip() for s in sentences if s.strip()]

    def extract_paragraphs(self, content: str) -> List[str]:
        """Extract paragraphs from content."""
        paragraphs = content.split('\n\n')
        return [p.strip() for p in paragraphs if p.strip()]

    def extract_key_phrases(self, content: str, max_phrases: int = 20) -> List[str]:
        """Extract key phrases from content."""
        # Simple n-gram extraction
        words = re.findall(r'\b[a-zA-Z]+\b', content.lower())

        # Extract 2-grams and 3-grams
        phrases = []

        # 2-grams
        for i in range(len(words) - 1):
            phrase = f"{words[i]} {words[i+1]}"
            phrases.append(phrase)

        # 3-grams
        for i in range(len(words) - 2):
            phrase = f"{words[i]} {words[i+1]} {words[i+2]}"
            phrases.append(phrase)

        # Count frequency
        phrase_freq = {}
        for phrase in phrases:
            phrase_freq[phrase] = phrase_freq.get(phrase, 0) + 1

        # Return top phrases
        top_phrases = sorted(phrase_freq.items(), key=lambda x: x[1], reverse=True)
        return [phrase for phrase, count in top_phrases[:max_phrases] if count > 1]