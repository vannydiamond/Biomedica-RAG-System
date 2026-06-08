"""
PHASE 3: Grounding Validation

Checks whether answers are supported by retrieved chunks.
Displays evidence clearly to users.
"""

from typing import List, Dict, Tuple
import re


class GroundingValidator:
    """
    Validates whether an answer is grounded in retrieved chunks.
    """
    
    @staticmethod
    def extract_key_entities(text: str) -> set:
        """
        Extract key entities (words/phrases) from text.
        Simple keyword extraction - can be improved with NER.
        """
        # Remove punctuation and convert to lowercase
        text_clean = re.sub(r'[^\w\s]', ' ', text.lower())
        words = text_clean.split()
        
        # Filter out common stopwords
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were',
            'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'can',
            'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from',
            'it', 'its', 'that', 'this', 'these', 'those', 'which', 'who',
            'what', 'where', 'when', 'why', 'how', 'not', 'no', 'yes',
            'i', 'you', 'he', 'she', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
        }
        
        return set(w for w in words if len(w) > 2 and w not in stopwords)
    
    @staticmethod
    def calculate_overlap(answer: str, chunks: List[Dict]) -> Tuple[float, List[str]]:
        """
        Calculate how much of the answer is covered by chunks.
        
        Args:
            answer: Generated answer
            chunks: List of retrieved chunks with 'content' key
        
        Returns:
            (overlap_score, matched_phrases)
        """
        answer_entities = GroundingValidator.extract_key_entities(answer)
        
        if not answer_entities:
            return 0.0, []
        
        chunk_text = " ".join(chunk.get('content', '') for chunk in chunks)
        chunk_entities = GroundingValidator.extract_key_entities(chunk_text)
        
        # Calculate overlap
        matching_entities = answer_entities & chunk_entities
        overlap_score = len(matching_entities) / len(answer_entities)
        
        return overlap_score, list(matching_entities)
    
    @staticmethod
    def is_grounded(answer: str, chunks: List[Dict], threshold: float = 0.5) -> bool:
        """
        Determine if answer is grounded in chunks.
        
        Args:
            answer: Generated answer
            chunks: List of retrieved chunks
            threshold: Minimum overlap score (0.0-1.0)
        
        Returns:
            True if answer is grounded
        """
        if not chunks:
            return False
        
        overlap_score, _ = GroundingValidator.calculate_overlap(answer, chunks)
        return overlap_score >= threshold
    
    @staticmethod
    def detect_hallucinations(answer: str, chunks: List[Dict]) -> List[str]:
        """
        Detect potential hallucinations (facts not in chunks).
        
        Args:
            answer: Generated answer
            chunks: List of retrieved chunks
        
        Returns:
            List of suspicious phrases
        """
        # Extract sentences from answer
        sentences = re.split(r'[.!?]+', answer)
        
        chunk_text = " ".join(chunk.get('content', '') for chunk in chunks)
        chunk_entities = GroundingValidator.extract_key_entities(chunk_text)
        
        suspicious = []
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence or len(sentence) < 5:
                continue
            
            entities = GroundingValidator.extract_key_entities(sentence)
            coverage = len(entities & chunk_entities) / max(len(entities), 1)
            
            # If less than 30% of entities are in chunks, flag it
            if coverage < 0.3 and len(entities) > 2:
                suspicious.append(sentence)
        
        return suspicious


def format_evidence_display(chunks: List[Dict], max_chunks: int = 5) -> str:
    """
    Format retrieved chunks as human-readable evidence.
    
    Returns:
        Formatted string for display
    """
    if not chunks:
        return "No evidence retrieved from the document."
    
    display = "**Retrieved Evidence**\n\n"
    
    for i, chunk in enumerate(chunks[:max_chunks], 1):
        score = chunk.get('score', 0)
        content = chunk.get('content', '')
        
        # Create a visual score indicator
        score_bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
        
        display += f"**Chunk {i}** | Relevance: {score:.2%} [{score_bar}]\n\n"
        display += f"> {content[:500]}"
        if len(content) > 500:
            display += "..."
        display += "\n\n---\n\n"
    
    return display


def generate_grounding_report(
    question: str,
    answer: str,
    chunks: List[Dict],
    mode: str = "document"
) -> Dict:
    """
    Generate a grounding report.
    
    Returns:
        Dict with grounding analysis
    """
    overlap_score, matched_entities = GroundingValidator.calculate_overlap(answer, chunks)
    is_grounded = GroundingValidator.is_grounded(answer, chunks)
    hallucinations = GroundingValidator.detect_hallucinations(answer, chunks)
    
    report = {
        "question": question,
        "answer_length": len(answer.split()),
        "chunks_used": len(chunks),
        "overlap_score": overlap_score,
        "is_grounded": is_grounded,
        "matched_entities": matched_entities,
        "potential_hallucinations": hallucinations,
        "confidence": {
            "grounding": "HIGH" if overlap_score > 0.7 else "MEDIUM" if overlap_score > 0.4 else "LOW",
            "recommendation": (
                "Answer is well-grounded in the document." if is_grounded
                else "⚠️ Answer may contain information not in the document. "
                     "Verify with the original source."
            )
        }
    }
    
    return report
