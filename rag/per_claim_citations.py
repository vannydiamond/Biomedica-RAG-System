"""
Per-Claim Citation Extraction

Implements proposal requirement: "Source citation per claim"

Instead of generic "9 sources", each fact is attributed to its source.
"""

from typing import List, Dict, Tuple
import re


class Citation:
    """Represents a citation for a fact/claim."""
    
    def __init__(
        self,
        claim: str,
        source_type: str,
        source_id: str,
        relevance_score: float,
        excerpt: str
    ):
        self.claim = claim
        self.source_type = source_type  # "PubMed", "WHO", "CDC", "MedQuAD"
        self.source_id = source_id  # PMID, URL, ID, etc
        self.relevance_score = relevance_score
        self.excerpt = excerpt
    
    def to_dict(self) -> Dict:
        return {
            "claim": self.claim,
            "source_type": self.source_type,
            "source_id": self.source_id,
            "relevance_score": round(self.relevance_score, 4),
            "excerpt": self.excerpt
        }
    
    def to_markdown(self) -> str:
        """Format as markdown citation."""
        return (
            f"[{self.source_type} {self.source_id}]\n"
            f"(Score: {self.relevance_score:.2%})"
        )


class ClaimExtractor:
    """Extract claims/sentences from generated answer."""
    
    @staticmethod
    def extract_claims(answer: str) -> List[str]:
        """
        Extract individual claims/sentences from answer.
        
        Args:
            answer: Generated answer text
        
        Returns:
            List of sentences/claims
        """
        # Split by common sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', answer.strip())
        
        # Clean up
        claims = [s.strip() for s in sentences if len(s.strip()) > 5]
        
        return claims
    
    @staticmethod
    def extract_key_entities(claim: str) -> set:
        """
        Extract key entities from a claim for matching.
        
        Args:
            claim: Single claim/sentence
        
        Returns:
            Set of important words
        """
        # Remove punctuation
        text = re.sub(r'[^\w\s]', ' ', claim.lower())
        words = text.split()
        
        # Filter stopwords
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were',
            'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'can', 'in', 'on',
            'at', 'to', 'for', 'of', 'with', 'by', 'from', 'it', 'its', 'that',
            'this', 'these', 'those', 'which', 'who', 'what', 'where', 'when',
        }
        
        return set(w for w in words if len(w) > 2 and w not in stopwords)


class CitationMatcher:
    """Match claims to source chunks."""
    
    @staticmethod
    def calculate_overlap(
        claim_entities: set,
        chunk_text: str
    ) -> float:
        """
        Calculate how well a chunk supports a claim.
        
        Args:
            claim_entities: Key entities from claim
            chunk_text: Source chunk text
        
        Returns:
            Overlap score (0.0-1.0)
        """
        # Get entities from chunk
        chunk_clean = re.sub(r'[^\w\s]', ' ', chunk_text.lower())
        chunk_words = set(chunk_clean.split())
        
        # Calculate overlap
        matching = len(claim_entities & chunk_words)
        total = len(claim_entities)
        
        return matching / total if total > 0 else 0.0
    
    @staticmethod
    def find_supporting_chunk(
        claim: str,
        retrieved_chunks: List[Dict],
        min_overlap: float = 0.3
    ) -> Tuple[Optional[Dict], float]:
        """
        Find the best supporting chunk for a claim.
        
        Args:
            claim: Claim/sentence to support
            retrieved_chunks: List of retrieved chunks with metadata
            min_overlap: Minimum overlap required
        
        Returns:
            (best_chunk, overlap_score) or (None, 0.0)
        """
        claim_entities = ClaimExtractor.extract_key_entities(claim)
        
        if not claim_entities:
            return None, 0.0
        
        best_chunk = None
        best_overlap = 0.0
        
        for chunk in retrieved_chunks:
            chunk_text = chunk.get('content', '')
            overlap = CitationMatcher.calculate_overlap(claim_entities, chunk_text)
            
            if overlap > best_overlap:
                best_overlap = overlap
                best_chunk = chunk
        
        if best_overlap >= min_overlap:
            return best_chunk, best_overlap
        
        return None, 0.0


class AnswerCiter:
    """Add citations to an answer."""
    
    @staticmethod
    def cite_answer(
        answer: str,
        retrieved_chunks: List[Dict],
        min_overlap: float = 0.3
    ) -> Dict:
        """
        Add citations to answer claims.
        
        Args:
            answer: Generated answer text
            retrieved_chunks: Retrieved source chunks
            min_overlap: Minimum overlap for citation
        
        Returns:
            {
                "answer": "...",
                "claims_with_citations": [
                    {
                        "claim": "...",
                        "citation": {...}
                    }
                ],
                "citation_rate": 0.75  # % of claims cited
            }
        """
        # Extract claims
        claims = ClaimExtractor.extract_claims(answer)
        
        if not claims:
            return {
                "answer": answer,
                "claims_with_citations": [],
                "citation_rate": 0.0
            }
        
        # Match each claim to sources
        claims_with_citations = []
        cited_count = 0
        
        matcher = CitationMatcher()
        
        for claim in claims:
            supporting_chunk, overlap = matcher.find_supporting_chunk(
                claim,
                retrieved_chunks,
                min_overlap
            )
            
            if supporting_chunk:
                citation = Citation(
                    claim=claim,
                    source_type=supporting_chunk.get('source_type', 'Unknown'),
                    source_id=supporting_chunk.get('source_id', 'N/A'),
                    relevance_score=supporting_chunk.get('score', 0.0),
                    excerpt=supporting_chunk.get('content', '')[:150]
                )
                claims_with_citations.append({
                    "claim": claim,
                    "citation": citation.to_dict(),
                    "overlap_score": overlap
                })
                cited_count += 1
            else:
                claims_with_citations.append({
                    "claim": claim,
                    "citation": None,
                    "overlap_score": 0.0
                })
        
        citation_rate = cited_count / len(claims) if claims else 0.0
        
        return {
            "answer": answer,
            "claims_with_citations": claims_with_citations,
            "citation_rate": round(citation_rate, 2),
            "claims_cited": cited_count,
            "total_claims": len(claims)
        }
    
    @staticmethod
    def format_answer_with_citations(
        answer: str,
        claims_with_citations: List[Dict]
    ) -> str:
        """
        Format answer with inline citations in markdown.
        
        Example:
        "Type 2 diabetes is associated with insulin resistance.
         [PubMed PMID:123456 - 0.92]"
        """
        if not claims_with_citations:
            return answer
        
        # Build citation map: claim -> citation
        citation_map = {}
        for item in claims_with_citations:
            if item.get('citation'):
                citation = item['citation']
                key = item['claim']
                citation_map[key] = (
                    f"[{citation['source_type']} {citation['source_id']} - "
                    f"{citation['relevance_score']:.2%}]"
                )
        
        # Add citations to answer
        formatted = answer
        for claim, citation_str in citation_map.items():
            # Add citation after claim
            formatted = formatted.replace(
                claim,
                f"{claim}\n{citation_str}"
            )
        
        return formatted


def generate_citation_summary(
    answer_dict: Dict
) -> str:
    """
    Generate a summary of sources used.
    
    Args:
        answer_dict: Output from cite_answer()
    
    Returns:
        Formatted summary
    """
    claims_with_cites = answer_dict['claims_with_citations']
    
    # Collect unique sources
    sources = {}
    for item in claims_with_cites:
        if item.get('citation'):
            cit = item['citation']
            source_key = f"{cit['source_type']} {cit['source_id']}"
            if source_key not in sources:
                sources[source_key] = {
                    "type": cit['source_type'],
                    "id": cit['source_id'],
                    "score": cit['relevance_score']
                }
    
    # Format
    summary = "**Sources Used:**\n\n"
    for i, (key, source) in enumerate(sources.items(), 1):
        summary += (
            f"{i}. {source['type']} {source['id']}\n"
            f"   Relevance: {source['score']:.0%}\n"
        )
    
    citation_rate = answer_dict.get('citation_rate', 0)
    summary += f"\n**Citation Rate:** {citation_rate:.0%} of claims"
    
    return summary
