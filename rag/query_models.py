"""
Updated Query Models - Aligned to Proposal

The proposal specifies:
- Medical questions only
- From medical corpora (PubMed, WHO, CDC, MedQuAD)
- Grounded answers only
- Per-claim citations
- Confidence indicators
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from enum import Enum


class ConfidenceLevel(str, Enum):
    """Confidence levels from proposal."""
    NONE = "none"  # No retrieval or refusal
    LOW = "low"  # avg_score < 0.6
    MEDIUM = "medium"  # 0.6 <= avg_score < 0.8
    HIGH = "high"  # avg_score >= 0.8


class SourceAttribution(BaseModel):
    """Attribution for a single source."""
    source_type: str  # "PubMed", "WHO", "CDC", "MedQuAD"
    source_id: str  # PMID, guideline ID, etc.
    relevance_score: float = Field(ge=0.0, le=1.0)
    excerpt: str = Field(description="First 200 chars of source")


class ClaimWithCitation(BaseModel):
    """A claim/sentence with its source citation."""
    claim: str
    citation: Optional[SourceAttribution] = None
    overlap_score: float = Field(ge=0.0, le=1.0, default=0.0)


class RetrievalDetails(BaseModel):
    """Details about the retrieval pipeline executed."""
    query_tokens: int
    dense_retrieved: int = Field(description="Chunks found by FAISS")
    sparse_retrieved: int = Field(description="Chunks found by BM25")
    after_fusion: int = Field(description="After RRF fusion")
    final_selected: int = Field(description="Final top-K")
    avg_score: float
    top_score: float


class MedicalQARequest(BaseModel):
    """Request for medical question-answering."""
    question: str = Field(
        description="Medical question from patient/clinician"
    )
    max_sources: int = Field(default=5, ge=1, le=10)
    
    class Config:
        example = {
            "question": "What are the main symptoms of type 2 diabetes?",
            "max_sources": 5
        }


class MedicalQAResponse(BaseModel):
    """
    Response from medical QA system.
    
    Aligned with proposal requirements:
    - Answer (grounded only)
    - Confidence indicator
    - Per-claim citations
    - Retrieval transparency
    """
    
    # Basic
    question: str
    answer: str
    
    # Confidence (from proposal)
    confidence: ConfidenceLevel
    grounded: bool = Field(
        description="Whether answer is grounded in retrieved medical evidence"
    )
    
    # Citations (per proposal: "Source citation per claim")
    claims_with_citations: List[ClaimWithCitation]
    citation_rate: float = Field(
        ge=0.0, le=1.0,
        description="Fraction of claims with citations"
    )
    
    # Transparency (show pipeline)
    retrieval_details: RetrievalDetails
    
    # Safety
    is_refusal: bool = Field(
        default=False,
        description="Whether system refused to answer (e.g., diagnosis request)"
    )
    refusal_reason: Optional[str] = None
    
    class Config:
        example = {
            "question": "What are symptoms of type 2 diabetes?",
            "answer": "Type 2 diabetes is characterized by...",
            "confidence": "high",
            "grounded": True,
            "claims_with_citations": [
                {
                    "claim": "Type 2 diabetes is characterized by insulin resistance.",
                    "citation": {
                        "source_type": "PubMed",
                        "source_id": "PMID:31234567",
                        "relevance_score": 0.92,
                        "excerpt": "Type 2 diabetes involves..."
                    },
                    "overlap_score": 0.85
                }
            ],
            "citation_rate": 0.95,
            "retrieval_details": {
                "query_tokens": 8,
                "dense_retrieved": 25,
                "sparse_retrieved": 18,
                "after_fusion": 40,
                "final_selected": 5,
                "avg_score": 0.89,
                "top_score": 0.95
            },
            "is_refusal": False
        }


class CorpusStatsResponse(BaseModel):
    """Statistics about indexed medical corpora."""
    
    class CorpusInfo(BaseModel):
        name: str  # "PubMed", "WHO", "CDC", "MedQuAD"
        chunks: int
        description: str
    
    corpora: List[CorpusInfo]
    total_chunks: int
    embedding_model: str
    retrieval_method: str
    last_updated: str


class SystemStatusResponse(BaseModel):
    """System status and configuration."""
    
    status: str = "ready"
    version: str
    corpora: CorpusStatsResponse
    
    class Config:
        example = {
            "status": "ready",
            "version": "1.0.0",
            "corpora": {
                "corpora": [
                    {"name": "PubMed", "chunks": 320000, "description": "..."},
                    {"name": "WHO", "chunks": 15000, "description": "..."},
                    {"name": "CDC", "chunks": 8000, "description": "..."},
                    {"name": "MedQuAD", "chunks": 47000, "description": "..."}
                ],
                "total_chunks": 390000,
                "embedding_model": "BioLORD",
                "retrieval_method": "Hybrid (Dense + Sparse + RRF)",
                "last_updated": "June 2026"
            }
        }


# Refusal Response
class RefusalResponse(MedicalQAResponse):
    """Specialized response for refused queries."""
    
    def __init__(
        self,
        question: str,
        refusal_reason: str,
        confidence: ConfidenceLevel = ConfidenceLevel.NONE
    ):
        super().__init__(
            question=question,
            answer=refusal_reason,
            confidence=confidence,
            grounded=False,
            claims_with_citations=[],
            citation_rate=0.0,
            retrieval_details=RetrievalDetails(
                query_tokens=len(question.split()),
                dense_retrieved=0,
                sparse_retrieved=0,
                after_fusion=0,
                final_selected=0,
                avg_score=0.0,
                top_score=0.0
            ),
            is_refusal=True,
            refusal_reason=refusal_reason
        )


# Out-of-Domain Response
class OutOfDomainResponse(MedicalQAResponse):
    """Response when query has no medical content or no retrieval."""
    
    def __init__(
        self,
        question: str,
        reason: str = "No relevant medical information found"
    ):
        super().__init__(
            question=question,
            answer=f"I don't have that information in my current knowledge base. "
                   f"Please consult a qualified healthcare professional.",
            confidence=ConfidenceLevel.NONE,
            grounded=False,
            claims_with_citations=[],
            citation_rate=0.0,
            retrieval_details=RetrievalDetails(
                query_tokens=len(question.split()),
                dense_retrieved=0,
                sparse_retrieved=0,
                after_fusion=0,
                final_selected=0,
                avg_score=0.0,
                top_score=0.0
            ),
            is_refusal=False,
            refusal_reason=reason
        )
