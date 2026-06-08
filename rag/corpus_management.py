"""
Corpus Configuration & Management

Aligns with proposal: Fixed medical corpora (PubMed, WHO, CDC, MedQuAD)
NOT arbitrary document uploads
"""

from dataclasses import dataclass
from typing import Dict, List
from enum import Enum


class CorpusType(str, Enum):
    """Allowed medical corpora per proposal."""
    PUBMED = "PubMed"
    WHO = "WHO"
    CDC = "CDC"
    MEDQUAD = "MedQuAD"


@dataclass
class CorpusMetadata:
    """Metadata for a medical corpus."""
    name: CorpusType
    description: str
    estimated_chunks: int
    chunk_size: int
    overlap: int
    embedding_model: str
    last_updated: str
    source_url: str


# Corpus Configurations
MEDICAL_CORPORA: Dict[CorpusType, CorpusMetadata] = {
    CorpusType.PUBMED: CorpusMetadata(
        name=CorpusType.PUBMED,
        description="PubMed abstracts from MEDLINE database",
        estimated_chunks=320000,
        chunk_size=500,
        overlap=100,
        embedding_model="BioLORD",
        last_updated="June 2026",
        source_url="https://pubmed.ncbi.nlm.nih.gov/"
    ),
    CorpusType.WHO: CorpusMetadata(
        name=CorpusType.WHO,
        description="World Health Organization clinical guidelines",
        estimated_chunks=15000,
        chunk_size=500,
        overlap=100,
        embedding_model="BioLORD",
        last_updated="June 2026",
        source_url="https://www.who.int/"
    ),
    CorpusType.CDC: CorpusMetadata(
        name=CorpusType.CDC,
        description="Centers for Disease Control disease guidelines",
        estimated_chunks=8000,
        chunk_size=500,
        overlap=100,
        embedding_model="BioLORD",
        last_updated="June 2026",
        source_url="https://www.cdc.gov/"
    ),
    CorpusType.MEDQUAD: CorpusMetadata(
        name=CorpusType.MEDQUAD,
        description="Medical question-answer dataset",
        estimated_chunks=47000,
        chunk_size=300,
        overlap=50,
        embedding_model="BioLORD",
        last_updated="June 2026",
        source_url="https://github.com/abachaa/MedQuAD"
    ),
}


class CorpusManager:
    """Manages medical corpus ingestion and configuration."""
    
    def __init__(self, allowed_corpora: List[CorpusType] = None):
        """
        Initialize corpus manager.
        
        Args:
            allowed_corpora: Which corpora to allow. 
                           If None, allows all per proposal.
        """
        if allowed_corpora is None:
            self.allowed_corpora = list(CorpusType)
        else:
            self.allowed_corpora = allowed_corpora
    
    def validate_corpus(self, corpus_name: str) -> bool:
        """Check if corpus is in the approved list."""
        try:
            corpus_type = CorpusType(corpus_name)
            return corpus_type in self.allowed_corpora
        except ValueError:
            return False
    
    def get_corpus_config(self, corpus_name: CorpusType) -> CorpusMetadata:
        """Get configuration for a corpus."""
        return MEDICAL_CORPORA[corpus_name]
    
    def get_all_corpus_stats(self) -> Dict:
        """Get statistics for all indexed corpora."""
        total_chunks = 0
        corpora_info = []
        
        for corpus_type in self.allowed_corpora:
            config = MEDICAL_CORPORA[corpus_type]
            corpora_info.append({
                "name": config.name.value,
                "chunks": config.estimated_chunks,
                "description": config.description,
            })
            total_chunks += config.estimated_chunks
        
        return {
            "corpora": corpora_info,
            "total_chunks": total_chunks,
            "embedding_model": "BioLORD",
            "retrieval_method": "Hybrid (Dense + Sparse + RRF)",
            "last_updated": "June 2026"
        }


def ingest_corpus(corpus_name: str, data: bytes, corpus_manager: CorpusManager) -> bool:
    """
    Ingest a medical corpus.
    
    Args:
        corpus_name: Name of corpus (must be in MEDICAL_CORPORA)
        data: Corpus data
        corpus_manager: Corpus manager instance
    
    Returns:
        True if ingestion successful
    
    Raises:
        ValueError: If corpus not in approved list
    """
    if not corpus_manager.validate_corpus(corpus_name):
        raise ValueError(
            f"Corpus '{corpus_name}' not in approved list. "
            f"Allowed: {[c.value for c in corpus_manager.allowed_corpora]}"
        )
    
    corpus_type = CorpusType(corpus_name)
    config = corpus_manager.get_corpus_config(corpus_type)
    
    # Ingestion logic here
    # - Parse corpus format
    # - Chunk with config.chunk_size and config.overlap
    # - Embed with BioLORD
    # - Index with FAISS + BM25
    
    return True


def ingest_test_document(file_path: str) -> bool:
    """
    DEVELOPMENT ONLY: Ingest arbitrary document for testing.
    
    Must be explicitly enabled and clearly marked as development mode.
    Not used in evaluation.
    """
    import os
    if os.environ.get("MODE") != "DEVELOPMENT":
        raise ValueError("Test document ingestion only allowed in DEVELOPMENT mode")
    
    # Test ingestion logic
    return True
