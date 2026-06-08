"""
PHASE 4: Improve Retrieval

Utilities for:
- Chunk deduplication
- Chunk size optimization
- Retrieval score inspection
- Retrieval debugging
"""

from typing import List, Tuple, Dict
from rag.document import BiomedicalDocument
from difflib import SequenceMatcher


class ChunkOptimizer:
    """Optimize chunks for retrieval."""
    
    @staticmethod
    def remove_duplicate_chunks(
        chunks: List[BiomedicalDocument],
        similarity_threshold: float = 0.95
    ) -> List[BiomedicalDocument]:
        """
        Remove duplicate or near-duplicate chunks.
        
        Args:
            chunks: List of chunks to deduplicate
            similarity_threshold: Similarity score (0-1) above which chunks are considered duplicates
        
        Returns:
            List of unique chunks
        """
        if not chunks:
            return []
        
        unique_chunks = []
        
        for chunk in chunks:
            is_duplicate = False
            
            for existing_chunk in unique_chunks:
                # Calculate similarity
                similarity = SequenceMatcher(
                    None,
                    chunk.content,
                    existing_chunk.content
                ).ratio()
                
                if similarity >= similarity_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_chunks.append(chunk)
        
        return unique_chunks
    
    @staticmethod
    def remove_empty_chunks(chunks: List[BiomedicalDocument]) -> List[BiomedicalDocument]:
        """Remove chunks that are empty or have only whitespace."""
        return [c for c in chunks if c.content.strip() and len(c.content.split()) > 2]
    
    @staticmethod
    def analyze_chunk_sizes(chunks: List[BiomedicalDocument]) -> Dict:
        """
        Analyze chunk size distribution.
        
        Returns:
            Stats about chunk sizes
        """
        if not chunks:
            return {"count": 0}
        
        sizes = [len(c.content.split()) for c in chunks]
        
        return {
            "count": len(chunks),
            "min_words": min(sizes),
            "max_words": max(sizes),
            "avg_words": sum(sizes) / len(sizes),
            "total_words": sum(sizes),
        }
    
    @staticmethod
    def recommend_chunk_size(text_length: int) -> Tuple[int, int]:
        """
        Recommend chunk size and overlap based on text length.
        
        For biomedical text:
        - Small docs (< 5K words): chunk_size=300, overlap=50
        - Medium docs (5K-50K): chunk_size=500, overlap=100
        - Large docs (> 50K): chunk_size=800, overlap=200
        """
        if text_length < 5000:
            return 300, 50
        elif text_length < 50000:
            return 500, 100
        else:
            return 800, 200


def create_optimized_chunks(
    document: BiomedicalDocument,
    chunk_size: int = 500,
    overlap: int = 100,
    deduplicate: bool = True
) -> List[BiomedicalDocument]:
    """
    Create optimized chunks with deduplication.
    
    Args:
        document: Input document
        chunk_size: Chunk size in words
        overlap: Overlap in words
        deduplicate: Whether to remove duplicates
    
    Returns:
        List of deduplicated chunks
    """
    from rag.chunking import chunk_document
    
    chunks = chunk_document(document, chunk_size, overlap)
    chunks = ChunkOptimizer.remove_empty_chunks(chunks)
    
    if deduplicate:
        chunks = ChunkOptimizer.remove_duplicate_chunks(chunks)
    
    return chunks


class RetrievalInspector:
    """Inspect and debug retrieval scores."""
    
    @staticmethod
    def inspect_dense_scores(
        scores: List[float],
        threshold: float = 0.1
    ) -> Dict:
        """
        Inspect dense (embedding) retrieval scores.
        
        Returns:
            Analysis of score distribution
        """
        if not scores:
            return {"issue": "No scores provided"}
        
        non_zero = [s for s in scores if s > 0]
        
        report = {
            "total_scores": len(scores),
            "non_zero_scores": len(non_zero),
            "all_zero": len(non_zero) == 0,
            "min_score": min(scores),
            "max_score": max(scores),
            "avg_score": sum(scores) / len(scores),
        }
        
        if len(non_zero) == 0:
            report["issue"] = "⚠️ All scores are zero! Check embeddings."
        elif max(scores) < threshold:
            report["issue"] = f"⚠️ Max score {max(scores):.4f} below threshold {threshold}"
        
        return report
    
    @staticmethod
    def inspect_bm25_scores(
        scores: List[float],
        threshold: float = 1.0
    ) -> Dict:
        """
        Inspect sparse (BM25) retrieval scores.
        
        Returns:
            Analysis of score distribution
        """
        if not scores:
            return {"issue": "No scores provided"}
        
        non_zero = [s for s in scores if s > 0]
        
        report = {
            "total_scores": len(scores),
            "non_zero_scores": len(non_zero),
            "all_zero": len(non_zero) == 0,
            "min_score": min(scores),
            "max_score": max(scores),
            "avg_score": sum(scores) / len(scores),
        }
        
        if len(non_zero) == 0:
            report["issue"] = "⚠️ All BM25 scores are zero! Check query tokenization."
        elif max(scores) < threshold:
            report["issue"] = f"⚠️ Max BM25 score {max(scores):.4f} below threshold {threshold}"
        
        return report
    
    @staticmethod
    def compare_retrieval_methods(
        dense_scores: List[float],
        bm25_scores: List[float],
        top_k: int = 5
    ) -> Dict:
        """
        Compare dense vs sparse retrieval results.
        
        Returns:
            Comparison report
        """
        dense_top = sorted(range(len(dense_scores)), 
                          key=lambda i: dense_scores[i], 
                          reverse=True)[:top_k]
        
        bm25_top = sorted(range(len(bm25_scores)), 
                         key=lambda i: bm25_scores[i], 
                         reverse=True)[:top_k]
        
        overlap = len(set(dense_top) & set(bm25_top))
        
        return {
            "top_k": top_k,
            "dense_indices": dense_top,
            "bm25_indices": bm25_top,
            "overlap": overlap,
            "diversity": "HIGH" if overlap < top_k * 0.5 else "LOW",
            "recommendation": (
                "Hybrid retrieval can benefit from combining both methods."
                if overlap < top_k * 0.5
                else "Methods are too similar. Consider tuning parameters."
            )
        }


def print_retrieval_stats(
    chunks: List[BiomedicalDocument],
    dense_results: List[Tuple],
    bm25_results: List[Tuple]
) -> None:
    """
    Pretty-print retrieval statistics.
    
    Args:
        chunks: Original chunks
        dense_results: Results from dense retrieval (doc, score)
        bm25_results: Results from BM25 (index, score)
    """
    print("\n" + "=" * 70)
    print("  RETRIEVAL STATISTICS")
    print("=" * 70)
    
    print(f"\nChunk Statistics:")
    print(f"  Total chunks: {len(chunks)}")
    
    stats = ChunkOptimizer.analyze_chunk_sizes(chunks)
    print(f"  Avg chunk size: {stats.get('avg_words', 0):.0f} words")
    print(f"  Size range: {stats.get('min_words', 0)}-{stats.get('max_words', 0)} words")
    
    print(f"\nDense Retrieval (FAISS):")
    dense_scores = [score for _, score in dense_results]
    inspector = RetrievalInspector()
    dense_report = inspector.inspect_dense_scores(dense_scores)
    
    for key, value in dense_report.items():
        if key != "issue":
            print(f"  {key}: {value}")
    if "issue" in dense_report:
        print(f"  {dense_report['issue']}")
    
    print(f"\nSparse Retrieval (BM25):")
    bm25_scores = [score for _, score in bm25_results]
    bm25_report = inspector.inspect_bm25_scores(bm25_scores)
    
    for key, value in bm25_report.items():
        if key != "issue":
            print(f"  {key}: {value}")
    if "issue" in bm25_report:
        print(f"  {bm25_report['issue']}")
    
    print("\n" + "=" * 70)
