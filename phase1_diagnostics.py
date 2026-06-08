"""
PHASE 1: Fix the Fundamentals - Diagnostic Tool

This script verifies:
1. Document extraction
2. Chunking
3. Indexing
4. Retrieval

Run with: python phase1_diagnostics.py <file_path>
"""

import sys
import os
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from app.utils import extract_text
from rag.chunking import chunk_document
from rag.document import BiomedicalDocument
from rag.bm25_retriever import BM25Retriever
from app.services import get_embeddings
from langchain_community.vectorstores import FAISS


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def step1_verify_extraction(file_path: str) -> str:
    """Step 1: Verify document extraction"""
    print_section("STEP 1: VERIFY DOCUMENT EXTRACTION")
    
    try:
        text = extract_text(file_path)
        
        print(f"✓ Extracted {len(text)} characters")
        print(f"✓ Word count: {len(text.split())}")
        print(f"\nFirst 1000 characters:")
        print("-" * 60)
        print(text[:1000])
        print("-" * 60)
        print(f"\nLast 1000 characters:")
        print("-" * 60)
        print(text[-1000:])
        print("-" * 60)
        
        # Check for common issues
        if len(text) < 100:
            print("⚠ WARNING: Text is very short. Check extraction.")
        if text.count(" ") < 10:
            print("⚠ WARNING: Very few spaces. Text might be corrupted.")
        
        return text
    except Exception as e:
        print(f"✗ ERROR: {e}")
        sys.exit(1)


def step2_verify_chunking(text: str) -> list:
    """Step 2: Verify chunking"""
    print_section("STEP 2: VERIFY CHUNKING")
    
    doc = BiomedicalDocument(
        content=text,
        metadata={"source": "uploaded_file"}
    )
    
    chunks = chunk_document(doc, chunk_size=500, overlap=100)
    
    print(f"✓ Created {len(chunks)} chunks")
    print(f"  Chunk size: 500 words")
    print(f"  Overlap: 100 words")
    
    # Check for duplicates
    unique_chunks = set(chunk.content for chunk in chunks)
    print(f"\n✓ Unique chunks: {len(unique_chunks)}")
    
    if len(unique_chunks) < len(chunks):
        duplicate_count = len(chunks) - len(unique_chunks)
        print(f"⚠ WARNING: {duplicate_count} duplicate chunks detected!")
    
    # Show first 5 chunks
    print(f"\nFirst 5 chunks preview:")
    print("-" * 60)
    for i, chunk in enumerate(chunks[:5]):
        print(f"\nCHUNK {i} ({len(chunk.content.split())} words)")
        print(chunk.content[:300])
        if len(chunk.content) > 300:
            print("...")
    
    # Check for identical chunks
    print("\n\nChecking for chunk similarity:")
    for i in range(min(3, len(chunks)-1)):
        similarity = len(set(chunks[i].content.split()) & 
                        set(chunks[i+1].content.split())) / max(
                        len(chunks[i].content.split()),
                        len(chunks[i+1].content.split()))
        print(f"Chunk {i} vs Chunk {i+1}: {similarity:.2%} overlap")
        if similarity > 0.9:
            print("  ⚠ WARNING: Very high overlap!")
    
    return chunks


def step3_verify_indexing(chunks: list) -> dict:
    """Step 3: Verify indexing"""
    print_section("STEP 3: VERIFY INDEXING")
    
    print(f"Chunks: {len(chunks)}")
    
    # Create embeddings
    try:
        embeddings = get_embeddings()
        print(f"✓ Embeddings model loaded: all-MiniLM-L6-v2")
        
        # Create FAISS index
        chunk_texts = [chunk.content for chunk in chunks]
        vectorstore = FAISS.from_texts(chunk_texts, embeddings)
        
        print(f"✓ FAISS index created")
        
        # Check index stats
        index_stats = vectorstore.index.ntotal if hasattr(vectorstore.index, 'ntotal') else len(chunks)
        print(f"✓ Vectors in index: {index_stats}")
        
        if index_stats != len(chunks):
            print(f"⚠ WARNING: Vector count ({index_stats}) != Chunk count ({len(chunks)})")
        
        return {
            "vectorstore": vectorstore,
            "embeddings": embeddings,
            "chunks": chunks
        }
    except Exception as e:
        print(f"✗ ERROR: {e}")
        sys.exit(1)


def step4_verify_retrieval(index_data: dict, test_queries: list = None) -> None:
    """Step 4: Verify retrieval"""
    print_section("STEP 4: VERIFY RETRIEVAL")
    
    if test_queries is None:
        test_queries = [
            "What is deep learning?",
            "How does neural networks work?",
            "What is machine learning?"
        ]
    
    vectorstore = index_data["vectorstore"]
    chunks = index_data["chunks"]
    
    # Initialize BM25
    try:
        from rank_bm25 import BM25Okapi
        tokenized_chunks = [chunk.content.split() for chunk in chunks]
        bm25 = BM25Okapi(tokenized_chunks)
        print("✓ BM25 index created")
    except Exception as e:
        print(f"⚠ WARNING: BM25 initialization failed: {e}")
        bm25 = None
    
    for query in test_queries:
        print(f"\n\nQuery: '{query}'")
        print("-" * 60)
        
        # Dense retrieval (FAISS)
        try:
            dense_results = vectorstore.similarity_search_with_scores(query, k=3)
            print(f"\nDense Retrieval (FAISS):")
            for i, (doc, score) in enumerate(dense_results):
                print(f"  [{i}] Score: {score:.4f}")
                print(f"      {doc.page_content[:150]}...")
            
            if all(score == 0 for _, score in dense_results):
                print("  ⚠ WARNING: All scores are 0!")
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
        
        # Sparse retrieval (BM25)
        if bm25:
            try:
                query_tokens = query.split()
                bm25_scores = bm25.get_scores(query_tokens)
                top_indices = sorted(range(len(bm25_scores)), 
                                   key=lambda i: bm25_scores[i], 
                                   reverse=True)[:3]
                
                print(f"\nSparse Retrieval (BM25):")
                for rank, idx in enumerate(top_indices):
                    score = bm25_scores[idx]
                    print(f"  [{rank}] Score: {score:.4f}")
                    print(f"      {chunks[idx].content[:150]}...")
                
                if all(bm25_scores[idx] == 0 for idx in top_indices):
                    print("  ⚠ WARNING: All BM25 scores are 0!")
            except Exception as e:
                print(f"  ✗ ERROR: {e}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python phase1_diagnostics.py <file_path>")
        print("\nExample:")
        print("  python phase1_diagnostics.py sample.docx")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    if not os.path.exists(file_path):
        print(f"✗ File not found: {file_path}")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("  PHASE 1: FIX THE FUNDAMENTALS")
    print("  Diagnostic Tool for RAG Pipeline")
    print("=" * 60)
    
    # Step 1: Extract
    text = step1_verify_extraction(file_path)
    
    # Step 2: Chunk
    chunks = step2_verify_chunking(text)
    
    # Step 3: Index
    index_data = step3_verify_indexing(chunks)
    
    # Step 4: Retrieve
    step4_verify_retrieval(index_data)
    
    print_section("DIAGNOSTICS COMPLETE")
    print("\nNext steps:")
    print("1. Fix any issues found above")
    print("2. Re-run this script to verify fixes")
    print("3. Review Phase 2: Add Strict Document Mode")


if __name__ == "__main__":
    main()
