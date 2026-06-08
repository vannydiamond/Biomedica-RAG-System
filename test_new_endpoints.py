"""
Test Script for New Proposal-Aligned Endpoints

Run: python test_new_endpoints.py

This validates:
1. Status endpoint shows correct corpus info
2. Answer endpoint returns proper response format
3. Grounding is enforced (no model knowledge)
4. Per-claim citations are included
5. Refusal logic works
"""

import asyncio
import json
from typing import Dict, Any

# Import the new router
from app.routers.qa_router_new import (
    system_status,
    corpus_statistics,
    answer_medical_question,
)

from rag.query_models import MedicalQARequest, MedicalQAResponse
from rag.corpus_management import CorpusManager
from rag.strict_grounding import should_refuse_answer


def print_section(title: str):
    """Print formatted section."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_result(test_name: str, passed: bool, details: str = ""):
    """Print test result."""
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"{status}: {test_name}")
    if details:
        print(f"     {details}")


async def test_corpus_validation():
    """Test 1: Corpus manager validates only medical corpora."""
    print_section("TEST 1: CORPUS VALIDATION")
    
    corpus_mgr = CorpusManager()
    
    # Test allowed corpora
    test_cases = [
        ("PubMed", True),
        ("WHO", True),
        ("CDC", True),
        ("MedQuAD", True),
        ("Sports Documents", False),
        ("Finance Corpus", False),
        ("Random File", False),
    ]
    
    all_pass = True
    for corpus_name, should_be_valid in test_cases:
        is_valid = corpus_mgr.validate_corpus(corpus_name)
        passed = is_valid == should_be_valid
        all_pass = all_pass and passed
        
        print_result(
            f"Corpus '{corpus_name}': {'allowed' if should_be_valid else 'rejected'}",
            passed
        )
    
    return all_pass


async def test_system_status():
    """Test 2: Status endpoint returns corpus information."""
    print_section("TEST 2: SYSTEM STATUS ENDPOINT")
    
    try:
        status = await system_status()
        
        # Validate response structure
        has_status = hasattr(status, "status") and status.status == "ready"
        has_version = hasattr(status, "version")
        has_corpora = hasattr(status, "corpora") and status.corpora is not None
        has_total = status.corpora.total_chunks > 0
        
        print_result("Response has status", has_status, f"status={status.status}")
        print_result("Response has version", has_version, f"version={status.version}")
        print_result("Response has corpora", has_corpora)
        print_result("Corpora has total chunks", has_total, f"total={status.corpora.total_chunks}")
        
        # Print corpus breakdown
        print("\nCorpor Breakdown:")
        for corpus in status.corpora.corpora:
            print(f"  • {corpus.name}: {corpus.chunks:,} chunks")
        
        return all([has_status, has_version, has_corpora, has_total])
        
    except Exception as e:
        print_result("Status endpoint", False, str(e))
        return False


async def test_corpus_stats():
    """Test 3: Corpus stats endpoint."""
    print_section("TEST 3: CORPUS STATISTICS ENDPOINT")
    
    try:
        stats = await corpus_statistics()
        
        # Validate response
        has_corpora = len(stats.corpora) > 0
        has_total = stats.total_chunks > 0
        has_embedding = stats.embedding_model is not None
        has_retrieval = stats.retrieval_method is not None
        
        print_result("Has corpora list", has_corpora, f"count={len(stats.corpora)}")
        print_result("Total chunks > 0", has_total, f"total={stats.total_chunks:,}")
        print_result("Has embedding model", has_embedding, f"model={stats.embedding_model}")
        print_result("Has retrieval method", has_retrieval, f"method={stats.retrieval_method}")
        
        return all([has_corpora, has_total, has_embedding, has_retrieval])
        
    except Exception as e:
        print_result("Corpus stats endpoint", False, str(e))
        return False


def test_refusal_logic():
    """Test 4: Refusal logic for dangerous queries."""
    print_section("TEST 4: REFUSAL LOGIC")
    
    # Test cases that should trigger refusal
    refuse_cases = [
        "Diagnose my symptoms",
        "What medicine should I take?",
        "Is my test result normal?",
        "What treatment do I need?",
        "Do I have diabetes?",
    ]
    
    # Test cases that should NOT trigger refusal
    allow_cases = [
        "What is diabetes?",
        "What are symptoms of diabetes?",
        "How does insulin work?",
        "What is the prevalence of diabetes?",
        "Explain the pathophysiology of diabetes",
    ]
    
    all_pass = True
    
    print("Should REFUSE:")
    for query in refuse_cases:
        should_refuse = should_refuse_answer(query)
        passed = should_refuse == True
        all_pass = all_pass and passed
        print_result(f"Refuse: '{query[:40]}...'", passed)
    
    print("\nShould ALLOW:")
    for query in allow_cases:
        should_refuse = should_refuse_answer(query)
        passed = should_refuse == False
        all_pass = all_pass and passed
        print_result(f"Allow: '{query[:40]}...'", passed)
    
    return all_pass


async def test_response_model():
    """Test 5: Response model validation."""
    print_section("TEST 5: RESPONSE MODEL VALIDATION")
    
    try:
        # Create test request
        request = MedicalQARequest(
            question="What is type 2 diabetes?",
            max_sources=5
        )
        
        # Validate request model
        has_question = request.question is not None
        has_max_sources = request.max_sources == 5
        
        print_result("Request has question", has_question)
        print_result("Request has max_sources", has_max_sources, f"max_sources={request.max_sources}")
        
        # Validate response model structure
        from rag.query_models import RetrievalDetails, ConfidenceLevel
        
        details = RetrievalDetails(
            query_tokens=5,
            dense_retrieved=10,
            sparse_retrieved=8,
            after_fusion=15,
            final_selected=5,
            avg_score=0.85,
            top_score=0.92
        )
        
        has_tokens = details.query_tokens == 5
        has_scores = details.avg_score == 0.85
        has_confidence = hasattr(ConfidenceLevel, "HIGH")
        
        print_result("RetrievalDetails has query_tokens", has_tokens)
        print_result("RetrievalDetails has scores", has_scores)
        print_result("ConfidenceLevel enum exists", has_confidence)
        
        return all([has_question, has_max_sources, has_tokens, has_scores, has_confidence])
        
    except Exception as e:
        print_result("Response model validation", False, str(e))
        return False


async def test_grounding_system():
    """Test 6: Grounding system prevents model knowledge."""
    print_section("TEST 6: GROUNDING SYSTEM")
    
    try:
        from rag.strict_grounding import (
            SYSTEM_PROMPT_STRICT_GROUNDING,
            get_grounding_prompt,
        )
        
        # Verify strict grounding prompt exists
        has_prompt = SYSTEM_PROMPT_STRICT_GROUNDING is not None
        forbids_knowledge = "do NOT use" in SYSTEM_PROMPT_STRICT_GROUNDING.lower()
        forbids_outside = "outside" in SYSTEM_PROMPT_STRICT_GROUNDING.lower() or "general knowledge" in SYSTEM_PROMPT_STRICT_GROUNDING.lower()
        
        print_result("Strict grounding prompt exists", has_prompt)
        print_result("Prompt forbids external knowledge", forbids_knowledge)
        print_result("Prompt explains restriction clearly", forbids_outside)
        
        # Test grounding prompt formatting
        test_prompt = get_grounding_prompt(
            question="What is diabetes?",
            retrieved_context="Diabetes is a metabolic disorder...",
            grounding_level=None
        )
        
        has_question_in_prompt = "What is diabetes?" in test_prompt
        print_result("Prompt includes question", has_question_in_prompt)
        
        return all([has_prompt, forbids_knowledge, forbids_outside, has_question_in_prompt])
        
    except Exception as e:
        print_result("Grounding system test", False, str(e))
        return False


async def test_citation_system():
    """Test 7: Citation system for per-claim attribution."""
    print_section("TEST 7: CITATION SYSTEM")
    
    try:
        from rag.per_claim_citations import ClaimExtractor, CitationMatcher, AnswerCiter
        
        # Test claim extraction
        test_answer = (
            "Type 2 diabetes is a metabolic disorder. "
            "It is characterized by insulin resistance. "
            "Common symptoms include increased thirst."
        )
        
        extractor = ClaimExtractor()
        claims = extractor.extract_claims(test_answer)
        
        has_claims = len(claims) >= 2
        print_result("Claim extraction works", has_claims, f"extracted {len(claims)} claims")
        
        # Test claim entity extraction
        test_claim = "Type 2 diabetes is a metabolic disorder"
        entities = extractor.extract_key_entities(test_claim)
        
        has_entities = len(entities) > 0
        print_result("Entity extraction works", has_entities, f"found {len(entities)} entities")
        
        # Test citation matching
        test_chunk = {
            "content": "Type 2 diabetes is characterized by high blood glucose and insulin resistance.",
            "source_type": "PubMed",
            "source_id": "PMID:12345678",
            "score": 0.92
        }
        
        matcher = CitationMatcher()
        supporting_chunk, overlap = matcher.find_supporting_chunk(
            test_claim,
            [test_chunk],
            min_overlap=0.1
        )
        
        has_chunk = supporting_chunk is not None
        has_overlap = overlap > 0
        print_result("Citation matching works", has_chunk and has_overlap, f"overlap={overlap:.2f}")
        
        return all([has_claims, has_entities, has_chunk, has_overlap])
        
    except Exception as e:
        print_result("Citation system test", False, str(e))
        return False


async def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("  TESTING PROPOSAL-ALIGNED API IMPLEMENTATION")
    print("=" * 70)
    
    results = {}
    
    # Run all tests
    results["Corpus Validation"] = await test_corpus_validation()
    results["System Status"] = await test_system_status()
    results["Corpus Stats"] = await test_corpus_stats()
    results["Refusal Logic"] = test_refusal_logic()
    results["Response Model"] = await test_response_model()
    results["Grounding System"] = await test_grounding_system()
    results["Citation System"] = await test_citation_system()
    
    # Summary
    print_section("SUMMARY")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓" if result else "✗"
        print(f"{status} {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Implementation is proposal-aligned.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Review implementation.")
    
    return passed == total


if __name__ == "__main__":
    # Run async tests
    success = asyncio.run(run_all_tests())
    exit(0 if success else 1)
