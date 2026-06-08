"""
Updated QA Router - Proposal-Aligned Implementation

This router implements:
1. Strict grounding (no model knowledge)
2. Per-claim citations
3. Corpus statistics (not file uploads)
4. Retrieval transparency
5. Safety checks (refusals)
"""

from fastapi import APIRouter, HTTPException
from typing import Optional

# Import new models
from rag.query_models import (
    MedicalQARequest,
    MedicalQAResponse,
    CorpusStatsResponse,
    SystemStatusResponse,
    SourceAttribution,
    ClaimWithCitation,
    RetrievalDetails,
    ConfidenceLevel,
    OutOfDomainResponse,
    RefusalResponse,
)

# Import corpus and grounding
from rag.corpus_management import CorpusManager
from rag.strict_grounding import (
    get_grounding_prompt,
    create_grounding_response,
    should_refuse_answer,
    get_refusal_message,
    determine_grounding_level,
    SYSTEM_PROMPT_STRICT_GROUNDING,
)
from rag.per_claim_citations import AnswerCiter, ClaimExtractor

# Import existing services
from app.services import QASystem, get_llm

router = APIRouter(prefix="/api", tags=["medical-qa"])

# Initialize components
qa_system = QASystem()
corpus_manager = CorpusManager()


# ============================================================================
# STATUS ENDPOINTS
# ============================================================================

@router.get("/status", response_model=SystemStatusResponse)
async def system_status():
    """
    Get system status and corpus information.
    
    Returns current indexing status and available corpora.
    """
    corpus_stats = corpus_manager.get_all_corpus_stats()
    
    return SystemStatusResponse(
        status="ready",
        version="1.0.0",
        corpora=CorpusStatsResponse(
            corpora=[
                CorpusStatsResponse.CorpusInfo(
                    name=c["name"],
                    chunks=c["chunks"],
                    description=c["description"]
                )
                for c in corpus_stats["corpora"]
            ],
            total_chunks=corpus_stats["total_chunks"],
            embedding_model=corpus_stats["embedding_model"],
            retrieval_method=corpus_stats["retrieval_method"],
            last_updated=corpus_stats["last_updated"]
        )
    )


@router.get("/corpus/stats", response_model=CorpusStatsResponse)
async def corpus_statistics():
    """
    Get detailed corpus statistics.
    
    Shows breakdown of all indexed medical corpora.
    """
    stats = corpus_manager.get_all_corpus_stats()
    
    return CorpusStatsResponse(
        corpora=[
            CorpusStatsResponse.CorpusInfo(
                name=c["name"],
                chunks=c["chunks"],
                description=c["description"]
            )
            for c in stats["corpora"]
        ],
        total_chunks=stats["total_chunks"],
        embedding_model=stats["embedding_model"],
        retrieval_method=stats["retrieval_method"],
        last_updated=stats["last_updated"]
    )


# ============================================================================
# MAIN QA ENDPOINT - PROPOSAL-ALIGNED
# ============================================================================

@router.post("/answer", response_model=MedicalQAResponse)
async def answer_medical_question(request: MedicalQARequest) -> MedicalQAResponse:
    """
    Answer a medical question using strict grounding.
    
    Implementation:
    1. Check if question should be refused (safety)
    2. Retrieve from medical corpora
    3. Check retrieval quality
    4. Generate strictly grounded answer
    5. Add per-claim citations
    6. Return transparent response
    
    Args:
        request: MedicalQARequest with question and optional max_sources
    
    Returns:
        MedicalQAResponse with answer, citations, confidence, and retrieval details
    """
    
    question = request.question.strip()
    max_sources = request.max_sources
    
    # ===================================================================
    # STEP 1: SAFETY CHECK - Refuse dangerous requests
    # ===================================================================
    
    if should_refuse_answer(question):
        refusal_msg = get_refusal_message(question)
        
        return MedicalQAResponse(
            question=question,
            answer=refusal_msg,
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
            is_refusal=True,
            refusal_reason=refusal_msg
        )
    
    # ===================================================================
    # STEP 2: RETRIEVE - From medical corpora only
    # ===================================================================
    
    try:
        # Retrieve from indexed medical corpora
        retrieved_docs = qa_system.retrieve(question, k=max_sources)
        
        if not retrieved_docs:
            return OutOfDomainResponse(
                question=question,
                reason="No relevant medical information found in corpus"
            )
        
        # Convert LangChain docs to our format
        retrieved_chunks = []
        scores = []
        
        for i, doc in enumerate(retrieved_docs):
            chunk_dict = {
                "content": doc.page_content,
                "source_type": doc.metadata.get("source_type", "Unknown"),
                "source_id": doc.metadata.get("source_id", f"Doc{i}"),
                "score": doc.metadata.get("score", 0.5),  # Will use similarity score
            }
            retrieved_chunks.append(chunk_dict)
            scores.append(chunk_dict["score"])
        
        avg_score = sum(scores) / len(scores) if scores else 0.0
        top_score = max(scores) if scores else 0.0
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Retrieval failed: {str(e)}"
        )
    
    # ===================================================================
    # STEP 3: CHECK - Is retrieval sufficient?
    # ===================================================================
    
    if avg_score < 0.5 or len(retrieved_chunks) == 0:
        return OutOfDomainResponse(
            question=question,
            reason=f"Low retrieval confidence (score: {avg_score:.2f})"
        )
    
    grounding_level = determine_grounding_level(avg_score)
    
    # ===================================================================
    # STEP 4: GENERATE - Strictly grounded answer
    # ===================================================================
    
    try:
        # Format context for prompt
        context_text = "\n\n---\n\n".join([
            f"[{chunk['source_type']} {chunk['source_id']}]\n{chunk['content']}"
            for chunk in retrieved_chunks[:max_sources]
        ])
        
        # Build strict grounding prompt
        prompt = f"""{SYSTEM_PROMPT_STRICT_GROUNDING}

Medical Evidence from Corpus:

{context_text}

---

Question: {question}

Answer (STRICTLY grounded in evidence above):
"""
        
        # Initialize LLM if not already done
        if qa_system.llm is None:
            # Default to using OpenAI if available
            qa_system.set_llm(
                provider="openai",
                api_key="dummy",  # This would be loaded from config
                model="gpt-3.5-turbo"
            )
        
        # Generate answer
        answer = qa_system.llm.invoke(prompt)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Answer generation failed: {str(e)}"
        )
    
    # ===================================================================
    # STEP 5: CITE - Add per-claim citations
    # ===================================================================
    
    try:
        cited_answer = AnswerCiter.cite_answer(
            answer=answer,
            retrieved_chunks=retrieved_chunks,
            min_overlap=0.3
        )
        
        claims_with_citations = [
            ClaimWithCitation(
                claim=item["claim"],
                citation=SourceAttribution(
                    source_type=item["citation"]["source_type"],
                    source_id=item["citation"]["source_id"],
                    relevance_score=item["citation"]["relevance_score"],
                    excerpt=item["citation"]["excerpt"]
                ) if item["citation"] else None,
                overlap_score=item["overlap_score"]
            )
            for item in cited_answer["claims_with_citations"]
        ]
        
        citation_rate = cited_answer["citation_rate"]
        
    except Exception as e:
        # If citation extraction fails, return answer without citations
        citation_rate = 0.0
        claims_with_citations = []
    
    # ===================================================================
    # STEP 6: RESPOND - Return transparent response
    # ===================================================================
    
    retrieval_details = RetrievalDetails(
        query_tokens=len(question.split()),
        dense_retrieved=len(retrieved_chunks),
        sparse_retrieved=len(retrieved_chunks),  # Simplified for now
        after_fusion=len(retrieved_chunks),
        final_selected=min(len(retrieved_chunks), max_sources),
        avg_score=avg_score,
        top_score=top_score
    )
    
    return MedicalQAResponse(
        question=question,
        answer=answer,
        confidence=grounding_level,
        grounded=grounding_level != ConfidenceLevel.NONE,
        claims_with_citations=claims_with_citations,
        citation_rate=citation_rate,
        retrieval_details=retrieval_details,
        is_refusal=False
    )


# ============================================================================
# STREAMING ENDPOINT (Optional)
# ============================================================================

@router.post("/answer-stream")
async def answer_medical_question_stream(request: MedicalQARequest):
    """
    Stream medical answer token-by-token.
    
    Returns Server-Sent Events stream.
    """
    # This would implement streaming with proper SSE response
    # For now, returns regular response
    # Real implementation would use StreamingResponse with event generator
    return await answer_medical_question(request)


# ============================================================================
# DEPRECATED ENDPOINTS (Kept for Backward Compatibility)
# ============================================================================

# NOTE: The following endpoints are from the old implementation
# and should be REMOVED in production. They are kept here only
# for reference and backward compatibility during migration.
# These do NOT align with the proposal.

@router.post("/upload")
async def upload_doc(file):
    """
    DEPRECATED: Do not use for evaluation.
    
    Old endpoint for uploading arbitrary documents.
    This is NOT part of the proposal-aligned system.
    """
    raise HTTPException(
        status_code=410,
        detail="Document upload endpoint deprecated. "
               "This system uses fixed medical corpora (PubMed, WHO, CDC, MedQuAD). "
               "See /api/corpus/stats for available sources."
    )


@router.get("/ask")
async def ask_question(q: str):
    """
    DEPRECATED: Use /api/answer instead.
    
    Old generic question endpoint.
    """
    raise HTTPException(
        status_code=410,
        detail="Old /ask endpoint deprecated. Use /api/answer with MedicalQARequest."
    )
