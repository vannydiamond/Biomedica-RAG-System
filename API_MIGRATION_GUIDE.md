"""
API MIGRATION: OLD → NEW ENDPOINTS

This guide explains how to switch from the old generic endpoints
to the new proposal-aligned endpoints.
"""

# ============================================================================
# CURRENT SETUP (Generic Document Chat)
# ============================================================================

# In app/main.py currently:
"""
from fastapi import FastAPI
from app.routers import qa_router

app = FastAPI()
app.include_router(qa_router.router)
"""

# This loads the OLD endpoints:
# POST /upload - Upload arbitrary documents
# GET /ask - Ask generic questions


# ============================================================================
# NEW SETUP (Proposal-Aligned Medical RAG)
# ============================================================================

# Change app/main.py to:
"""
from fastapi import FastAPI
from app.routers import qa_router_new  # Import NEW router

app = FastAPI(
    title="Biomedical Medical QA System",
    description="Evidence-based medical question-answering grounded in "
                "PubMed, WHO, CDC, and MedQuAD corpora",
    version="1.0.0"
)

# Include NEW router with medical endpoints
app.include_router(qa_router_new.router)
"""

# This loads the NEW endpoints:
# GET /api/status - System status
# GET /api/corpus/stats - Corpus statistics
# POST /api/answer - Medical Q&A with strict grounding
# POST /api/answer-stream - Streaming medical Q&A (optional)


# ============================================================================
# ENDPOINT COMPARISON
# ============================================================================

"""
OLD ENDPOINTS (Generic, Not Proposal-Aligned):
┌─────────────────────────────────────────────────────────┐
│ POST /upload                                            │
│ Upload any document (.pdf, .docx, .txt)                │
│ Response: {"status": "document uploaded"}              │
│                                                         │
│ GET /ask?q=question                                    │
│ Ask generic question                                   │
│ Response: {"answer": "..."}                            │
└─────────────────────────────────────────────────────────┘

NEW ENDPOINTS (Proposal-Aligned, Grounded):
┌─────────────────────────────────────────────────────────┐
│ GET /api/status                                         │
│ Check system status                                    │
│ Response: {status, version, corpus stats}             │
│                                                         │
│ GET /api/corpus/stats                                  │
│ Get corpus statistics                                  │
│ Response: {corpora[], total_chunks, embedding_model}  │
│                                                         │
│ POST /api/answer                                        │
│ Answer medical question with grounding                │
│ Request: {question, max_sources}                       │
│ Response: {answer, confidence, citations, retrieval}  │
│                                                         │
│ POST /api/answer-stream                                │
│ Stream answer token-by-token                           │
│ Response: Server-Sent Events stream                    │
└─────────────────────────────────────────────────────────┘
"""


# ============================================================================
# MIGRATION STEPS
# ============================================================================

"""
Step 1: Update app/main.py
───────────────────────────

BEFORE:
    from app.routers import qa_router
    app.include_router(qa_router.router)

AFTER:
    from app.routers import qa_router_new
    app.include_router(qa_router_new.router)


Step 2: Update tests to use new endpoints
──────────────────────────────────────────

BEFORE:
    POST /upload
    File: document.pdf
    Response: {"status": "document uploaded"}

    GET /ask?q=What is diabetes?
    Response: {"answer": "..."}


AFTER:
    GET /api/status
    Response: {
        "status": "ready",
        "version": "1.0.0",
        "corpora": {...}
    }

    POST /api/answer
    Request: {"question": "What is diabetes?", "max_sources": 5}
    Response: {
        "question": "What is diabetes?",
        "answer": "Diabetes is a metabolic disorder...",
        "confidence": "high",
        "grounded": true,
        "claims_with_citations": [
            {
                "claim": "Diabetes is a metabolic disorder",
                "citation": {
                    "source_type": "PubMed",
                    "source_id": "PMID:12345678",
                    "relevance_score": 0.92,
                    "excerpt": "..."
                }
            }
        ],
        "citation_rate": 0.95,
        "retrieval_details": {
            "query_tokens": 4,
            "dense_retrieved": 10,
            "sparse_retrieved": 8,
            "after_fusion": 15,
            "final_selected": 5,
            "avg_score": 0.87,
            "top_score": 0.94
        }
    }


Step 3: Remove upload functionality (NOT for evaluation)
────────────────────────────────────────────────────────

The new system uses FIXED medical corpora.
Remove code that accepts arbitrary document uploads.

KEEP for testing/development:
- rag/corpus_management.py → ingest_test_document() (marked DEVELOPMENT ONLY)

REMOVE from production:
- /upload endpoint
- File upload UI
- Document storage logic


Step 4: Update UI (Streamlit)
──────────────────────────────

BEFORE:
    - Upload documents section
    - Generic Q&A interface

AFTER:
    - Corpus status display (PubMed, WHO, CDC, MedQuAD)
    - Medical question interface
    - Per-claim citation display
    - Retrieval pipeline visualization
    - Confidence indicator


Step 5: Test with medical questions only
─────────────────────────────────────────

OLD evaluation:
    ✗ Upload sports document
    ✗ Ask about football
    ✗ Evaluate generic answer

NEW evaluation:
    ✓ No file uploads
    ✓ Medical questions only
    ✓ Evaluate grounding + citations
"""


# ============================================================================
# EXAMPLE CURL REQUESTS
# ============================================================================

"""
Test the new endpoints:

1. Check system status:
   curl http://localhost:8000/api/status

2. Get corpus statistics:
   curl http://localhost:8000/api/corpus/stats

3. Ask a medical question:
   curl -X POST http://localhost:8000/api/answer \\
     -H "Content-Type: application/json" \\
     -d '{
       "question": "What are symptoms of type 2 diabetes?",
       "max_sources": 5
     }'
"""


# ============================================================================
# PARALLEL RUNNING (During Migration)
# ============================================================================

"""
You can run both old and new routes during migration:

from fastapi import FastAPI
from app.routers import qa_router, qa_router_new

app = FastAPI()

# OLD routes (temporary)
app.include_router(qa_router.router, tags=["deprecated"])

# NEW routes (production)
app.include_router(qa_router_new.router)

Then gradually move clients to /api/answer

Once all clients migrated, remove qa_router completely.
"""


# ============================================================================
# KEY CHANGES FOR DEVELOPERS
# ============================================================================

"""
What changed in the new system:

1. KNOWLEDGE SOURCE
   OLD: Any uploaded document
   NEW: Fixed medical corpora (PubMed, WHO, CDC, MedQuAD)

2. MODEL KNOWLEDGE
   OLD: Mixed with retrieval
   NEW: Strictly forbidden (SYSTEM_PROMPT_STRICT_GROUNDING)

3. CITATIONS
   OLD: Generic "Sources: 9"
   NEW: Per-claim (each fact linked to source)

4. TRANSPARENCY
   OLD: Hidden
   NEW: Full retrieval pipeline shown

5. CONFIDENCE
   OLD: Not provided
   NEW: Based on retrieval scores (high/medium/low/none)

6. SAFETY
   OLD: None
   NEW: Refuses diagnosis/treatment requests

7. RESPONSE MODEL
   OLD: {"answer": str}
   NEW: MedicalQAResponse with 10+ fields
"""


# ============================================================================
# IMPLEMENTATION TIMELINE
# ============================================================================

"""
Week 1: Update API Endpoints
├─ Replace main.py router import
├─ Test new endpoints with Postman/curl
├─ Verify responses match MedicalQAResponse model
└─ Keep old routes running in parallel

Week 2: Update UI (Streamlit)
├─ Remove "Upload Documents" button
├─ Add "Corpus Status" section
├─ Display per-claim citations
├─ Show retrieval pipeline
└─ Add confidence indicator

Week 3: Corpus Management
├─ Load medical corpora (sample)
├─ Extract source IDs (PMID, guideline IDs)
├─ Enforce corpus restriction
└─ Test with actual corpora

Week 4: Testing & Validation
├─ Test with medical questions only
├─ Verify grounding works
├─ Verify citations are accurate
├─ Verify refusal logic works
└─ Prepare for evaluation

Week 5: Documentation
├─ Update API docs
├─ Document corpus sources
├─ Prepare viva materials
└─ Remove old code
"""


# ============================================================================
# CONFIGURATION (Optional)
# ============================================================================

"""
New system uses environment variables:

MODE=PRODUCTION  # Strict medical QA only
OR
MODE=DEVELOPMENT  # Allows test document ingestion

Corpus restriction is enforced in corpus_management.py:

ALLOWED_CORPORA = {
    CorpusType.PUBMED,
    CorpusType.WHO,
    CorpusType.CDC,
    CorpusType.MEDQUAD,
}

To add/remove corpora, modify corpus_management.py
"""
