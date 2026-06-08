# Implementation Migration Guide

## From Generic Document Chat → Proposal-Aligned Medical RAG

This guide shows how to migrate from the current implementation to the proposal-aligned system.

---

## Phase 1: Update Query Endpoints

### Current Endpoint
```python
# Current routers/qa_router.py
@app.post("/answer")
async def answer(question: str):
    # Generic - works on any uploaded file
    chunks = retriever.retrieve(question)
    answer = llm(chunks + question)
    return {"answer": answer, "sources": 5}
```

### Updated Endpoint
```python
# NEW routers/qa_router.py
from rag.query_models import MedicalQARequest, MedicalQAResponse
from rag.strict_grounding import (
    get_grounding_prompt,
    create_grounding_response,
    should_refuse_answer,
    get_refusal_message
)
from rag.per_claim_citations import AnswerCiter
from rag.corpus_management import CorpusManager

@app.post("/answer", response_model=MedicalQAResponse)
async def answer_medical_question(request: MedicalQARequest):
    """
    Answer medical questions strictly grounded in medical corpora.
    
    Implements proposal exactly:
    - Only medical corpora (PubMed, WHO, CDC, MedQuAD)
    - Strict grounding (no model knowledge)
    - Per-claim citations
    - Confidence indicators
    """
    
    question = request.question
    
    # 1. SAFETY CHECK: Refuse dangerous requests
    if should_refuse_answer(question):
        return MedicalQAResponse(
            question=question,
            answer=get_refusal_message(question),
            confidence="none",
            grounded=False,
            claims_with_citations=[],
            citation_rate=0.0,
            retrieval_details={...},
            is_refusal=True,
            refusal_reason=get_refusal_message(question)
        )
    
    # 2. RETRIEVE: From medical corpora only
    retrieval_result = retriever.retrieve(question, k=5)
    chunks = retrieval_result["chunks"]
    retrieval_details = retrieval_result["details"]
    
    # 3. CHECK: Is retrieval sufficient?
    if not chunks or retrieval_result["avg_score"] < 0.5:
        return OutOfDomainResponse(question)
    
    # 4. GROUND: Use strict grounding prompt
    grounding_prompt = get_grounding_prompt(
        question,
        format_chunks(chunks),
        grounding_level
    )
    answer = llm(grounding_prompt)
    
    # 5. CITE: Add per-claim citations
    cited_answer = AnswerCiter.cite_answer(answer, chunks)
    
    # 6. RESPOND: Return complete, transparent response
    return MedicalQAResponse(
        question=question,
        answer=answer,
        confidence=grounding_level.value,
        grounded=True,
        claims_with_citations=cited_answer["claims_with_citations"],
        citation_rate=cited_answer["citation_rate"],
        retrieval_details=retrieval_details,
        is_refusal=False
    )
```

---

## Phase 2: Update Knowledge Base Display

### Current UI
```html
<!-- Current Streamlit -->
<div class="knowledge-base">
  <h3>Knowledge Base</h3>
  <p>Upload biomedical documents</p>
  <p>Documents indexed: 1</p>
</div>
```

### Updated UI
```python
# NEW app/streamlit_app.py
import streamlit as st
from rag.corpus_management import CorpusManager

st.set_page_config(page_title="Biomedical Medical QA")

# HEADER
st.title("Biomedical Evidence Retrieval System")
st.markdown("""
A medical question-answering system grounded in authoritative sources:
- **PubMed** - MEDLINE abstracts
- **WHO** - Clinical guidelines
- **CDC** - Disease guidelines  
- **MedQuAD** - Medical QA dataset

*Research prototype - answers grounded in retrieved evidence only.*
""")

# CORPUS STATUS
with st.expander("📚 Corpus Status", expanded=True):
    corpus_mgr = CorpusManager()
    stats = corpus_mgr.get_all_corpus_stats()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Indexed Chunks", f"{stats['total_chunks']:,}")
        st.metric("Retrieval Method", stats['retrieval_method'])
    
    with col2:
        st.metric("Embedding Model", stats['embedding_model'])
        st.metric("Last Updated", stats['last_updated'])
    
    st.write("**Corpus Breakdown:**")
    for corpus in stats['corpora']:
        st.write(f"• **{corpus['name']}** - {corpus['chunks']:,} chunks")
        st.write(f"  {corpus['description']}")

# QUERY INTERFACE
st.header("Ask a Medical Question")

question = st.text_input(
    "Medical question:",
    placeholder="e.g., What are symptoms of type 2 diabetes?"
)

if question:
    with st.spinner("Searching medical corpora..."):
        response = requests.post(
            "http://localhost:8000/answer",
            json={"question": question, "max_sources": 5}
        ).json()
    
    # ANSWER
    st.markdown("## Answer")
    st.write(response["answer"])
    
    # CONFIDENCE
    confidence_icon = {
        "high": "🟢",
        "medium": "🟡",
        "low": "🔴",
        "none": "⚫"
    }
    st.markdown(
        f"**Confidence:** {confidence_icon[response['confidence']]} "
        f"{response['confidence'].upper()}"
    )
    
    # PER-CLAIM CITATIONS
    st.markdown("## Evidence")
    for item in response['claims_with_citations']:
        with st.expander(f"📌 {item['claim'][:60]}..."):
            if item['citation']:
                cite = item['citation']
                st.write(f"**Source:** {cite['source_type']} {cite['source_id']}")
                st.write(f"**Relevance:** {cite['relevance_score']:.0%}")
                st.write(f"> {cite['excerpt']}")
            else:
                st.write("*No direct source found for this claim*")
    
    # RETRIEVAL TRANSPARENCY
    with st.expander("🔍 Retrieval Pipeline Details"):
        det = response['retrieval_details']
        st.write(f"**Query tokens:** {det['query_tokens']}")
        st.write(f"**Dense retrieval:** {det['dense_retrieved']} candidates")
        st.write(f"**Sparse retrieval:** {det['sparse_retrieved']} candidates")
        st.write(f"**After RRF fusion:** {det['after_fusion']} candidates")
        st.write(f"**Final selection:** {det['final_selected']} chunks")
        st.write(f"**Average score:** {det['avg_score']:.2%}")
```

---

## Phase 3: Update Retrieval to Show Pipeline

### Current Retrieval
```python
# Current rag/hybrid_retriever.py
def retrieve(query, k=5):
    dense = faiss_search(query, k=10)
    sparse = bm25_search(query, k=10)
    fused = rrf_fusion(dense, sparse)
    return fused[:k]
```

### Updated Retrieval
```python
# NEW rag/hybrid_retriever.py
def retrieve(query, k=5):
    """
    Retrieve from medical corpora with detailed pipeline info.
    
    Returns:
        {
            "chunks": [...],
            "details": RetrievalDetails object
        }
    """
    
    # Dense retrieval
    dense_results = faiss_search(query, k=10)
    dense_retrieved = len(dense_results)
    
    # Sparse retrieval
    sparse_results = bm25_search(query, k=10)
    sparse_retrieved = len(sparse_results)
    
    # Fusion
    fused = rrf_fusion(dense_results, sparse_results)
    after_fusion = len(fused)
    
    # Reranking
    reranked = cross_encoder_rerank(fused, query, k=k)
    final_selected = len(reranked)
    
    # Scores
    scores = [chunk['score'] for chunk in reranked]
    avg_score = sum(scores) / len(scores) if scores else 0.0
    top_score = max(scores) if scores else 0.0
    
    # Details
    retrieval_details = RetrievalDetails(
        query_tokens=len(query.split()),
        dense_retrieved=dense_retrieved,
        sparse_retrieved=sparse_retrieved,
        after_fusion=after_fusion,
        final_selected=final_selected,
        avg_score=avg_score,
        top_score=top_score
    )
    
    return {
        "chunks": reranked,
        "details": retrieval_details
    }
```

---

## Phase 4: Enforce Corpus Restriction

### Current Ingestion
```python
# Current app/services.py
def add_document(file):
    text = extract_text(file)
    chunks = chunk_text(text)
    index.add(chunks)  # Accept ANYTHING
```

### Updated Ingestion
```python
# NEW rag/corpus_management.py
def ingest_corpus(corpus_name: str, data: bytes):
    """Only allow medical corpora per proposal."""
    
    # Check corpus is approved
    allowed = [c.value for c in CorpusType]
    if corpus_name not in allowed:
        raise ValueError(f"Only {allowed} allowed")
    
    # Ingest with corpus-specific config
    config = MEDICAL_CORPORA[corpus_name]
    chunks = chunk_text(data, config.chunk_size, config.overlap)
    
    # Add metadata
    for chunk in chunks:
        chunk.metadata['source_type'] = corpus_name
        chunk.metadata['source_id'] = extract_source_id(chunk)
    
    index.add(chunks)

# DEVELOPMENT MODE - NOT FOR EVALUATION
def ingest_test_document(file_path: str):
    """Testing only - clearly marked."""
    if os.environ.get("MODE") != "DEVELOPMENT":
        raise ValueError("Testing disabled in production")
    # ... test ingestion ...
```

---

## Phase 5: Update Config

### Create config file
```python
# NEW rag/config.py
import os

# PRODUCTION
ALLOWED_CORPORA = ["PubMed", "WHO", "CDC", "MedQuAD"]
STRICT_GROUNDING = True
MODE = os.environ.get("MODE", "PRODUCTION")

# Retrieval
RETRIEVAL_K = 5
DENSE_K = 10
SPARSE_K = 10
MIN_RETRIEVAL_SCORE = 0.5

# Embeddings
EMBEDDING_MODEL = "BioLORD"
EMBEDDING_DIMENSION = 768

# Safety
REFUSE_DIAGNOSIS = True
REFUSE_TREATMENT = True
```

---

## Migration Checklist

- [ ] **Core Models**
  - [ ] Update `rag/query_models.py` (DONE)
  - [ ] Create `rag/corpus_management.py` (DONE)
  - [ ] Create `rag/strict_grounding.py` (DONE)
  - [ ] Create `rag/per_claim_citations.py` (DONE)

- [ ] **API Endpoints**
  - [ ] Update `/answer` endpoint (show retrieval details)
  - [ ] Update `/status` endpoint (show corpus stats)
  - [ ] Add `/corpus/stats` endpoint (corpus breakdown)
  - [ ] Add safety checks (refusal logic)

- [ ] **Retrieval Pipeline**
  - [ ] Update to return pipeline details
  - [ ] Add RetrievalDetails object
  - [ ] Show dense/sparse/fusion/rerank steps

- [ ] **UI**
  - [ ] Remove "Upload Documents" button
  - [ ] Add "Corpus Status" section
  - [ ] Show per-claim citations
  - [ ] Show retrieval pipeline
  - [ ] Add confidence indicator
  - [ ] Add safety disclaimers

- [ ] **Corpus Management**
  - [ ] Load medical corpora (sample)
  - [ ] Enforce corpus restriction
  - [ ] Add source ID extraction
  - [ ] Add corpus statistics

- [ ] **Testing**
  - [ ] Test strict grounding (model doesn't use knowledge)
  - [ ] Test per-claim citations
  - [ ] Test refusal on diagnosis requests
  - [ ] Test confidence scoring
  - [ ] Test with medical questions only

- [ ] **Documentation**
  - [ ] Update API docs
  - [ ] Document corpus sources
  - [ ] Explain confidence levels
  - [ ] Explain per-claim citations

---

## Timeline

- **Week 1:** Update core models + API endpoints
- **Week 2:** Implement corpus management + retrieval pipeline
- **Week 3:** Update UI to show transparency
- **Week 4:** Add safety checks + testing
- **Week 5:** Evaluation on medical benchmarks
- **Week 6:** Final documentation + viva prep

---

## Success Criteria

✓ No arbitrary file uploads in evaluation
✓ Only medical questions answered
✓ Per-claim citations shown
✓ Corpus statistics displayed
✓ Retrieval pipeline visible
✓ Confidence levels accurate
✓ Strict grounding enforced
✓ Safety checks working

When all are true: System matches proposal.
