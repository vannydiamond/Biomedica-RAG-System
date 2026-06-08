# Architecture Alignment: Proposal vs Current Implementation

## The Problem: Implementation Drift

### What the Proposal Says

```
Medical Question
        ↓
Medical Corpus
(PubMed + WHO + CDC + MedQuAD)
        ↓
Hybrid Retrieval
(FAISS/BioLORD + BM25 + RRF)
        ↓
Top-K Chunks
        ↓
Grounded LLM
(Strictly evidence-based)
        ↓
Answer + Per-Claim Citations
```

**Core principle:** Retrieve from **specific medical corpora**, ground strictly in retrieved evidence only.

### What Current Implementation Does

```
Upload Any File
        ↓
Generic Chunking
        ↓
Hybrid Retrieval
(on uploaded file)
        ↓
Top-K Chunks
        ↓
LLM (with model knowledge)
        ↓
Answer + Generic "Sources"
```

**Problem:** This is a **general-purpose document chatbot**, not a **medical evidence system**.

---

## Specific Gaps

| Aspect | Proposal | Current | Status |
|--------|----------|---------|--------|
| **Knowledge Source** | Fixed medical corpora | Any uploaded file | ❌ DRIFT |
| **Corpus** | PubMed + WHO + CDC + MedQuAD | User uploads | ❌ DRIFT |
| **Grounding** | Context ONLY, no model knowledge | Context + LLM knowledge mixed | ❌ DRIFT |
| **Citations** | Per-claim sources (PMID, etc.) | Generic "9 sources" | ❌ DRIFT |
| **Evaluation** | Medical QA benchmarks | Any questions | ❌ DRIFT |
| **Transparency** | Show retrieval pipeline | Show generic results | ⚠️ PARTIAL |
| **Corpus Stats** | Show indexed corpora | Show uploaded docs | ❌ DRIFT |

---

## Why This Matters for Your Viva

### Viva Question: "Isn't this just document chat?"

**If you've implemented:** Generic upload system
**Answer:** Weak - Yes, it's essentially document chat with medical questions

**If you've implemented:** Corpus-based medical RAG
**Answer:** Strong - No, it's a retrieval system over specific medical evidence sources. We chose these corpora because they represent authoritative medical information (PubMed, WHO, CDC, MedQuAD).

---

## The Fix: Align Implementation to Proposal

### Step 1: Remove "Upload Documents" from MVP Evaluation

**Current UI:**
```
Knowledge Base
Upload biomedical documents
Documents indexed: 1
```

**Corrected UI:**
```
Knowledge Base (Corpus)

✓ PubMed Abstracts
  Chunks: 320,000
  Last Updated: June 2026

✓ WHO Guidelines
  Chunks: 15,000
  
✓ CDC Guidelines  
  Chunks: 8,000
  
✓ MedQuAD
  Chunks: 47,000

Total Indexed: 390,000 chunks
Embedding Model: BioLORD
Index Type: FAISS + BM25
```

---

### Step 2: Implement Strict Grounding

**Current:**
```python
# Mix of retrieval + model knowledge
context = retrieved_chunks
answer = llm(f"Answer based on: {context}")
# LLM can use outside knowledge
```

**Corrected:**
```python
# STRICTLY retrieval-based
context = retrieved_chunks

if not context or retrieval_score < THRESHOLD:
    return "I don't have enough information..."

# Strict system prompt - no model knowledge
system_prompt = """
You are a medical information assistant.

CRITICAL RULE: You MUST ONLY use the provided medical context.
Do NOT use general knowledge.
Do NOT make medical claims not in the context.

If information is not in the context, say:
'I don't have that information in my current knowledge base.'
"""

answer = llm(system_prompt + context + question)
```

---

### Step 3: Per-Claim Citations

**Current Output:**
```
Answer: Type 2 diabetes is characterized by insulin resistance...
Sources: 9
```

**Corrected Output:**
```
Answer with Sources:

Type 2 diabetes is characterized by insulin resistance.
[Source: PubMed PMID:31234567 - Score: 0.92]

Common symptoms include increased thirst and frequent urination.
[Source: MedQuAD ID:Q847 - Score: 0.88]

Risk factors include obesity and family history.
[Source: CDC Guidelines v2.1 - Score: 0.85]
```

---

### Step 4: Show Retrieval Pipeline

**Current:**
```
Hybrid Retrieval
Results: 5 chunks
```

**Corrected:**
```
Retrieval Pipeline

Query: "What causes type 2 diabetes?"

1. Dense Retrieval (BioLORD + FAISS)
   └─ Found: 20 candidates
   └─ Top score: 0.92
   
2. Sparse Retrieval (BM25)
   └─ Found: 20 candidates
   └─ Top score: 4.5
   
3. RRF Fusion
   └─ Combined: 40 unique candidates
   
4. Cross-Encoder Reranking
   └─ Selected: 5 top chunks
   
5. Final Context
   └─ 2,100 tokens
   └─ Confidence: HIGH (avg score: 0.89)
```

---

### Step 5: Corpus Restriction

**Current - allows any file:**
```python
def ingest(file):
    # Accept any document
    chunks = chunk_document(file)
    index.add(chunks)
```

**Corrected - medical corpora only:**
```python
ALLOWED_SOURCES = {
    "PubMed",
    "WHO",
    "CDC",
    "MedQuAD"
}

def ingest(corpus_name, data):
    if corpus_name not in ALLOWED_SOURCES:
        raise ValueError(f"Only {ALLOWED_SOURCES} allowed")
    
    # Strict ingestion for specific corpus
    chunks = chunk_corpus(corpus_name, data)
    index.add(chunks)

# DEVELOPMENT MODE: Optional test harness
def ingest_test_document(file, mode="TEST"):
    if mode != "TEST":
        raise ValueError("Test mode only")
    # ... for debugging, not evaluation
```

---

### Step 6: Confidence Indicator

**Current:**
```python
# No explicit confidence
return answer
```

**Corrected:**
```python
def generate_answer(question):
    chunks = retrieve(question, k=5)
    
    if not chunks:
        return {
            "answer": "I don't have that information...",
            "confidence": "none",
            "source": "NO_RETRIEVAL"
        }
    
    retrieval_scores = [c.score for c in chunks]
    avg_score = sum(retrieval_scores) / len(retrieval_scores)
    
    # Map to confidence
    if avg_score > 0.8:
        confidence = "high"
    elif avg_score > 0.6:
        confidence = "medium"
    else:
        confidence = "low"
    
    answer = llm(strict_prompt(chunks, question))
    
    return {
        "answer": answer,
        "confidence": confidence,
        "avg_score": avg_score,
        "num_sources": len(chunks),
        "sources": [
            {
                "source_id": c.metadata.get("id"),
                "source_type": c.metadata.get("source"),  # PubMed, WHO, CDC, MedQuAD
                "score": c.score,
                "text": c.content[:200]
            }
            for c in chunks
        ]
    }
```

---

## Reworded UI/Marketing

### Current
```
Biomedical RAG Medical Question Answering System
Upload your documents and ask questions
```

### Corrected
```
Biomedical Evidence Retrieval System

A medical question-answering system grounded in authoritative sources:
- PubMed Abstracts
- WHO Guidelines  
- CDC Guidelines
- MedQuAD

Retrieval: Hybrid Dense + Sparse + RRF
Grounding: Strict (retrieved evidence only)
Citations: Per-claim source attribution

Research Prototype - For Evaluation Only
```

---

## MVP Scope (Following Proposal)

### Phase 1: Corpus Ingestion & Indexing
- [ ] PubMed MEDLINE ingestion (sample)
- [ ] WHO ingestion (sample)
- [ ] CDC ingestion (sample)
- [ ] MedQuAD ingestion
- [ ] Document → Chunks
- [ ] BioLORD embeddings
- [ ] FAISS indexing
- [ ] BM25 indexing

### Phase 2: Retrieval & Generation
- [ ] Dense retrieval (FAISS)
- [ ] Sparse retrieval (BM25)
- [ ] RRF fusion
- [ ] Cross-encoder reranking
- [ ] Strict grounding prompt
- [ ] Per-claim citation extraction
- [ ] Confidence scoring

### Phase 3: Evaluation
- [ ] Retrieval metrics (NDCG, MRR, MAP)
- [ ] Hallucination detection
- [ ] Grounding validation
- [ ] Citation accuracy
- [ ] Medical QA benchmark testing

### Phase 4: Safety & Transparency
- [ ] Refusal logic (out-of-domain questions)
- [ ] Medical disclaimer
- [ ] Explainability dashboard
- [ ] Pipeline visualization
- [ ] Source attribution

---

## Evaluation Protocol (Follows Proposal)

### What TO Evaluate
✓ Medical corpus (PubMed, WHO, CDC, MedQuAD)
✓ Medical questions from benchmarks
✓ Retrieval quality
✓ Grounding accuracy
✓ Citation correctness
✓ Hallucination rates

### What NOT to Evaluate  
✗ Arbitrary document uploads
✗ Non-medical questions
✗ Generic QA performance
✗ Model knowledge without retrieval
✗ Weak citations

---

## Key Changes to Code

### 1. Corpus Configuration
```python
# config.py - NEW
MEDICAL_CORPORA = {
    "PubMed": {
        "path": "data/pubmed/",
        "chunk_size": 500,
        "source_type": "PubMed"
    },
    "WHO": {
        "path": "data/who/",
        "chunk_size": 500,
        "source_type": "WHO"
    },
    "CDC": {
        "path": "data/cdc/",
        "chunk_size": 500,
        "source_type": "CDC"
    },
    "MedQuAD": {
        "path": "data/medquad/",
        "chunk_size": 300,
        "source_type": "MedQuAD"
    }
}
```

### 2. Strict Grounding Prompt
```python
# prompts.py - NEW
STRICT_MEDICAL_PROMPT = """
You are a medical information assistant. Your role is to answer questions 
using ONLY the provided medical evidence below.

CRITICAL RULES:
1. ONLY use information in the provided context
2. Do NOT use general knowledge
3. Do NOT make medical claims not explicitly stated
4. If asked something not in the context, say "I don't have that information"
5. Quote relevant portions of the source material
6. Be precise and factual

Provided Medical Context:
{context}

Question: {question}

Answer (strictly grounded in context above):
"""
```

### 3. Citation Extraction
```python
# citations.py - NEW
def extract_claims_with_citations(
    answer: str,
    source_chunks: List[Dict]
) -> List[Dict]:
    """
    Map answer claims to source chunks.
    Returns: [{"claim": "...", "source_id": "...", "confidence": 0.9}, ...]
    """
    claims = []
    # Implement claim extraction + matching
    return claims
```

### 4. API Response Structure
```python
# routers/qa_router.py - MODIFIED
from pydantic import BaseModel

class SourceAttribution(BaseModel):
    source_id: str  # PubMed PMID, MedQuAD ID, etc
    source_type: str  # "PubMed", "WHO", "CDC", "MedQuAD"
    score: float
    excerpt: str

class AnswerWithCitations(BaseModel):
    question: str
    answer: str
    confidence: str  # "high", "medium", "low", "none"
    sources: List[SourceAttribution]
    retrieval_details: dict  # Show pipeline
```

---

## Migration Strategy

### What Stays (Good Stuff)
- ✓ Hybrid retrieval infrastructure
- ✓ Grounding validation
- ✓ Transparency framework
- ✓ Chunking logic
- ✓ Embedding models

### What Changes (Implementation Drift Fixes)
- ✗ Remove arbitrary file upload from evaluation
- ✗ Add corpus management
- ✗ Enforce strict grounding
- ✗ Add per-claim citations
- ✗ Show retrieval pipeline
- ✗ Add corpus statistics

### Timeline
- **Week 1:** Set up corpus data structures, create sample indexing
- **Week 2:** Implement strict grounding prompt, test on MedQuAD
- **Week 3:** Add per-claim citation extraction
- **Week 4:** Update UI to show corpus + retrieval pipeline
- **Week 5:** Evaluation on medical benchmarks
- **Week 6:** Safety & refusal logic

---

## Defense Statement (For Viva)

> "My proposal describes a medical question-answering system grounded in authoritative medical sources: PubMed, WHO, CDC, and MedQuAD. The system retrieves relevant evidence from these corpora using hybrid retrieval (dense + sparse + RRF), then generates answers strictly grounded in the retrieved context.
> 
> This is fundamentally different from a general-purpose document chatbot. The research question is: 'Can we build a medical QA system that provides accurate, grounded answers with clear source attribution, without hallucinating medical claims?'
> 
> To stay true to this research objective, I've aligned the implementation strictly with the proposal:
> - Fixed medical corpora (not arbitrary uploads)
> - Strict grounding (retrieved evidence only)
> - Per-claim citations (not generic source lists)
> - Retrieval transparency (showing the complete pipeline)
> 
> This makes the system defensible, evaluable, and true to the research design."

---

## This Is The Key Insight

**Implementation shouldn't drift to match convenience. Implementation should match proposal.**

Your original proposal was strong. The current implementation has compromised it by becoming too generic.

Reverting to the proposal makes your research stronger and more defensible.
