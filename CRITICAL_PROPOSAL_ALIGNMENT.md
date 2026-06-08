# CRITICAL: Proposal ↔ Implementation Alignment

**This is the most important document for your project.**

---

## Executive Summary

Your **proposal is excellent**. Your **current implementation has drifted away from it**.

The fix is not to modify the proposal to match implementation. The fix is to **modify implementation to match proposal**.

This document explains:
1. Where the drift occurred
2. Why it matters for your viva
3. How to fix it systematically

---

## What Your Proposal Says

```
SYSTEM DESIGN

Medical Question
        ↓
Medical Corpus (PubMed + WHO + CDC + MedQuAD)
        ↓
Hybrid Retrieval (Dense + Sparse + RRF)
        ↓
Grounded LLM (evidence-based only)
        ↓
Answer + Citations
```

**Key constraints from proposal:**
- ✓ SPECIFIC medical corpora (not any file)
- ✓ STRICT grounding (no model knowledge)
- ✓ CITATION per claim (not "9 sources")
- ✓ TRANSPARENT (show pipeline)

---

## What Current Implementation Does

```
Upload Any File
        ↓
Generic Chunking
        ↓
Retrieval (on uploaded file)
        ↓
LLM (context + model knowledge mixed)
        ↓
Generic "Sources: 9"
```

**The problem:**
- ✗ Any file accepted (not medical corpora)
- ✗ Model knowledge injected (not strict)
- ✗ Generic source list (not per-claim)
- ✗ No transparency (pipeline hidden)

---

## Why This Matters for Your Viva

### Viva Question 1: "Isn't this just a document chatbot?"

**If implementation:** Generic upload → Yes, it is
**Answer fails.** Examiner: "That's not what your proposal says."

**If implementation:** Medical corpus only → No
**Answer succeeds.** Examiner: "Right, it's a medical evidence system."

### Viva Question 2: "How do you prevent hallucinations?"

**If implementation:** Generic LLM
**Answer fails.** Examiner: "Your model can use general knowledge, so hallucinations are likely."

**If implementation:** Strict grounding only
**Answer succeeds.** Examiner: "And how do you enforce that?" You show the strict prompt.

### Viva Question 3: "How transparent is your system?"

**If implementation:** Generic answers
**Answer fails.** Examiner: "Users can't tell where information comes from."

**If implementation:** Per-claim citations + pipeline
**Answer succeeds.** Examiner: "Excellent. Users know exactly what was retrieved."

---

## The 5 Key Differences

| Aspect | Proposal | Current | Status |
|--------|----------|---------|--------|
| **1. Knowledge Source** | Fixed medical corpora | Any uploaded file | ❌ BROKEN |
| **2. Model Knowledge** | Strictly forbidden | Mixed with retrieval | ❌ BROKEN |
| **3. Citations** | Per-claim (e.g., PMID:123) | Generic ("9 sources") | ❌ BROKEN |
| **4. Transparency** | Show pipeline | Hidden | ❌ BROKEN |
| **5. Evaluation** | Medical benchmarks | Any questions | ❌ BROKEN |

---

## The Fix: 5 Concrete Changes

### 1. Fix the Knowledge Base

**Current:**
```
Knowledge Base
Upload biomedical documents
Documents: 1
```

**Fixed:**
```
Knowledge Base (Medical Corpora)

✓ PubMed: 320,000 chunks
✓ WHO: 15,000 chunks
✓ CDC: 8,000 chunks
✓ MedQuAD: 47,000 chunks

Total: 390,000 chunks
Last Updated: June 2026
```

**Files created:**
- `rag/corpus_management.py`

---

### 2. Fix Grounding

**Current:**
```python
answer = llm(retrieved_chunks + question)
# LLM can use general knowledge
```

**Fixed:**
```python
system_prompt = """
You MUST use ONLY the provided context.
Do NOT use general knowledge.
If information is not in context, say:
'I don't have that information.'
"""
answer = llm(system_prompt + context + question)
```

**Files created:**
- `rag/strict_grounding.py`

---

### 3. Fix Citations

**Current:**
```
Answer: ...
Sources: 9
```

**Fixed:**
```
Answer: Type 2 diabetes is characterized by insulin resistance.
[PubMed PMID:31234567 - Score: 0.92]

Common symptoms include increased thirst.
[MedQuAD ID:Q847 - Score: 0.88]
```

**Files created:**
- `rag/per_claim_citations.py`

---

### 4. Fix Transparency

**Current:**
```
Hybrid Retrieval
Results: 5
```

**Fixed:**
```
Retrieval Pipeline:
1. Dense retrieval: 20 candidates
2. Sparse retrieval: 18 candidates
3. RRF fusion: 40 combined
4. Reranking: 5 final
Average score: 0.89
```

**Files created:**
- `rag/query_models.py` (RetrievalDetails)

---

### 5. Fix Response Structure

**Current:**
```json
{
  "answer": "...",
  "sources": 5
}
```

**Fixed:**
```json
{
  "question": "...",
  "answer": "...",
  "confidence": "high",
  "grounded": true,
  "claims_with_citations": [
    {
      "claim": "...",
      "citation": {"source_type": "PubMed", "source_id": "..."}
    }
  ],
  "retrieval_details": {...},
  "is_refusal": false
}
```

**Files created:**
- `rag/query_models.py` (MedicalQAResponse)

---

## Files Created & What They Do

```
rag/corpus_management.py
└─ Enforce medical corpora only (PubMed, WHO, CDC, MedQuAD)
└─ Show corpus statistics
└─ Prevent arbitrary uploads

rag/strict_grounding.py
└─ Implement grounding-only prompt (no model knowledge)
└─ Add refusal logic (diagnosis, treatment requests)
└─ Calculate confidence levels

rag/per_claim_citations.py
└─ Extract claims from answer
└─ Match each claim to source chunks
└─ Generate per-claim citations

rag/query_models.py
└─ MedicalQAResponse (per-claim citations, confidence)
└─ RetrievalDetails (pipeline transparency)
└─ SourceAttribution (per-source info)

ALIGNMENT_PROPOSAL_TO_IMPLEMENTATION.md
└─ Documents the drift
└─ Shows what needs to change
└─ Explains why it matters

MIGRATION_GUIDE.md
└─ Step-by-step how to fix the implementation
└─ Code examples for each change
└─ Timeline and checklist
```

---

## Implementation Path

### Step 1: Update API Endpoints (This Week)
- Use new query models with confidence + citations
- Add corpus statistics endpoint
- Implement strict grounding prompt
- Add refusal logic

### Step 2: Update UI (Next Week)
- Show corpus breakdown (not file uploads)
- Display per-claim citations
- Show retrieval pipeline
- Add confidence indicator

### Step 3: Corpus Management (Week 3)
- Load medical corpora (start with MedQuAD sample)
- Extract source IDs (PMID, etc.)
- Restrict ingestion to allowed corpora

### Step 4: Evaluation (Week 4-5)
- Test only on medical questions
- Use medical benchmarks (not sports documents)
- Evaluate grounding (not hallucination rate)
- Evaluate citation accuracy

### Step 5: Documentation (Week 6)
- Update README to explain proposal alignment
- Document each component
- Prepare viva presentation

---

## For Your Viva: The Perfect Explanation

> "My proposal describes a **medical evidence retrieval system** grounded in authoritative medical sources. The system is designed to:
> 
> 1. **Answer using specific medical corpora** (PubMed, WHO, CDC, MedQuAD) - not arbitrary documents
> 2. **Generate strictly grounded answers** (retrieved evidence only, no model knowledge)
> 3. **Cite every claim** (per-claim source attribution, not generic sources)
> 4. **Be transparent** (show users the retrieval pipeline)
> 
> To ensure the implementation matches this design, I've made these key changes:
> 
> - Removed arbitrary file uploads from the evaluation system
> - Implemented strict grounding (no model knowledge injection)
> - Added per-claim citations (each fact traced to source)
> - Added retrieval pipeline transparency
> - Restricted to medical corpora (PubMed, WHO, CDC, MedQuAD)
> 
> This makes the system defensible because:
> - ✓ Every answer is traceable to medical evidence
> - ✓ We don't inject general knowledge
> - ✓ We clearly show which information comes from which source
> - ✓ We refuse to provide diagnoses or treatment recommendations
> 
> The research question is: 'Can we build a medical QA system that provides accurate, grounded answers with transparent source attribution?' This implementation allows us to answer that question rigorously."

---

## Decision Point

You have two choices:

### Choice 1: Keep Current Implementation
- Generic document chat
- Mixed model knowledge
- Generic sources
- Black box generation

**Viva Risk:** High
**Question:** "Why is this a medical RAG system instead of document chat?"
**Your Answer:** Weak

### Choice 2: Implement Proposal-Aligned System (Recommended)
- Medical corpora only
- Strict grounding
- Per-claim citations
- Transparent pipeline

**Viva Risk:** Low
**Question:** "Why is this a medical RAG system?"
**Your Answer:** Strong - "Because the proposal requires it, and the implementation enforces it."

---

## Final Thought

The proposal is **excellent**. It's a well-designed research study.

The implementation has drifted into being a generic tool instead of a focused research system.

Correcting this drift takes 2-3 weeks of work, but it:
- ✓ Makes the system stronger
- ✓ Makes the viva defensible
- ✓ Makes the research genuine
- ✓ Makes your findings credible

**Recommendation: Do the migration. It's the right decision for your research.**

---

## Next Steps

1. **Read** ALIGNMENT_PROPOSAL_TO_IMPLEMENTATION.md (understanding)
2. **Review** MIGRATION_GUIDE.md (implementation plan)
3. **Start Week 1** (API endpoints)
4. **Check off items** in migration checklist
5. **Prepare viva explanation** using the text above

Files to focus on:
- `rag/corpus_management.py` - Enforce medical corpora
- `rag/strict_grounding.py` - Implement strict grounding
- `rag/per_claim_citations.py` - Add per-claim citations
- `rag/query_models.py` - Update response structure
