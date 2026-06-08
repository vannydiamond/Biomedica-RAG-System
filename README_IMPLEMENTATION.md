# MSc Biomedical RAG: Complete Implementation Framework

## Overview

This framework implements a **systematic, explainable RAG system** focused on **correctness** rather than sophistication. Perfect for academic research and viva demonstrations.

## Problem We Solve

Traditional RAG systems often:
- ✗ Hallucinate (make up facts)
- ✗ Don't show evidence
- ✗ Have zero retrieval scores
- ✗ Can't distinguish document vs. model knowledge
- ✗ Are black boxes (unexplainable)

**This framework addresses all of these.**

---

## The 7 Phases

### Phase 0: Setup
- Install dependencies
- Verify environment
- ✓ **DONE** - We installed `langchain-huggingface`

### Phase 1: Fix the Fundamentals ⭐ START HERE
**Goal:** Verify the entire pipeline works before optimization

**What it checks:**
1. Document extraction (is text extracted correctly?)
2. Chunking (are chunks unique or duplicated?)
3. Indexing (do embeddings match chunks?)
4. Retrieval (are scores non-zero?)

**How to run:**
```bash
python quick_start.py document.pdf
# or
python phase1_diagnostics.py document.pdf
```

**Expected output:**
```
✓ Extracted 12,543 characters
✓ Created 25 chunks
✓ Unique chunks: 25 (no duplicates)
✓ FAISS scores: 0.82, 0.79, 0.71...
✓ BM25 scores: 4.5, 3.8, 2.9...
```

**Issues to look for:**
- ✗ Text is empty → Extraction failed
- ✗ All chunks identical → Chunking problem
- ✗ Vector count ≠ Chunk count → Indexing problem
- ✗ All scores 0.000 → Embeddings/BM25 problem

**Don't proceed until Phase 1 passes.**

### Phase 2: Strict Document Mode
**Goal:** Implement answering modes so users choose how to query

**Modes:**
1. **Document Only** - Strict, grounded, no hallucinations
2. **Knowledge Only** - Model knowledge, no retrieval
3. **Hybrid** - Document first, knowledge if needed

**Key feature:**
- In Document mode, if the answer isn't in the document, the system says so
- No hiding, no hallucinations

**Implementation:**
```python
from phase2_query_modes import QueryRequest, AnsweringMode

# User chooses:
query = QueryRequest(
    question="What is deep learning?",
    mode=AnsweringMode.DOCUMENT  # Strict mode
)
```

**Files:**
- `phase2_query_modes.py` - Query modes and prompts

### Phase 3: Grounding Validation
**Goal:** Verify answers are actually grounded in documents

**What it does:**
- Calculates overlap between answer and chunks
- Detects hallucinations (facts not in document)
- Assigns grounding score (0.0-1.0)
- Flags suspicious statements

**Key metric:**
```
Overlap Score: 0.85
Interpretation: 85% of answer is supported by document
```

**Implementation:**
```python
from phase3_grounding import GroundingValidator

validator = GroundingValidator()
is_grounded = validator.is_grounded(
    answer=answer,
    chunks=chunks,
    threshold=0.5  # 50% minimum
)

hallucinations = validator.detect_hallucinations(answer, chunks)
```

**Files:**
- `phase3_grounding.py` - Grounding validation and evidence formatting

### Phase 4: Improve Retrieval
**Goal:** Optimize retrieval quality

**Improvements:**
- Remove duplicate chunks
- Find optimal chunk size for biomedical text
- Inspect FAISS (dense) scores
- Inspect BM25 (sparse) scores
- Compare retrieval methods

**Recommended settings for biomedical text:**

| Document Size | Chunk Size | Overlap |
|---|---|---|
| < 5K words | 300 | 50 |
| 5K-50K | 500 | 100 |
| > 50K | 800 | 200 |

**Implementation:**
```python
from phase4_retrieval_improvements import ChunkOptimizer

chunk_size, overlap = ChunkOptimizer.recommend_chunk_size(text_length)
chunks = create_optimized_chunks(
    doc,
    chunk_size=chunk_size,
    overlap=overlap,
    deduplicate=True
)
```

**Files:**
- `phase4_retrieval_improvements.py` - Chunk optimization and score inspection

### Phase 5: Improved Transparency
**Goal:** Show users exactly what the system did

**Displays:**
- Query processing pipeline (ASCII visualization)
- Retrieved evidence chunks with scores
- Confidence assessment
- Grounding explanation
- Attribution (document vs. model knowledge)

**Example:**
```
┌─────────────────────────────────┐
│  Query Processing Pipeline      │
└─────────────────────────────────┘

📝 Query: "What is deep learning?"
🎯 Mode: DOCUMENT

🔍 Retrieval:
   Dense (FAISS): 0.87
   Sparse (BM25): 4.2
   
📚 Retrieved: 5 chunks

🎯 Generation: LLM + Context

✅ Answer with evidence shown
```

**Files:**
- `phase5_transparency.py` - Pipeline visualization and evidence display

### Phase 6: Knowledge Mode (Optional)
**Goal:** Allow queries without document retrieval

- User asks general questions
- No retrieval needed
- System uses model knowledge only
- Clearly labeled as non-grounded

### Phase 7: Hybrid Mode (Optional)
**Goal:** Advanced: document + knowledge with clear attribution

**Output distinguishes:**
- "From document: ..."
- "Additional knowledge: ..."

---

## Implementation Roadmap

### Week 1: Foundation (Phase 1)
- [ ] Run Phase 1 diagnostics on sample document
- [ ] Fix any extraction/chunking/retrieval issues
- [ ] Verify non-zero retrieval scores
- [ ] Document all findings

### Week 2: Query Modes (Phase 2)
- [ ] Update API to support QueryRequest with modes
- [ ] Implement Document mode (strict prompts)
- [ ] Implement Knowledge mode
- [ ] Test with sample questions

### Week 3: Grounding (Phase 3)
- [ ] Add GroundingValidator to answer generation
- [ ] Calculate overlap scores for all answers
- [ ] Detect hallucinations
- [ ] Format evidence for display

### Week 4: Retrieval Optimization (Phase 4)
- [ ] Analyze chunk sizes in your documents
- [ ] Remove duplicates if any
- [ ] Tune FAISS and BM25 parameters
- [ ] Inspect retrieval scores systematically

### Week 5: Transparency (Phase 5)
- [ ] Update Streamlit UI to show pipeline
- [ ] Display evidence chunks with scores
- [ ] Show confidence indicators
- [ ] Add mode selector to UI

### Week 6: Testing & Documentation
- [ ] Test all modes (Document, Knowledge, Hybrid)
- [ ] Create test cases
- [ ] Document approach for viva
- [ ] Prepare presentation

---

## For Your MSc Viva

### What Examiners Will Want to See
1. **Systematic approach** - "I followed 7 phases..."
2. **Verification at each stage** - "Phase 1 verified extraction..."
3. **Grounding evidence** - "85% overlap with document"
4. **No hallucinations** - "We detect and flag them"
5. **Clear evidence** - "Here's exactly which chunks we used"
6. **Honesty** - "Our system can't answer questions not in the document"

### What NOT to Do
- ✗ Claim 99% accuracy without evidence
- ✗ Show only pretty UI, not correctness
- ✗ Use black-box generation without grounding
- ✗ Mix document evidence with hallucinations
- ✗ Ignore retrieval score issues

### The Perfect Presentation
> "I built a RAG system focused on correctness and explainability. 
> 
> First, I verified document extraction works (Phase 1 diagnostics shown).
> 
> Then I implemented three answering modes: Document-only (strict, grounded), 
> Knowledge-only (model knowledge), and Hybrid (combined).
> 
> Every answer is validated for grounding against retrieved chunks. 
> If an answer contains facts not in the document, we flag it.
> 
> Users see exactly which evidence supports each answer, 
> with relevance scores and a confidence assessment.
> 
> This approach prioritizes accuracy over sophistication—perfect for 
> biomedical information where hallucinations could be dangerous."

---

## Quick Reference: Files Created

```
Created Files:
├── phase1_diagnostics.py              ← Start here
├── phase2_query_modes.py
├── phase3_grounding.py
├── phase4_retrieval_improvements.py
├── phase5_transparency.py
├── phase0_7_implementation_guide.py   ← Detailed walkthrough
├── quick_start.py                      ← Easy entry point
└── README_IMPLEMENTATION.md            ← This file
```

---

## Getting Started Now

### Option 1: Quick Start (Recommended)
```bash
python quick_start.py your_document.pdf
```

### Option 2: Detailed Diagnostics
```bash
python phase1_diagnostics.py your_document.pdf
```

### Option 3: Read the Guide
```bash
# Read the complete implementation guide
cat phase0_7_implementation_guide.py
```

---

## Key Principles

1. **Correctness First** - A system that's 70% accurate but trustworthy beats 90% accurate but confusing
2. **Explainability** - Users must understand why the system answered the way it did
3. **Grounding** - Every answer must be traceable to evidence
4. **Honesty** - Be upfront about limitations
5. **Transparency** - Show the entire pipeline to users

---

## Common Issues & Solutions

### Issue: All retrieval scores are 0.000
**Solution:** 
- Check embeddings model is loaded
- Verify chunks aren't empty
- Check BM25 tokenization
- Run Phase 1 diagnostics

### Issue: Chunks are duplicated
**Solution:**
- Use `ChunkOptimizer.remove_duplicate_chunks()`
- Check text extraction (may have duplicates)
- See Phase 4 for deduplication

### Issue: Answers don't match document
**Solution:**
- Run grounding validator
- Check chunk size (too small = fragmented)
- Verify retrieval actually finds relevant chunks
- See Phase 3 for validation

### Issue: Retrieved chunks aren't relevant
**Solution:**
- Tune chunk size (Phase 4)
- Check document structure
- Verify embeddings model choice
- Compare dense vs. sparse retrieval scores

---

## Need Help?

1. **Phase-specific questions:** See individual phase files
2. **Complete walkthrough:** See `phase0_7_implementation_guide.py`
3. **Quick diagnostics:** Run `quick_start.py`
4. **Implementation examples:** Check each phase file for usage examples

---

## Success Criteria

✓ Phase 1: All retrieval scores non-zero
✓ Phase 2: Can toggle between Document/Knowledge/Hybrid modes
✓ Phase 3: Grounding scores calculated and hallucinations detected
✓ Phase 4: No duplicate chunks, optimal retrieval parameters
✓ Phase 5: User sees complete pipeline and evidence

**When all 5 are true: You have a defensible MSc RAG system.**

---

**Created:** June 2026
**For:** MSc Biomedical RAG Projects
**Philosophy:** Correctness > Sophistication, Explainability > Black Boxes
