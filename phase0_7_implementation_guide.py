"""
PHASE 0-7: Complete RAG Implementation Guide for MSc Projects

This guide explains the systematic approach to building a correct, 
explainable RAG system for biomedical QA.

Author's Notes on RAG for Academic Research:
- Correctness > Sophistication
- Explainability > Black boxes
- Grounding > Hallucinations
- Evidence > Confidence scores alone
"""

# ============================================================================
# PHASE 0: SETUP
# ============================================================================

"""
Before running anything:

1. Install dependencies:
   pip install -r requirements.txt

2. Verify environment:
   python -c "import langchain_huggingface; print('✓ OK')"

3. Place test document in project root
   Example: biomedical_doc.pdf or paper.docx
"""

# ============================================================================
# PHASE 1: FIX THE FUNDAMENTALS
# ============================================================================

"""
GOAL: Verify the entire pipeline works correctly before optimization.

Files created:
- phase1_diagnostics.py

Steps:
1. Verify document extraction
2. Verify chunking (check for duplicates)
3. Verify indexing (embedding count = chunk count)
4. Verify retrieval (scores are not all zero)

USAGE:

    python phase1_diagnostics.py your_document.pdf

Expected output:
- First 1000 characters of extracted text
- Last 1000 characters of extracted text
- Number of chunks created
- Whether duplicates were found
- Retrieval results with non-zero scores
- Dense (FAISS) and sparse (BM25) scores compared

Issues to look for:
✗ Text is empty or very short → extraction problem
✗ All chunks are identical → chunking problem
✗ Vector count ≠ Chunk count → indexing problem
✗ All scores are 0 → embeddings or BM25 problem

DO NOT PROCEED TO PHASE 2 UNTIL ALL ISSUES ARE RESOLVED.
"""

# ============================================================================
# PHASE 2: ADD STRICT DOCUMENT MODE
# ============================================================================

"""
GOAL: Implement query modes so users can choose:
- Document Only: Strict, no hallucinations
- Knowledge Only: General knowledge, no retrieval
- Hybrid: Document first, knowledge second

Files created:
- phase2_query_modes.py

Key Classes:
- AnsweringMode (Enum): DOCUMENT, KNOWLEDGE, HYBRID
- QueryRequest: User query with mode selection
- QueryResponse: Answer with evidence and grounding info

How to use in your API:

    from phase2_query_modes import QueryRequest, AnsweringMode, get_system_prompt

    @app.post("/answer")
    async def answer_question(req: QueryRequest):
        query = req.question
        mode = req.mode  # "document", "knowledge", or "hybrid"
        
        # Get the right prompt for the mode
        system_prompt = get_system_prompt(mode)
        
        # For document mode, retrieve chunks first
        if mode in [AnsweringMode.DOCUMENT, AnsweringMode.HYBRID]:
            chunks = retriever.retrieve(query, k=5)
        else:
            chunks = []
        
        # Format chunks for LLM context
        context = format_retrieved_chunks(chunks)
        
        # Build full prompt
        full_prompt = system_prompt.format(context=context, question=query)
        
        # Call LLM
        answer = llm(full_prompt)
        
        return QueryResponse(
            question=query,
            answer=answer,
            mode=mode,
            retrieved_chunks=chunks,
            grounded=True  # Will be validated in Phase 3
        )

Testing:

    Upload: biomedical_paper.pdf
    
    Question: "What is deep learning?"
    Mode: DOCUMENT
    Expected: "The document does not contain..."
    
    Question: "What is photosynthesis?"
    Mode: KNOWLEDGE
    Expected: Full explanation (model knowledge)
    
    Question: "What is deep learning?" (from paper)
    Mode: DOCUMENT
    Expected: Answer from paper

CRITICAL: In DOCUMENT mode, you MUST enforce that answers come from
the document. If the model tries to use outside knowledge, it's failing.
"""

# ============================================================================
# PHASE 3: ADD GROUNDING VALIDATION
# ============================================================================

"""
GOAL: Check whether answers are actually grounded in retrieved chunks.
Show users exactly which evidence supports each answer.

Files created:
- phase3_grounding.py

Key Classes:
- GroundingValidator: Check answer grounding
- format_evidence_display(): Show evidence to users
- generate_grounding_report(): Detailed analysis

How to use:

    from phase3_grounding import GroundingValidator, format_evidence_display

    # After generating answer
    answer = llm(prompt)
    chunks = retriever.retrieve(query, k=5)
    
    # Validate grounding
    is_grounded = GroundingValidator.is_grounded(
        answer=answer,
        chunks=chunks,
        threshold=0.5  # 50% of answer must be in chunks
    )
    
    # Detect hallucinations
    hallucinations = GroundingValidator.detect_hallucinations(answer, chunks)
    
    if hallucinations:
        print("⚠️ Potentially made-up facts:")
        for h in hallucinations:
            print(f"  - {h}")
    
    # Show evidence to user
    evidence_html = format_evidence_display(chunks, max_chunks=5)

Testing:

    Upload: "Diabetic Neuropathy in Pregnancy"
    
    Question: "What causes diabetic neuropathy?"
    
    Expected output:
    - Answer grounded: ✓ YES
    - Overlap score: 85%
    - No hallucinations
    - Evidence shown with scores
    
    Upload: "Python Programming"
    
    Question: "What are the causes of Alzheimer's disease?"
    
    Expected output:
    - Answer grounded: ✗ NO
    - Overlap score: 10%
    - Flag as ungrounded

CRITICAL: Phase 3 is what makes your RAG system trustworthy.
Without grounding validation, users can't verify answers.
"""

# ============================================================================
# PHASE 4: IMPROVE RETRIEVAL
# ============================================================================

"""
GOAL: Optimize retrieval quality by:
- Removing duplicate chunks
- Finding optimal chunk size
- Inspecting dense and sparse scores
- Comparing retrieval methods

Files created:
- phase4_retrieval_improvements.py

Key Classes:
- ChunkOptimizer: Chunk deduplication and analysis
- RetrievalInspector: Score inspection and debugging

How to use:

    from phase4_retrieval_improvements import (
        ChunkOptimizer,
        create_optimized_chunks,
        RetrievalInspector,
        print_retrieval_stats
    )
    
    # Step 1: Analyze document
    text = extract_text("document.pdf")
    text_length = len(text.split())
    
    # Get recommended chunk size
    chunk_size, overlap = ChunkOptimizer.recommend_chunk_size(text_length)
    print(f"Recommended: chunk_size={chunk_size}, overlap={overlap}")
    
    # Step 2: Create optimized chunks
    doc = BiomedicalDocument(content=text, metadata={})
    chunks = create_optimized_chunks(doc, chunk_size, overlap, deduplicate=True)
    
    # Step 3: Inspect retrieval scores
    dense_results = vectorstore.similarity_search_with_scores(query, k=5)
    bm25_results = [(i, score) for i, score in enumerate(bm25_scores)]
    
    inspector = RetrievalInspector()
    dense_report = inspector.inspect_dense_scores([s for _, s in dense_results])
    bm25_report = inspector.inspect_bm25_scores([s for _, s in bm25_results])
    
    print(dense_report)
    print(bm25_report)
    
    # Step 4: Compare methods
    comparison = inspector.compare_retrieval_methods(
        dense_scores=[s for _, s in dense_results],
        bm25_scores=[s for _, s in bm25_results]
    )

Recommended settings for biomedical text:

    Small documents (< 5K words):
    - chunk_size = 300
    - overlap = 50
    
    Medium documents (5K-50K):
    - chunk_size = 500
    - overlap = 100
    
    Large documents (> 50K):
    - chunk_size = 800
    - overlap = 200

Testing:

    Upload: 50K biomedical paper
    
    Expected:
    - Chunks created: ~100
    - No duplicates
    - Dense scores: 0.7-0.9
    - BM25 scores: 2.5-6.0
    - Both methods retrieve relevant chunks

If all scores are 0:
    ✗ Check embeddings model
    ✗ Check BM25 tokenization
    ✗ Check chunk content (not empty?)
"""

# ============================================================================
# PHASE 5: IMPROVE TRANSPARENCY
# ============================================================================

"""
GOAL: Show users exactly what the system did, step by step.

Files created:
- phase5_transparency.py

Key Classes:
- RetrievalPipeline: Visualize the retrieval pipeline
- format_pipeline_visualization(): ASCII art pipeline
- format_evidence_panel(): Show evidence clearly
- create_answer_summary(): Complete answer with context

How to use:

    from phase5_transparency import (
        RetrievalPipeline,
        format_pipeline_visualization,
        format_evidence_panel,
        format_confidence_indicator,
        create_answer_summary
    )
    
    # Build pipeline visualization
    pipeline = RetrievalPipeline(query, mode=AnsweringMode.DOCUMENT)
    pipeline.add_step("Query Tokenization", f"Query: {query}")
    pipeline.add_step("Dense Retrieval", f"Top score: 0.87")
    pipeline.add_step("Sparse Retrieval", f"Top score: 4.2")
    pipeline.add_step("RRF Fusion", f"5 chunks selected")
    pipeline.add_step("LLM Generation", f"2.3s inference time")
    
    print(pipeline.render_text())
    
    # Show evidence
    evidence_html = format_evidence_panel(chunks, mode)
    
    # Show confidence
    confidence_html = format_confidence_indicator(
        grounded=True,
        overlap_score=0.82,
        hallucination_risk="LOW"
    )
    
    # Complete summary
    full_answer = create_answer_summary(
        question=query,
        answer=answer,
        mode=mode,
        chunks=chunks,
        confidence=0.82,
        grounded=True
    )

Example output:

┌─────────────────────────────────────┐
│      Query Processing Pipeline      │
└─────────────────────────────────────┘

📝 Input Query
    └─ "What causes deep learning?"
       (5 tokens)

        ↓

🔍 Retrieval Phase
    ├─ Dense Retrieval (FAISS)
    │  └─ Top score: 0.873
    │
    ├─ Sparse Retrieval (BM25)
    │  └─ Top score: 4.125
    │
    └─ Fusion Method: RRF Applied

        ↓

📚 Retrieved Documents
    └─ 5 chunks selected
       (with relevance scores)

        ↓

🎯 Generation Phase
    └─ LLM generates answer
       using retrieved context

        ↓

✅ Output Answer
    └─ With evidence links
       and confidence scores

CRITICAL: This is what makes your system defensible in an MSc viva.
You can show exactly what the model did, why, and where evidence came from.
"""

# ============================================================================
# PHASE 6: ADD KNOWLEDGE MODE (Optional)
# ============================================================================

"""
GOAL: Sometimes users genuinely want model knowledge without documents.
Provide a way to query the LLM without retrieval.

Implementation:

    @app.post("/answer")
    async def answer_question(req: QueryRequest):
        if req.mode == AnsweringMode.KNOWLEDGE:
            # No retrieval
            prompt = f"Question: {req.question}\n\nAnswer:"
            answer = llm(prompt)
            
            return QueryResponse(
                question=req.question,
                answer=answer,
                mode=AnsweringMode.KNOWLEDGE,
                retrieved_chunks=[],
                grounded=False,  # Not document-grounded
                model_knowledge_used=True
            )
        
        # ... continue with document modes ...

Testing:

    No document uploaded.
    
    Question: "What is photosynthesis?"
    Mode: KNOWLEDGE
    
    Expected: Full explanation using model knowledge
    retrieved_chunks: []
    model_knowledge_used: True
"""

# ============================================================================
# PHASE 7: ADD HYBRID KNOWLEDGE + DOCUMENTS (Optional)
# ============================================================================

"""
GOAL: Advanced mode that combines document evidence with model knowledge,
clearly distinguishing which is which.

Implementation:

    if req.mode == AnsweringMode.HYBRID:
        chunks = retriever.retrieve(req.question, k=5)
        
        prompt = f\"\"\"
        You have access to a document and your knowledge.
        
        First, carefully read the document context.
        If the document fully answers the question, use ONLY the document.
        
        If the document is incomplete, label clearly:
        - "From document: ..."
        - "Additional information: ..."
        
        Document:
        {format_retrieved_chunks(chunks)}
        
        Question: {req.question}
        \"\"\"
        
        answer = llm(prompt)
        
        return QueryResponse(
            question=req.question,
            answer=answer,
            mode=AnsweringMode.HYBRID,
            retrieved_chunks=chunks,
            grounded=True,  # Partially grounded
            model_knowledge_used=True
        )

Testing:

    Upload: "Basic ML Overview"
    
    Question: "How does deep learning differ from traditional ML?"
    Mode: HYBRID
    
    Expected:
    - Document section: "ML methods include..."
    - Additional: "Deep learning adds neural networks..."
    - Both sources clearly labeled

CRITICAL: In hybrid mode, clarity is essential.
Users must know which information is grounded vs. from general knowledge.
"""

# ============================================================================
# FINAL ARCHITECTURE
# ============================================================================

"""
                User Question
                       │
                       ▼
              Select Answer Mode
                       │
      ┌────────────┬─────────────┐
      ▼            ▼             ▼
Knowledge     Document      Hybrid
Only          Only          Mode
      │            │             │
      ▼            ▼             ▼
     LLM      Retriever      Retriever
                  │              │
                  ▼              ▼
             Top Chunks     Top Chunks
                  │              │
                  ▼              ▼
             Strict LLM     LLM + Context
                  │              │
                  ▼              ▼
              Grounding Check
                       │
                       ▼
                  Final Answer
                       │
                       ▼
          Show Evidence + Scores
          + Confidence Indicator
          + Source Attribution
"""

# ============================================================================
# FOR YOUR MSc VIVA
# ============================================================================

"""
When you present this system:

1. SHOW THE ARCHITECTURE
   "This RAG system follows a systematic 7-phase approach..."

2. DEMONSTRATE PHASE 1 DIAGNOSTICS
   "First, we verified document extraction..."
   (Show extraction quality, chunk distribution, no duplicates)

3. SHOW RETRIEVAL QUALITY
   "Here's the retrieval pipeline with FAISS and BM25 scores..."
   (Show non-zero scores, relevance scores, top chunks)

4. DEMONSTRATE DOCUMENT MODE
   Upload document, ask question IN the document
   Ask question NOT in document
   Show it says "The document does not contain..."

5. SHOW GROUNDING VALIDATION
   "This answer is 85% grounded in the document"
   "Overlap score: 0.85"
   "No hallucinations detected"

6. SHOW EVIDENCE PANEL
   Each chunk with relevance score
   Confidence indicator
   Clearly labeled as document evidence

7. EXPLAIN TRANSPARENCY
   "Users know exactly what the system did"
   Show pipeline visualization
   Show reasoning at each step

YOUR EXAMINERS WILL RESPECT:
✓ Systematic approach
✓ Verification at each stage
✓ Grounding validation
✓ Clear evidence
✓ Honest about limitations

THEY WILL BE SKEPTICAL OF:
✗ Pretty UI without correctness
✗ High scores without explanation
✗ Black-box generation
✗ Claims of 99% accuracy
✗ No evidence display
"""

# ============================================================================
# QUICK REFERENCE: FILE STRUCTURE
# ============================================================================

"""
Your project structure after all phases:

Biomedical-RAG-main/
├── phase1_diagnostics.py          ← Start here
├── phase2_query_modes.py           ← Add answering modes
├── phase3_grounding.py             ← Validate grounding
├── phase4_retrieval_improvements.py ← Optimize retrieval
├── phase5_transparency.py          ← Show pipeline
├── phase0_7_implementation_guide.py ← This file
│
├── app/
│   ├── services.py                 ← Update with Phase 2-5
│   ├── routers/
│   │   └── qa_router.py            ← Add endpoints using phases
│   └── streamlit_app.py            ← Show transparency UI
│
├── rag/
│   ├── chunking.py                 ← Use Phase 4 optimization
│   ├── retriever.py
│   ├── document.py
│   └── hybrid_retriever.py
│
└── requirements.txt                ← Add langchain-huggingface
"""

# ============================================================================
# NEXT STEPS
# ============================================================================

"""
1. Run Phase 1 diagnostics on your document
   python phase1_diagnostics.py your_document.pdf

2. Fix any issues found (extraction, chunking, retrieval)

3. Update your API to support query modes (Phase 2)

4. Add grounding validation (Phase 3)

5. Optimize retrieval parameters (Phase 4)

6. Update UI to show transparency (Phase 5)

7. Test all modes (Document, Knowledge, Hybrid)

8. Prepare presentation showing each phase

9. Practice viva answers explaining the approach
"""
