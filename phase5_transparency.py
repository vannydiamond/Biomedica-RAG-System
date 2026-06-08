"""
PHASE 5: Improved Transparency

Show users exactly what happened:
- Query processing
- Retrieval pipeline
- Mode selection
- Evidence scores
"""

from typing import List, Dict, Optional
from enum import Enum
from phase2_query_modes import AnsweringMode


class RetrievalPipeline:
    """
    Represents the complete retrieval pipeline for transparency.
    """
    
    def __init__(self, query: str, mode: AnsweringMode):
        self.query = query
        self.mode = mode
        self.steps = []
    
    def add_step(self, step_name: str, details: str = "", status: str = "✓"):
        """Add a step to the pipeline."""
        self.steps.append({
            "name": step_name,
            "details": details,
            "status": status
        })
    
    def render_text(self) -> str:
        """Render pipeline as ASCII art."""
        lines = []
        lines.append("\n" + "=" * 70)
        lines.append("  RETRIEVAL PIPELINE")
        lines.append("=" * 70)
        
        lines.append(f"\n📝 Query: {self.query}")
        lines.append(f"🎯 Mode: {self.mode.value.upper()}")
        
        lines.append("\n" + "─" * 70)
        lines.append("Pipeline Steps:\n")
        
        for i, step in enumerate(self.steps, 1):
            lines.append(f"{step['status']} Step {i}: {step['name']}")
            if step['details']:
                for detail_line in step['details'].split('\n'):
                    lines.append(f"    {detail_line}")
            lines.append("")
        
        lines.append("=" * 70)
        return "\n".join(lines)
    
    def render_markdown(self) -> str:
        """Render pipeline as Markdown."""
        lines = []
        lines.append("## Retrieval Pipeline\n")
        lines.append(f"**Query:** {self.query}\n")
        lines.append(f"**Mode:** `{self.mode.value}`\n")
        lines.append("---\n")
        
        for i, step in enumerate(self.steps, 1):
            lines.append(f"### Step {i}: {step['name']}")
            lines.append(f"Status: {step['status']}\n")
            if step['details']:
                lines.append(step['details'])
            lines.append("")
        
        return "\n".join(lines)


def show_answer_mode_selection() -> str:
    """
    Display the answer mode selection UI as Markdown.
    """
    return """
# Select Answer Mode

Choose how you want the system to answer your question:

## 🎯 Document Only
- **Uses:** Only the uploaded document
- **Grounding:** Strictly enforced
- **Best for:** Fact-checking what's in the document
- **Example:** "Does the document mention X?"

**If the answer isn't in the document, we'll tell you.**

---

## 🧠 Knowledge Only  
- **Uses:** Model knowledge (no document)
- **Grounding:** Not applicable
- **Best for:** General questions not related to your document
- **Example:** "What is machine learning?"

**No retrieval from your document. Uses my training data.**

---

## 🔀 Hybrid
- **Uses:** Document first, then model knowledge if needed
- **Grounding:** Partial
- **Best for:** Deep exploration with context
- **Example:** "Explain X in the context of Y"

**Sources are clearly labeled as document or general knowledge.**

---

**Current mode:** Document Only *(strict and safe)*
"""


def format_pipeline_visualization(
    query: str,
    retrieved_chunks: int,
    dense_score: Optional[float] = None,
    bm25_score: Optional[float] = None,
    rrf_applied: bool = False
) -> str:
    """
    Format a visual representation of what happened.
    """
    pipeline = f"""
┌─────────────────────────────────────┐
│      Query Processing Pipeline      │
└─────────────────────────────────────┘

📝 Input Query
    └─ "{query}"
       ({len(query.split())} tokens)

        ↓

🔍 Retrieval Phase
    ├─ Dense Retrieval (FAISS)
    │  └─ Top score: {dense_score:.3f if dense_score else "N/A"}
    │
    ├─ Sparse Retrieval (BM25)
    │  └─ Top score: {bm25_score:.3f if bm25_score else "N/A"}
    │
    └─ Fusion Method: {"RRF Applied" if rrf_applied else "Dense Only"}

        ↓

📚 Retrieved Documents
    └─ {retrieved_chunks} chunks selected
       (with relevance scores)

        ↓

🎯 Generation Phase
    └─ LLM generates answer
       using retrieved context

        ↓

✅ Output Answer
    └─ With evidence links
       and confidence scores

"""
    return pipeline


def format_evidence_panel(
    chunks: List[Dict],
    mode: AnsweringMode,
    show_scores: bool = True
) -> str:
    """
    Format the evidence panel for display.
    
    Args:
        chunks: List of retrieved chunks
        mode: Current answering mode
        show_scores: Whether to show relevance scores
    
    Returns:
        Formatted markdown for display
    """
    if not chunks:
        return "### 📄 Evidence\nNo evidence retrieved from the document."
    
    lines = ["### 📄 Evidence from Document\n"]
    
    if mode == AnsweringMode.DOCUMENT:
        lines.append("*(Strictly grounded in the uploaded document)*\n")
    elif mode == AnsweringMode.HYBRID:
        lines.append("*(From the uploaded document - additional knowledge may be included above)*\n")
    
    for i, chunk in enumerate(chunks, 1):
        score = chunk.get('score', 0)
        content = chunk.get('content', '')
        
        if show_scores:
            # Score bar visualization
            bar_length = 20
            filled = int(score * bar_length)
            bar = "█" * filled + "░" * (bar_length - filled)
            lines.append(f"**Chunk {i}** | Relevance: `{score:.1%}` | `[{bar}]`\n")
        else:
            lines.append(f"**Chunk {i}**\n")
        
        lines.append(f"> {content[:500]}")
        if len(content) > 500:
            lines.append("...\n")
        else:
            lines.append("\n")
        lines.append("")
    
    return "\n".join(lines)


def format_confidence_indicator(
    grounded: bool,
    overlap_score: float,
    hallucination_risk: str = "LOW"
) -> str:
    """
    Format a confidence indicator.
    
    Args:
        grounded: Whether answer is grounded
        overlap_score: Overlap with retrieved chunks (0-1)
        hallucination_risk: "LOW", "MEDIUM", "HIGH"
    
    Returns:
        Formatted indicator
    """
    
    # Determine confidence level
    if grounded and overlap_score > 0.7:
        confidence = "🟢 HIGH"
        reason = "Answer is well-grounded in the document"
    elif grounded and overlap_score > 0.4:
        confidence = "🟡 MEDIUM"
        reason = "Answer is partially grounded in the document"
    else:
        confidence = "🔴 LOW"
        reason = "Answer may contain information not in the document"
    
    indicator = f"""
### 📊 Confidence Assessment

**Confidence Level:** {confidence}

**Reason:** {reason}

**Grounding Score:** {overlap_score:.0%}

**Hallucination Risk:** {hallucination_risk}

{
    "✅ You can trust this answer for document-based facts." 
    if hallucination_risk == "LOW" 
    else "⚠️ Verify with the original source for critical decisions."
}
"""
    
    return indicator


def create_answer_summary(
    question: str,
    answer: str,
    mode: AnsweringMode,
    chunks: List[Dict],
    confidence: float,
    grounded: bool
) -> str:
    """
    Create a complete answer summary with all context.
    """
    summary = f"""
# Answer to Your Question

**Question:** {question}

**Mode:** {mode.value.upper()}

---

## Answer

{answer}

---

{format_evidence_panel(chunks, mode)}

{format_confidence_indicator(grounded, confidence)}

---

**Learn more:** Each answer includes the exact chunks we used, their relevance scores, and a confidence assessment.
"""
    
    return summary
