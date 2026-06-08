"""
PHASE 2: Strict Document Mode Implementation

Defines query request with answering modes:
- document: ONLY use retrieved chunks, strictly grounded
- knowledge: LLM knowledge only, no retrieval
- hybrid: Retrieve first, use model knowledge if needed
"""

from enum import Enum
from pydantic import BaseModel
from typing import List, Optional


class AnsweringMode(str, Enum):
    """Different modes for answering questions."""
    DOCUMENT = "document"
    KNOWLEDGE = "knowledge"
    HYBRID = "hybrid"


class QueryRequest(BaseModel):
    """
    Query request with mode selection.
    """
    question: str
    mode: AnsweringMode = AnsweringMode.DOCUMENT
    file_id: Optional[str] = None  # Track which document was uploaded


class QueryResponse(BaseModel):
    """
    Query response with traceability.
    """
    question: str
    answer: str
    mode: AnsweringMode
    retrieved_chunks: List[dict]  # Evidence chunks
    confidence_score: Optional[float] = None
    grounded: bool  # Whether answer is grounded in retrieved chunks
    model_knowledge_used: Optional[bool] = None  # Whether external knowledge was used


# ========================================
# PROMPTS FOR EACH MODE
# ========================================

DOCUMENT_ONLY_PROMPT = """
You must answer ONLY using the provided context from the uploaded document.

CRITICAL RULES:
1. Only use information explicitly stated in the context
2. Do NOT use your general knowledge
3. If the answer is not in the context, you MUST say:
   "The uploaded document does not contain enough information to answer this question."
4. Quote relevant portions of the context
5. Be precise and factual

Context from the uploaded document:
{context}

Question:
{question}

Answer:
"""

KNOWLEDGE_ONLY_PROMPT = """
You are answering using your general knowledge, without any document context.

The user has NOT provided a document for this question.

Answer the question accurately based on your training knowledge:

Question:
{question}

Answer:
"""

HYBRID_PROMPT = """
You have access to a document context AND your general knowledge.

PRIORITY: Use the document first if it contains relevant information.

Instructions:
1. Check if the context answers the question well
2. If yes, use ONLY the context (clearly cite it)
3. If context is incomplete, you may use your knowledge, BUT clearly label it:
   - "From the document: ..."
   - "Additional information from general knowledge: ..."
4. Be precise and transparent about the source

Document context:
{context}

Question:
{question}

Answer:
"""


def get_system_prompt(mode: AnsweringMode) -> str:
    """Get the appropriate system prompt for the mode."""
    if mode == AnsweringMode.DOCUMENT:
        return DOCUMENT_ONLY_PROMPT
    elif mode == AnsweringMode.KNOWLEDGE:
        return KNOWLEDGE_ONLY_PROMPT
    elif mode == AnsweringMode.HYBRID:
        return HYBRID_PROMPT
    else:
        return DOCUMENT_ONLY_PROMPT


def format_retrieved_chunks(chunks: List[dict], max_chunks: int = 5) -> str:
    """
    Format retrieved chunks for prompt context.
    
    Args:
        chunks: List of retrieved chunk dicts with 'content' and 'score'
        max_chunks: Maximum chunks to include
    
    Returns:
        Formatted string for prompt context
    """
    if not chunks:
        return "[No relevant chunks found in the document]"
    
    formatted = []
    for i, chunk in enumerate(chunks[:max_chunks], 1):
        score = chunk.get('score', 0)
        content = chunk.get('content', '')
        formatted.append(f"[Chunk {i}] (Relevance: {score:.2f})\n{content}")
    
    return "\n\n---\n\n".join(formatted)
