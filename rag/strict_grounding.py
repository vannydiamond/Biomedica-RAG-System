"""
Strict Grounding System

Implements the core principle from the proposal:
"Your ONLY role is to answer using the provided medical context below.
Do not use any knowledge outside the provided context."

NO model knowledge mixed in.
NO hallucinations.
ONLY retrieved evidence.
"""

from typing import List, Dict, Optional, Tuple
from enum import Enum


class GroundingLevel(str, Enum):
    """Grounding confidence levels based on retrieval quality."""
    NONE = "none"  # No retrieval
    LOW = "low"  # avg_score < 0.6
    MEDIUM = "medium"  # 0.6 <= avg_score < 0.8
    HIGH = "high"  # avg_score >= 0.8


# ============================================================================
# STRICT GROUNDING PROMPTS
# ============================================================================

SYSTEM_PROMPT_STRICT_GROUNDING = """
You are a medical information assistant.

*** CRITICAL INSTRUCTIONS - DO NOT VIOLATE ***

You MUST answer ONLY using the provided context below.

Do NOT use your own knowledge.
Do NOT use external medical knowledge.
Do NOT infer information not present in the context.
Do NOT supplement the answer with general knowledge.

If the answer is not contained in the provided context, you MUST respond:
"I don't have that information in my current knowledge base. Please consult a qualified healthcare professional."

Rules:
1. Answer ONLY from the provided context
2. Do NOT use knowledge outside the provided context
3. Do NOT make claims not explicitly stated in the context
4. Quote or closely paraphrase the source material
5. Be precise and factual
6. If unsure or missing information, say so

The medical context provided comes from authoritative sources:
PubMed, WHO, CDC, and MedQuAD.

Your ONLY role is to synthesize this evidence accurately.
"""


SYSTEM_PROMPT_OUT_OF_DOMAIN = """
You are a medical question-answering system. 

The question appears to be outside the scope of medical information retrieval,
or no relevant medical evidence was found in our knowledge base.

Response format:
- If medical: "I don't have that information in my current knowledge base. 
             Please consult a qualified healthcare professional."
- If non-medical: "I'm designed to answer medical questions. 
                 This question is outside my scope."
"""


SYSTEM_PROMPT_REFUSAL = """
You are a medical question-answering system with strict safety guidelines.

Some questions cannot be answered responsibly by an AI system:

Examples:
- "Diagnose my symptoms" → Requires a real doctor
- "What medication should I take?" → Requires medical consultation
- "Is my test result normal?" → Requires professional interpretation

Appropriate response:
"I cannot provide medical advice or diagnoses. 
Please consult a qualified healthcare professional."
"""


# ============================================================================
# GROUNDING RESPONSE STRUCTURE
# ============================================================================

class GroundingResponse:
    """Strictly grounded answer response."""
    
    def __init__(
        self,
        question: str,
        answer: str,
        confidence: GroundingLevel,
        num_sources: int,
        avg_score: float,
        sources: List[Dict]
    ):
        self.question = question
        self.answer = answer
        self.confidence = confidence
        self.num_sources = num_sources
        self.avg_score = avg_score
        self.sources = sources
        self.is_grounded = confidence != GroundingLevel.NONE
    
    def to_dict(self) -> Dict:
        """Convert to API response format."""
        return {
            "question": self.question,
            "answer": self.answer,
            "confidence": self.confidence.value,
            "grounded": self.is_grounded,
            "num_sources": self.num_sources,
            "avg_score": round(self.avg_score, 4),
            "sources": self.sources
        }


# ============================================================================
# GROUNDING LOGIC
# ============================================================================

def determine_grounding_level(avg_retrieval_score: float) -> GroundingLevel:
    """
    Map retrieval scores to confidence levels.
    
    Based on proposal's confidence requirements.
    """
    if avg_retrieval_score < 0.1:
        return GroundingLevel.NONE
    elif avg_retrieval_score < 0.6:
        return GroundingLevel.LOW
    elif avg_retrieval_score < 0.8:
        return GroundingLevel.MEDIUM
    else:
        return GroundingLevel.HIGH


def should_attempt_answer(retrieval_score: float, min_threshold: float = 0.5) -> bool:
    """
    Decide if retrieval quality is sufficient to attempt an answer.
    
    If score is below threshold, system should decline to answer.
    """
    return retrieval_score >= min_threshold


def get_grounding_prompt(
    question: str,
    retrieved_context: str,
    grounding_level: GroundingLevel
) -> str:
    """
    Get the appropriate prompt for the grounding level.
    
    Args:
        question: User's medical question
        retrieved_context: Retrieved medical evidence
        grounding_level: Confidence level
    
    Returns:
        System prompt to send to LLM
    """
    
    if grounding_level == GroundingLevel.NONE:
        return SYSTEM_PROMPT_OUT_OF_DOMAIN + f"\n\nQuestion: {question}"
    
    # Standard strict grounding prompt
    prompt = f"""{SYSTEM_PROMPT_STRICT_GROUNDING}

Medical Evidence (from PubMed, WHO, CDC, MedQuAD):

{retrieved_context}

---

Question: {question}

Answer (strictly grounded in the evidence above):
"""
    
    return prompt


def create_grounding_response(
    question: str,
    answer: str,
    retrieved_chunks: List[Dict],
    is_refusal: bool = False,
    refusal_reason: str = ""
) -> GroundingResponse:
    """
    Create a grounded answer response.
    
    Args:
        question: User's question
        answer: Generated answer
        retrieved_chunks: Retrieved source chunks
        is_refusal: Whether to refuse answering
        refusal_reason: Reason for refusal if applicable
    
    Returns:
        GroundingResponse object
    """
    
    if is_refusal:
        return GroundingResponse(
            question=question,
            answer=refusal_reason or answer,
            confidence=GroundingLevel.NONE,
            num_sources=0,
            avg_score=0.0,
            sources=[]
        )
    
    if not retrieved_chunks:
        return GroundingResponse(
            question=question,
            answer="I don't have that information in my current knowledge base. "
                   "Please consult a qualified healthcare professional.",
            confidence=GroundingLevel.NONE,
            num_sources=0,
            avg_score=0.0,
            sources=[]
        )
    
    # Calculate scores
    scores = [chunk.get('score', 0) for chunk in retrieved_chunks]
    avg_score = sum(scores) / len(scores) if scores else 0
    confidence = determine_grounding_level(avg_score)
    
    # Format sources
    sources = [
        {
            "source_type": chunk.get('source_type', 'Unknown'),
            "source_id": chunk.get('source_id', 'N/A'),
            "score": round(chunk.get('score', 0), 4),
            "excerpt": chunk.get('content', '')[:200]
        }
        for chunk in retrieved_chunks
    ]
    
    return GroundingResponse(
        question=question,
        answer=answer,
        confidence=confidence,
        num_sources=len(retrieved_chunks),
        avg_score=avg_score,
        sources=sources
    )


# ============================================================================
# SAFETY CHECKS - Per Proposal Requirements
# ============================================================================

def check_for_diagnosis_request(question: str) -> bool:
    """Detect if user is asking for a diagnosis."""
    diagnosis_keywords = [
        "diagnose",
        "diagnosis",
        "what's wrong with me",
        "why do i have",
        "am i suffering from",
        "do i have",
    ]
    
    question_lower = question.lower()
    return any(kw in question_lower for kw in diagnosis_keywords)


def check_for_treatment_request(question: str) -> bool:
    """Detect if user is asking for treatment recommendations."""
    treatment_keywords = [
        "what medicine",
        "what drug",
        "what treatment",
        "should i take",
        "prescribe",
        "medication for me",
        "how to treat my",
    ]
    
    question_lower = question.lower()
    return any(kw in question_lower for kw in treatment_keywords)


def check_for_personal_medical_data_request(question: str) -> bool:
    """Detect if user is asking for interpretation of their personal medical data."""
    personal_data_keywords = [
        "is my test result",
        "are my results",
        "what do my blood results",
        "interpret my test",
        "what's my",
        "what is my diagnosis",
        "what's wrong with me",
        "what do my results mean",
        "my scan shows",
        "my blood test",
    ]
    
    question_lower = question.lower()
    return any(kw in question_lower for kw in personal_data_keywords)


def should_refuse_answer(question: str) -> bool:
    """
    Check if question should be refused per safety guidelines.
    
    Refuses:
    - Diagnosis requests
    - Treatment recommendations
    - Medical advice for individuals
    - Interpretation of personal medical data
    """
    return (
        check_for_diagnosis_request(question)
        or check_for_treatment_request(question)
        or check_for_personal_medical_data_request(question)
    )


def get_refusal_message(question: str) -> str:
    """Get appropriate refusal message."""
    if check_for_diagnosis_request(question):
        return (
            "I cannot provide medical diagnoses. "
            "Please consult a qualified healthcare professional for diagnosis."
        )
    
    if check_for_treatment_request(question):
        return (
            "I cannot recommend treatments or medications for individual patients. "
            "Please consult a qualified healthcare professional."
        )
    
    if check_for_personal_medical_data_request(question):
        return (
            "I cannot interpret personal medical data or test results. "
            "Please share your results with a qualified healthcare professional."
        )
    
    return (
        "I cannot provide personalized medical advice. "
        "Please consult a qualified healthcare professional."
    )
