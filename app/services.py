# app/services.py

import os
from typing import List
from dotenv import load_dotenv

# -------------------------
# LOAD ENVIRONMENT VARIABLES
# -------------------------
load_dotenv()

HF_TOKEN = os.getenv("HF_API_TOKEN")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
COHERE_KEY = os.getenv("COHERE_API_KEY")

# -------------------------
# EMBEDDINGS & VECTOR STORE
# -------------------------
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

INDEX_PATH = "vectorstore_index"


def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# -------------------------
# BASE LLM INTERFACE
# -------------------------
class BaseLLM:
    def invoke(self, prompt: str) -> str:
        raise NotImplementedError

    def stream(self, prompt: str):
        """Yield text chunks"""
        raise NotImplementedError


# -------------------------
# HUGGING FACE LLM
# -------------------------
from huggingface_hub import InferenceClient


class HFLLM(BaseLLM):
    def __init__(self, token: str, model_name: str, temperature: float = 0.7, max_tokens: int = 512):
        if not token:
            raise ValueError("Hugging Face API token required")

        self.client = InferenceClient(
            model=model_name,
            token=token
        )
        self.temperature = temperature
        self.max_tokens = max_tokens

    def invoke(self, prompt: str) -> str:
        return self.client.text_generation(
            prompt,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature
        ).strip()

    # ⚠️ HF Inference API does NOT support true streaming
    def stream(self, prompt: str):
        # Return the full response as a single chunk since HF doesn't support streaming
        text = self.invoke(prompt)
        yield text


# -------------------------
# COHERE LLM (STREAMING ✅)
# -------------------------
import cohere


class CohereLLM(BaseLLM):
    def __init__(self, api_key: str, model: str, temperature: float = 0.7, max_tokens: int = 512):
        self.client = cohere.Client(api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Validate
        self.client.chat(model=self.model, message="hi", max_tokens=1)

    def invoke(self, prompt: str) -> str:
        response = self.client.chat(
            model=self.model,
            message=prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )

        # Handle different response formats
        if hasattr(response, "text") and response.text:
            return response.text.strip()
        elif hasattr(response, "message") and response.message:
            if hasattr(response.message, "content") and response.message.content:
                return response.message.content[0].text.strip() if isinstance(response.message.content, list) else str(response.message.content).strip()
            return str(response.message).strip()
        else:
            return str(response).strip()

    def stream(self, prompt: str):
        stream = self.client.chat_stream(
            model=self.model,
            message=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        for event in stream:
            if event.event_type == "text-generation":
                yield event.text


# -------------------------
# OPENAI LLM (STREAMING ✅)
# -------------------------
from openai import OpenAI


class OpenAILLM(BaseLLM):
    def __init__(self, api_key: str, model: str, temperature: float = 0.7, max_tokens: int = 512):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def invoke(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return response.choices[0].message.content

    def stream(self, prompt: str):
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True
        )

        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


# -------------------------
# LLM FACTORY
# -------------------------
def get_llm(provider: str, api_key: str, model: str, temperature: float = 0.7, max_tokens: int = 512) -> BaseLLM:
    provider = provider.lower()

    if provider == "hf":
        return HFLLM(api_key, model, temperature, max_tokens)

    elif provider == "cohere":
        return CohereLLM(api_key, model, temperature, max_tokens)

    elif provider == "openai":
        return OpenAILLM(api_key, model, temperature, max_tokens)

    else:
        raise ValueError(f"Unsupported provider: {provider}")


# -------------------------
# QA SYSTEM (RAG CORE)
# -------------------------
class QASystem:
    def __init__(self):
        self.vectorstore = self._load_index()
        self.llm: BaseLLM | None = None

    def set_llm(self, provider: str, api_key: str, model: str, temperature: float = 0.7, max_tokens: int = 512):
        self.llm = get_llm(
            provider=provider,
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )

    def build_index(self, texts: List[str]):
        """
        Build FAISS index from multiple documents
        """
        if not texts:
            raise ValueError("No texts provided")

        # Combine all texts and split into chunks
        combined_text = "\n\n".join(texts)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )
        docs = splitter.split_text(combined_text)

        # Create embeddings and FAISS index
        embeddings = get_embeddings()
        self.vectorstore = FAISS.from_texts(docs, embeddings)
        self.vectorstore.save_local(INDEX_PATH)

    def _load_index(self):
        # Check for both FAISS index files
        faiss_file = os.path.join(INDEX_PATH, "index.faiss")
        pkl_file = os.path.join(INDEX_PATH, "index.pkl")

        if os.path.exists(faiss_file) and os.path.exists(pkl_file):
            try:
                return FAISS.load_local(
                    INDEX_PATH,
                    get_embeddings(),
                    allow_dangerous_deserialization=True
                )
            except Exception as e:
                # If loading fails, return None to start fresh
                print(f"Warning: Failed to load existing index: {e}")
                return None
        return None

    def retrieve(self, query: str, k: int = 3):
        if not self.vectorstore:
            return []
        return self.vectorstore.similarity_search(query, k=k)

    def answer(self, query: str) -> str:
        if not self.vectorstore:
            return "No document uploaded."

        if not self.llm:
            return "LLM not initialized."

        docs = self.retrieve(query)

        context = "\n\n".join([d.page_content for d in docs])

        prompt = f"""
Answer ONLY from the context below.

Context:
{context}

Question: {query}

Answer:
"""
        return self.llm.invoke(prompt)