"""
llm_client.py
-------------
Handles communication with OpenRouter's API using the OpenAI-compatible SDK.
Builds the system+user prompt and returns the raw answer text.
"""

from typing import List

from openai import OpenAI

from src.config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, LLM_MODEL


# System prompt — instructs the model to stay grounded in the provided context
_SYSTEM_PROMPT = """You are a precise document question-answering assistant.

Your job is to answer the user's question using ONLY the context excerpts provided below.

Rules:
1. Base your answer exclusively on the provided context. Do NOT use any outside knowledge.
2. If the context does not contain enough information to answer the question, respond with exactly:
   "The information is not available in the supplied documents."
3. Be concise and factual. Quote or closely paraphrase the relevant parts of the context.
4. Do NOT fabricate details, names, numbers, or dates that are not explicitly stated in the context.
"""


def _build_user_prompt(question: str, context_chunks: List[dict]) -> str:
    """
    Assembles the user turn of the prompt: numbered context excerpts + the question.

    Args:
        question:       The user's question string.
        context_chunks: List of dicts with keys: text, doc_name, page_number, score.

    Returns:
        Formatted prompt string.
    """
    context_block = ""
    for i, chunk in enumerate(context_chunks, start=1):
        context_block += (
            f"[{i}] Source: {chunk['doc_name']} | Page {chunk['page_number']}\n"
            f"{chunk['text']}\n\n"
        )

    return (
        f"Context:\n"
        f"{'─' * 60}\n"
        f"{context_block}"
        f"{'─' * 60}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )


class LLMClient:
    """Thin wrapper around OpenRouter (OpenAI-compatible API)."""

    def __init__(self) -> None:
        if not OPENROUTER_API_KEY:
            raise EnvironmentError(
                "OPENROUTER_API_KEY is not set. Please add it to your .env file."
            )

        self._client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=OPENROUTER_API_KEY,
        )

    def generate_answer(self, question: str, context_chunks: List[dict]) -> str:
        """
        Sends the question + context to the LLM and returns the answer text.

        Args:
            question:       The user's question.
            context_chunks: Retrieved chunks from Qdrant (with metadata).

        Returns:
            The model's response as a plain string.
        """
        user_prompt = _build_user_prompt(question, context_chunks)

        try:
            response = self._client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,   # low temperature = more factual, less creative
                max_tokens=1024,
            )
            return response.choices[0].message.content.strip()

        except Exception as exc:
            raise RuntimeError(
                f"OpenRouter API call failed: {exc}\n"
                f"Model used: {LLM_MODEL}\n"
                "Check your OPENROUTER_API_KEY and network connection."
            ) from exc
