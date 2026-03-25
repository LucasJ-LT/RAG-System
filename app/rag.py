from azure.ai.openai import OpenAIClient
from azure.core.credentials import AzureKeyCredential

from .config import settings
from .retrieve import retrieve_context

SYSTEM_PROMPT = "You are an enterprise IT support assistant.\nUse ONLY the retrieved knowledge base content.\nIf the answer is not present, say exactly:\n'Not found in IT knowledge base.'"


def _openai_client() -> OpenAIClient:
    return OpenAIClient(endpoint=settings.azure_openai_endpoint, credential=AzureKeyCredential(settings.azure_openai_key))


def generate_rag_answer(question: str):
    context, sources = retrieve_context(question, top_k=settings.vector_top_k)

    if not context:
        return "Not found in IT knowledge base.", []

    prompt = (
        "Retrieved KB excerpts:\n"
        f"{context}\n\n"
        f"User question: {question}\n"
        "Answer using ONLY the above context."
    )

    client = _openai_client()
    response = client.chat.completions.create(
        deployment_id=settings.azure_openai_deployment,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=512,
    )

    answer = response.choices[0].message.content.strip()
    if not answer:
        answer = "Not found in IT knowledge base."

    # Strict behavior: if the LLM goes out of scope, fallback
    if "not found in it knowledge base" in answer.lower():
        answer = "Not found in IT knowledge base."

    return answer, sources
