from typing import List, Tuple

from azure.ai.openai import OpenAIClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

from .config import settings


def _get_search_client() -> SearchClient:
    return SearchClient(endpoint=settings.azure_search_endpoint, index_name=settings.azure_search_index_name,
                        credential=AzureKeyCredential(settings.azure_search_api_key))


def _get_openai_client() -> OpenAIClient:
    from .ingest import _get_openai_client as _create_ai_client
    return _create_ai_client()


def vector_search(query: str, top_k: int = None) -> List[dict]:
    if top_k is None:
        top_k = settings.vector_top_k

    openai_client = _get_openai_client()
    search_client = _get_search_client()

    embed_resp = openai_client.embeddings.create(
        model=settings.azure_openai_embedding_model,
        input=query,
    )

    query_vector = embed_resp.data[0].embedding

    results = search_client.search(
        search_text="",
        vector={"value": query_vector, "fields": "embedding", "k": top_k},
        select=["id", "content", "source"],
    )

    docs = []
    for r in results:
        docs.append({
            "id": r["id"],
            "content": r["content"],
            "source": r["source"],
            "score": r["@search.score"],
        })

    return docs


def retrieve_context(question: str, top_k: int = None) -> Tuple[str, List[str]]:
    hits = vector_search(question, top_k)
    if not hits:
        return "", []

    context_parts = []
    sources = []
    for hit in hits:
        sources.append(hit["id"])
        context_parts.append(f"[{hit['id']} - {hit['source']}]:\n{hit['content']}")

    return "\n\n".join(context_parts), sources
