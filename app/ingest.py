import glob
import os
import math
from typing import List, Dict

from azure.ai.openai import OpenAIClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SearchableField,
    SimpleField,
    VectorSearch,
    VectorSearchAlgorithmConfiguration,
    HnswParameters,
)

from .config import settings

SYSTEM_PROMPT = "You are an enterprise IT support assistant.\nUse ONLY the retrieved knowledge base content.\nIf the answer is not present, say exactly:\n'Not found in IT knowledge base.'"


def _get_search_index_client() -> SearchIndexClient:
    credential = AzureKeyCredential(settings.azure_search_api_key)
    return SearchIndexClient(endpoint=settings.azure_search_endpoint, credential=credential)


def _get_search_client() -> SearchClient:
    credential = AzureKeyCredential(settings.azure_search_api_key)
    return SearchClient(endpoint=settings.azure_search_endpoint, index_name=settings.azure_search_index_name, credential=credential)


def _get_openai_client() -> OpenAIClient:
    return OpenAIClient(endpoint=settings.azure_openai_endpoint, credential=AzureKeyCredential(settings.azure_openai_key))


def _ensure_vector_index():
    index_client = _get_search_index_client()
    index_name = settings.azure_search_index_name

    if index_name in [idx.name for idx in index_client.list_index_names()]:
        return

    vector_config = VectorSearch(
        algorithm_configurations=[
            VectorSearchAlgorithmConfiguration(
                name="hnsw",
                kind="hnsw",
                parameters=HnswParameters(m=4, ef_construction=200, ef_search=100, metric="cosine"),
            )
        ]
    )

    index = SearchIndex(
        name=index_name,
        fields=[
            SimpleField(name="id", type=SearchFieldDataType.String, key=True, filterable=True),
            SearchableField(name="content", type=SearchFieldDataType.String, searchable=True),
            SimpleField(name="source", type=SearchFieldDataType.String, filterable=True),
            SearchField(
                name="embedding",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=False,
                filterable=False,
                sortable=False,
                facetable=False,
                vector_search_dimensions=1536,
            ),
        ],
        vector_search=vector_config,
    )

    index_client.create_index(index)


def _load_kb_files() -> List[Dict]:
    kb_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "kb")
    files = glob.glob(os.path.join(kb_path, "**", "*.md"), recursive=True) + glob.glob(os.path.join(kb_path, "**", "*.txt"), recursive=True)
    docs = []

    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if not text:
            continue

        base_id = os.path.splitext(os.path.basename(file_path))[0]
        for idx, chunk in enumerate(_chunk_text(text, settings.chunk_size, settings.chunk_overlap)):
            docs.append({
                "id": f"{base_id}_{idx}",
                "source": os.path.basename(file_path),
                "content": chunk,
            })

    return docs


def _chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    tokens = text.split()
    if not tokens:
        return []

    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk = " ".join(tokens[start:end])
        chunks.append(chunk)
        if end == len(tokens):
            break
        start += chunk_size - overlap

    return chunks


def ingest_documents() -> int:
    docs = _load_kb_files()
    if not docs:
        return 0

    _ensure_vector_index()
    search_client = _get_search_client()
    openai_client = _get_openai_client()

    batch_size = 25
    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        texts = [d["content"] for d in batch]

        embeddings_resp = openai_client.embeddings.create(
            model=settings.azure_openai_embedding_model,
            input=texts,
        )

        for d, emb in zip(batch, embeddings_resp.data):
            d["embedding"] = emb.embedding

        search_client.upload_documents(documents=batch)

    return len(docs)
