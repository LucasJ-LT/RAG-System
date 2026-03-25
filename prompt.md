You are a senior backend engineer experienced with FastAPI,
Azure OpenAI, and Retrieval-Augmented Generation (RAG).
 
Your task is to generate a SMALL, CLEAN Proof-of-Concept (PoC)
RAG system for IT issue solutions, implemented as a FastAPI service,
designed for local development in VS Code.
 
========================
POC GOALS
========================
- Simple, understandable, minimal code
- Demonstrates end-to-end RAG clearly
- No over-engineering
- Easy to extend to production later
 
========================
TECH STACK
========================
- Language: Python 3.10+
- API Framework: FastAPI
- Server: Uvicorn
- LLM: Azure OpenAI (GPT-4o or GPT-4o-mini)
- Embeddings: text-embedding-3-large
- Vector Store: Azure AI Search
- No LangChain / LlamaIndex
- Use environment variables for secrets
 
========================
FUNCTIONAL REQUIREMENTS
========================
1. Load IT knowledge files from a local `data/kb/` folder (.txt or .md)
2. Chunk documents (400 tokens, ~80 overlap)
3. Create embeddings and store them in Azure AI Search
4. Expose ONE ingestion endpoint:
   POST /ingest
5. Expose ONE query endpoint:
   POST /query
   Input: user question
   Output: RAG answer
6. Vector search Top K = 5
7. Strict RAG behavior:
   - Answer ONLY from retrieved context
   - If information is missing, return exactly:
     "Not found in IT knowledge base."
 
========================
RAG SYSTEM PROMPT
========================
Use this exact SYSTEM prompt in generation:
 
"You are an enterprise IT support assistant.
Use ONLY the retrieved knowledge base content.
If the answer is not present, say exactly:
'Not found in IT knowledge base.'"
 
========================
API DESIGN
========================
POST /ingest
- Reads files from data/kb/
- Creates embeddings
- Indexes into Azure AI Search
- Returns number of documents indexed
 
POST /query
Request JSON:
{
  "question": "string"
}
 
Response JSON:
{
  "answer": "string",
  "sources": ["doc_id1", "doc_id2"]
}
 
========================
PROJECT STRUCTURE (PoC)
========================
rag-fastapi-poc/
├── app/
│   ├── main.py              # FastAPI app
│   ├── ingest.py            # ingestion logic
│   ├── retrieve.py          # vector search
│   ├── rag.py               # prompt + generation
│   ├── config.py            # settings
├── data/
│   └── kb/                  # sample IT docs
├── requirements.txt
├── .env.example
├── README.md
 
========================
DELIVERY RULES
========================
- Generate FULL runnable code for every file
- Do not leave TODOs or placeholders
- Use clear comments
- Keep the code simple and readable
- Assume the user runs:
  uvicorn app.main:app --reload
 
========================
DOCUMENTATION
========================
README.md must include:
1. Prerequisites
2. Environment variables
3. How to ingest documents
4. How to query the API
5. Example curl commands
 
Start with a concise explanation of the architecture,
then generate the full project file-by-file.