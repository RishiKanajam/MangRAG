import os
import time
import dns.resolver
from groq import Groq
from langchain_community.embeddings import HuggingFaceEmbeddings
from pymongo import MongoClient
from pymongo.operations import SearchIndexModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Fix DNS resolution issues on some networks
dns.resolver.default_resolver = dns.resolver.Resolver(configure=False)
dns.resolver.default_resolver.nameservers = ['8.8.8.8', '8.8.4.4']

EMBEDDING_MODEL = 'all-MiniLM-L6-v2'   # local, no API, 80MB, 384 dims
EMBEDDING_DIMS = 384
CHAT_MODEL = 'llama-3.3-70b-versatile'  # Groq free tier
INDEX_NAME = 'vector_index'
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K = 5
NUM_CANDIDATES = 150

_api_key = os.environ.get('GROQ_API_KEY')
if not _api_key:
    raise ValueError("GROQ_API_KEY is not set. Add it to your .env file.")

_groq = Groq(api_key=_api_key)
_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


# ─── Stage 0: Connect ────────────────────────────────────────────────────────

def get_mongo_collection():
    uri = os.environ['MONGODB_URI']
    client = MongoClient(uri)
    db = os.environ.get('MONGODB_DB', 'smiple_mflix')
    col = os.environ.get('MONGODB_COLLECTION', 'rag_pdf')
    return client[db][col]


# ─── Stage 1: Load ───────────────────────────────────────────────────────────

def load_pdf(pdf_path_or_url: str, on_step=None) -> list:
    if on_step:
        on_step("load", f"Loading: {pdf_path_or_url}")
    loader = PyPDFLoader(pdf_path_or_url)
    pages = loader.load()
    if on_step:
        on_step("load", f"Loaded {len(pages)} pages")
    return pages


# ─── Stage 2: Chunk ──────────────────────────────────────────────────────────

def chunk_documents(pages: list, on_step=None) -> list:
    if on_step:
        on_step("chunk", f"Chunking ({CHUNK_SIZE} chars, {CHUNK_OVERLAP} overlap)")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(pages)
    chunks = [c for c in chunks if c.page_content.strip()]
    if on_step:
        on_step("chunk", f"Produced {len(chunks)} chunks")
    return chunks


# ─── Stage 3: Embed & Store ──────────────────────────────────────────────────

def get_embedding(text: str) -> list[float]:
    return _embeddings.embed_query(text)


def ensure_vector_index(collection, on_step=None):
    existing = [idx['name'] for idx in collection.list_search_indexes()]
    if INDEX_NAME in existing:
        return
    if on_step:
        on_step("store", "Creating vector search index...")
    model = SearchIndexModel(
        definition={
            'fields': [{
                'type': 'vector',
                'numDimensions': EMBEDDING_DIMS,
                'path': 'embedding',
                'similarity': 'cosine',
            }]
        },
        name=INDEX_NAME,
        type='vectorSearch',
    )
    collection.create_search_index(model=model)
    for _ in range(30):
        statuses = [
            idx.get('status')
            for idx in collection.list_search_indexes()
            if idx['name'] == INDEX_NAME
        ]
        if statuses and statuses[0] == 'READY':
            break
        time.sleep(2)


def embed_and_store(chunks: list, collection, source: str,
                    on_step=None, on_progress=None) -> int:
    if on_step:
        on_step("embed", f"Embedding {len(chunks)} chunks...")
    docs_to_insert = []
    for i, chunk in enumerate(chunks):
        docs_to_insert.append({
            "content": chunk.page_content,
            "embedding": get_embedding(chunk.page_content),
            "source": source,
            "page": chunk.metadata.get("page", 0),
        })
        if on_progress:
            on_progress(i + 1, len(chunks))

    collection.insert_many(docs_to_insert)
    if on_step:
        on_step("store", f"Stored {len(docs_to_insert)} documents in MongoDB")
    return len(docs_to_insert)


# ─── Stage 4: Retrieve ───────────────────────────────────────────────────────

def retrieve(query: str, collection, top_k: int = TOP_K, on_step=None) -> list[dict]:
    if on_step:
        on_step("retrieve", f'Retrieving top {top_k} chunks for: "{query}"')
    query_embedding = get_embedding(query)

    pipeline = [
        {
            '$vectorSearch': {
                'index': INDEX_NAME,
                'queryVector': query_embedding,
                'path': 'embedding',
                'numCandidates': NUM_CANDIDATES,
                'limit': top_k,
            }
        },
        {
            '$project': {
                '_id': 0,
                'content': 1,
                'source': 1,
                'page': 1,
                'score': {'$meta': 'vectorSearchScore'},
            }
        }
    ]
    return list(collection.aggregate(pipeline))


# ─── Stage 5: Generate ───────────────────────────────────────────────────────

def generate_answer(query: str, context_docs: list[dict], on_step=None) -> str:
    if on_step:
        on_step("generate", f"Generating answer with {CHAT_MODEL}...")
    if not context_docs:
        return "I could not find relevant information to answer your question."

    context = "\n\n---\n\n".join(
        f"[Page {doc.get('page', '?')}] {doc['content']}"
        for doc in context_docs
    )

    prompt = f"""You are a helpful assistant. Answer the question using ONLY the provided context.
If the context does not contain enough information, say "I don't have enough information to answer that."

Context:
{context}

Question: {query}
Answer:"""

    response = _groq.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{'role': 'user', 'content': prompt}],
        temperature=0,
    )
    return response.choices[0].message.content


# ─── Full Pipelines ──────────────────────────────────────────────────────────

def ingest_pipeline(pdf_path_or_url: str, collection,
                    on_step=None, on_progress=None) -> int:
    pages = load_pdf(pdf_path_or_url, on_step=on_step)
    chunks = chunk_documents(pages, on_step=on_step)
    count = embed_and_store(chunks, collection, source=pdf_path_or_url,
                            on_step=on_step, on_progress=on_progress)
    ensure_vector_index(collection, on_step=on_step)
    return count


def query_pipeline(query: str, collection,
                   on_step=None) -> tuple[str, list[dict]]:
    docs = retrieve(query, collection, on_step=on_step)
    answer = generate_answer(query, docs, on_step=on_step)
    return answer, docs
