import dotenv
dotenv.load_dotenv()

import tempfile
import os
import streamlit as st
from mangrag import ingest, query
from mangrag.config import settings
from mangrag.db import get_collection

st.set_page_config(page_title="MangRAG", page_icon="🥭", layout="wide")


# ─── Connection ──────────────────────────────────────────────────────────────

@st.cache_resource
def get_col():
    try:
        return get_collection()
    except Exception as e:
        st.error(f"MongoDB connection failed: {e}")
        st.stop()


collection = get_col()


# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🥭 MangRAG")
    st.caption("MongoDB Atlas · Groq · HuggingFace")
    st.divider()

    st.subheader("🚀 How to get started")
    st.markdown("""
1. Go to the **📄 Ingest** tab
2. Paste a **PDF URL** (or local path) and click **Ingest**
3. Wait for all pipeline stages to complete
4. Switch to the **💬 Ask** tab
5. Type your question and click **Ask**
""")
    st.divider()

    st.subheader("Collection Stats")
    total = collection.count_documents({})
    sources = collection.distinct("source")
    c1, c2 = st.columns(2)
    c1.metric("Chunks", total)
    c2.metric("Sources", len(sources))

    if sources:
        with st.expander("Indexed sources"):
            for s in sources:
                st.caption(s if len(s) < 60 else f"...{s[-57:]}")

    st.divider()
    with st.expander("Config"):
        st.code(
            f"embeddings : {settings.embedding_model}\n"
            f"llm        : {settings.chat_model}\n"
            f"chunk_size : {settings.chunk_size}\n"
            f"overlap    : {settings.chunk_overlap}\n"
            f"top_k      : {settings.top_k}",
            language="text",
        )


# ─── Tabs ────────────────────────────────────────────────────────────────────

ask_tab, ingest_tab = st.tabs(["💬 Ask", "📄 Ingest"])


# ─── Ask Tab ─────────────────────────────────────────────────────────────────

with ask_tab:
    st.header("💬 Ask a question")

    if total == 0:
        st.info("No documents ingested yet. Go to the **📄 Ingest** tab first to load a PDF.", icon="ℹ️")
    else:
        st.caption(f"Searching across **{total} chunks** from **{len(sources)} source(s)**.")
        with st.expander("💡 Tips for better answers"):
            st.markdown("""
**Ask specific questions** — vector search finds chunks that are *semantically similar* to your query.

| Works well | Works poorly |
|---|---|
| "What is MongoDB Atlas?" | "summary of the uploaded doc" |
| "What are the revenue figures?" | "tell me everything" |
| "How does vector search work?" | "what is this document about?" |

For a summary, try: **"What are the main topics covered in this document?"**
""")


    user_query = st.text_input(
        "Question",
        placeholder="e.g. What are the key findings in this document?",
        label_visibility="collapsed",
        disabled=total == 0,
    )
    ask_btn = st.button("Ask", type="primary", disabled=total == 0)

    if ask_btn and user_query.strip():
        with st.status("Running query pipeline...", expanded=True) as status_box:
            st.write("🔍 Embedding query and searching MongoDB...")
            docs = query.retrieve(user_query.strip(), collection)
            st.write(f"📄 Retrieved {len(docs)} chunks")
            st.write("🤖 Generating answer with Groq...")
            answer = query.build_answer(user_query.strip(), docs)
            status_box.update(label="Done", state="complete", expanded=False)

        st.subheader("Answer")
        st.markdown(answer)

        if docs:
            st.divider()
            st.subheader(f"Retrieved Chunks ({len(docs)})")
            st.caption("🟢 score ≥ 0.85 · highly relevant   🟡 score ≥ 0.70 · relevant   🔴 score < 0.70 · low relevance")
            for i, doc in enumerate(docs):
                score = doc.get("score", 0.0)
                badge = "🟢" if score >= 0.85 else ("🟡" if score >= 0.70 else "🔴")
                with st.expander(
                    f"{badge} Chunk {i+1}  —  Page {doc.get('page', '?')}  |  score {score:.3f}"
                ):
                    st.progress(float(score), text=f"Similarity: {score:.1%}")
                    st.caption(f"Source: {doc.get('source', 'unknown')}")
                    st.markdown(doc["content"])

    elif ask_btn:
        st.warning("Please enter a question.")


# ─── Ingest Tab ──────────────────────────────────────────────────────────────

with ingest_tab:
    st.header("📄 Ingest a PDF")
    st.markdown("""
Add a PDF to your knowledge base. Once ingested, you can ask questions about it in the **💬 Ask** tab.

| Stage | What it does |
|---|---|
| 📥 Load | Reads the PDF and extracts text |
| ✂️ Chunk | Splits text into overlapping segments (~800 chars each) |
| 🧠 Embed | Converts each chunk into a vector using HuggingFace |
| 💾 Store | Saves chunks + vectors to MongoDB Atlas |
| 📐 Index | Creates/verifies the vector search index |
""")
    st.divider()

    input_mode = st.radio(
        "Source type",
        ["🔗 URL", "📁 Upload file"],
        horizontal=True,
    )

    source = None
    tmp_path = None

    if input_mode == "🔗 URL":
        url = st.text_input(
            "Public PDF URL",
            placeholder="https://example.com/report.pdf",
        )
        if url.strip():
            source = url.strip()

    else:
        uploaded = st.file_uploader("Choose a PDF file", type=["pdf"])
        if uploaded:
            # Save to a named temp file so PyPDFLoader can read it
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            tmp.write(uploaded.read())
            tmp.close()
            tmp_path = tmp.name
            source = tmp_path
            st.caption(f"Uploaded: **{uploaded.name}** ({uploaded.size / 1024:.1f} KB)")

    ingest_btn = st.button("Ingest", type="primary", disabled=source is None)

    if ingest_btn and source:
        progress_bar = st.progress(0, text="Starting...")

        def on_progress(current, total):
            progress_bar.progress(
                current / total,
                text=f"Embedding chunk {current}/{total}...",
            )

        display_name = source if input_mode == "🔗 URL" else uploaded.name

        try:
            with st.status("Running ingest pipeline...", expanded=True) as status_box:
                st.write("📥 Loading PDF...")
                pages = ingest.load_pdf(source)
                st.write(f"✅ Loaded {len(pages)} pages")

                st.write("✂️ Chunking text...")
                chunks = ingest.chunk_documents(pages)
                st.write(f"✅ Produced {len(chunks)} chunks")

                st.write("🧠 Embedding and storing in MongoDB...")
                count = ingest.embed_and_store(
                    chunks, collection, display_name, on_progress=on_progress
                )
                st.write(f"✅ Stored {count} chunks")

                st.write("📐 Ensuring vector search index...")
                from mangrag.db import ensure_vector_index
                ensure_vector_index(collection)
                st.write("✅ Index ready")

                status_box.update(label="Ingestion complete!", state="complete", expanded=False)

            progress_bar.empty()
            st.success(f"Done! Stored **{count}** chunks from **{display_name}**.")
            st.rerun()

        except Exception as e:
            progress_bar.empty()
            st.error(f"Ingestion failed: {e}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
