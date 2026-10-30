"""
frontend/app.py
---------------
Enterprise-Grade AI Policy QA System.
Stunning dark-mode UI. No technical clutter exposed to users.
"""

import os
import requests
import streamlit as st
from pathlib import Path

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PolicyAI — Intelligent Policy Assistant",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"Get help": None, "Report a bug": None, "About": None}
)

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

# ─── Inject CSS ────────────────────────────────────────────────────────────────
css_path = Path(__file__).parent / "style.css"
if css_path.exists():
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ─── Backend helpers ───────────────────────────────────────────────────────────

def backend_ok() -> bool:
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def get_stats() -> dict:
    if backend_ok():
        try:
            r = requests.get(f"{BACKEND_URL}/documents", timeout=5)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
    try:
        from rag.vector_store import VectorStoreManager
        return VectorStoreManager().get_collection_stats()
    except Exception as e:
        return {"total_chunks": 0, "total_documents": 0, "documents": [], "error": str(e)}


def ask(question: str, top_k: int = 4) -> dict:
    if backend_ok():
        r = requests.post(f"{BACKEND_URL}/ask", json={"question": question, "top_k": top_k}, timeout=60)
        if r.status_code == 200:
            return r.json()
        raise RuntimeError(r.json().get("detail", r.text))
    from rag.qa_chain import PolicyQAEngine
    return PolicyQAEngine(top_k=top_k).answer_question(question=question, top_k=top_k)


def upload(files) -> dict:
    if backend_ok():
        payload = [("files", (f.name, f.getvalue(), "application/pdf")) for f in files]
        r = requests.post(f"{BACKEND_URL}/upload", files=payload, timeout=120)
        if r.status_code == 200:
            return r.json()
        raise RuntimeError(r.json().get("detail", r.text))
    from rag.config import DOCUMENTS_DIR
    from rag.document_loader import load_pdf_file
    from rag.text_splitter import split_documents
    from rag.vector_store import VectorStoreManager
    vsm = VectorStoreManager()
    names, docs = [], []
    for f in files:
        p = Path(DOCUMENTS_DIR) / f.name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(f.getvalue())
        names.append(f.name)
        docs.extend(load_pdf_file(str(p)))
    chunks = split_documents(docs)
    vsm.add_documents(chunks)
    s = vsm.get_collection_stats()
    return {"message": f"Indexed {len(names)} document(s).", "uploaded_files": names,
            "pages_extracted": len(docs), "chunks_created": len(chunks),
            "total_collection_chunks": s.get("total_chunks", len(chunks))}


def do_reindex() -> dict:
    if backend_ok():
        r = requests.post(f"{BACKEND_URL}/reindex", timeout=120)
        return r.json()
    from rag.config import DOCUMENTS_DIR
    from rag.document_loader import load_pdf_directory
    from rag.text_splitter import split_documents
    from rag.vector_store import VectorStoreManager
    vsm = VectorStoreManager()
    chunks = split_documents(load_pdf_directory(DOCUMENTS_DIR))
    vsm.add_documents(chunks)
    return {"message": "Re-indexed successfully.", "chunks_indexed": len(chunks)}


def do_reset() -> dict:
    if backend_ok():
        r = requests.post(f"{BACKEND_URL}/reset", timeout=15)
        return r.json()
    from rag.vector_store import VectorStoreManager
    VectorStoreManager().reset_collection()
    return {"message": "Reset complete."}


# ─── Fetch live stats on every render ─────────────────────────────────────────
stats    = get_stats()
n_docs   = stats.get("total_documents", 0)
n_chunks = stats.get("total_chunks", 0)

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="sb-logo">🏛️ PolicyAI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-tagline">Intelligent Policy Assistant</div>', unsafe_allow_html=True)
    st.divider()

    # Live stats
    st.markdown('<div class="sb-section-label">Knowledge Base</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f'<div class="sb-stat"><div class="sb-stat-num">{n_docs}</div><div class="sb-stat-lbl">Policies</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="sb-stat"><div class="sb-stat-num">{n_chunks}</div><div class="sb-stat-lbl">Passages</div></div>', unsafe_allow_html=True)

    st.write("")

    # Backend badge
    if backend_ok():
        st.markdown('<div class="badge-ok">● API Connected</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="badge-warn">⚡ Direct Mode</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="sb-section-label">Actions</div>', unsafe_allow_html=True)

    if st.button("🔄  Re-Index Knowledge Base", use_container_width=True):
        with st.spinner("Re-indexing …"):
            try:
                do_reindex()
                st.success("Knowledge base refreshed!")
                st.rerun()
            except Exception as e:
                st.error(str(e))

    if st.button("🗑️  Clear Knowledge Base", use_container_width=True):
        with st.spinner("Clearing …"):
            try:
                do_reset()
                st.warning("Knowledge base cleared!")
                st.rerun()
            except Exception as e:
                st.error(str(e))

    st.divider()
    st.markdown("""
    <div class="sb-info">
        <b>How it works</b><br>
        Upload policy PDFs → AI indexes & understands them → Ask questions in plain English → Get cited, fact-grounded answers.
    </div>
    """, unsafe_allow_html=True)


# ─── HERO HEADER ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-pill">AI-Powered · Fact-Grounded · Zero Hallucination</div>
    <h1 class="hero-title">Policy Intelligence Assistant</h1>
    <p class="hero-sub">Ask anything about your organization's policies and get precise, cited answers — instantly.</p>
    <div class="hero-stats">
        <div class="hs-item"><span class="hs-n">{n_docs}</span><span class="hs-l">Active Policies</span></div>
        <div class="hs-sep">·</div>
        <div class="hs-item"><span class="hs-n">{n_chunks}</span><span class="hs-l">Indexed Passages</span></div>
        <div class="hs-sep">·</div>
        <div class="hs-item"><span class="hs-n">100%</span><span class="hs-l">Grounded Answers</span></div>
    </div>
</div>
""".format(n_docs=n_docs, n_chunks=n_chunks), unsafe_allow_html=True)


# ─── MAIN TABS ────────────────────────────────────────────────────────────────
tab_ask, tab_upload, tab_library = st.tabs(["💬 Ask Questions", "📤 Upload Policies", "📑 Policy Library"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — ASK QUESTIONS
# ══════════════════════════════════════════════════════════════════════════════
with tab_ask:
    st.write("")

    SAMPLES = [
        "What is the minimum attendance requirement to appear in examinations?",
        "What is the fee and procedure for examination re-evaluation?",
        "What is the hostel curfew time for resident students?",
        "What are the penalties for violating the anti-ragging policy?",
        "What is the fee refund policy if an admission is cancelled?",
        "What are the library overdue fine charges?",
    ]

    st.markdown('<div class="section-label">💡 Try a Sample Question</div>', unsafe_allow_html=True)

    # 3 columns for sample chips
    cols = st.columns(3)
    selected_sample = None
    for i, q in enumerate(SAMPLES):
        with cols[i % 3]:
            if st.button(q, key=f"sample_{i}", use_container_width=True):
                selected_sample = q

    st.write("")

    # Pull selected sample into session state
    if selected_sample:
        st.session_state["ask_input"] = selected_sample

    user_q = st.text_area(
        "Your question",
        value=st.session_state.get("ask_input", ""),
        height=80,
        placeholder="e.g. What is the attendance policy for medical leave?",
        key="ask_text_area",
        label_visibility="collapsed"
    )

    search_btn = st.button("🔍 Search & Answer", type="primary", use_container_width=True, key="search_btn")

    if (search_btn or selected_sample) and user_q.strip():
        if n_chunks == 0:
            st.markdown("""
            <div class="info-box">
                📂 <strong>No documents indexed yet.</strong>
                Go to the <strong>Upload Policies</strong> tab to add PDF files first.
            </div>
            """, unsafe_allow_html=True)
        else:
            with st.spinner("Searching knowledge base and generating answer …"):
                try:
                    result = ask(user_q.strip())

                    answer  = result.get("answer", "No answer found.")
                    sources = result.get("sources", [])

                    st.markdown('<div class="answer-card">', unsafe_allow_html=True)
                    st.markdown('<div class="answer-label">📋 Answer</div>', unsafe_allow_html=True)
                    st.markdown(answer)
                    st.markdown('</div>', unsafe_allow_html=True)

                    if sources:
                        st.write("")
                        st.markdown('<div class="section-label">📚 Source Citations</div>', unsafe_allow_html=True)
                        for idx, src in enumerate(sources, 1):
                            with st.expander(f"📍 Source {idx}  —  {src.get('source')}  (Page {src.get('page')})", expanded=(idx == 1)):
                                cc1, cc2, cc3 = st.columns(3)
                                cc1.markdown(f"**Document**  \n`{src.get('source')}`")
                                cc2.markdown(f"**Page**  \n`{src.get('page')}`")
                                cc3.markdown(f"**Section ID**  \n`{src.get('chunk_id')}`")
                                st.markdown("**Relevant excerpt:**")
                                st.markdown(f'<div class="excerpt">{src.get("excerpt", "")}</div>', unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — UPLOAD POLICIES
# ══════════════════════════════════════════════════════════════════════════════
with tab_upload:
    st.write("")
    st.markdown("""
    <div class="upload-hero">
        <div class="uh-icon">📂</div>
        <div class="uh-title">Add Policy Documents</div>
        <div class="uh-sub">Upload PDF files and they'll be instantly indexed into the knowledge base.</div>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drop PDF files here",
        type=["pdf"],
        accept_multiple_files=True,
        help="Only PDF files are supported.",
        label_visibility="collapsed"
    )

    if uploaded:
        st.markdown(f'<div class="file-preview">📄 {len(uploaded)} file(s) selected: {", ".join(f.name for f in uploaded)}</div>', unsafe_allow_html=True)
        if st.button("🚀 Index Documents", type="primary", use_container_width=True):
            with st.spinner(f"Processing and indexing {len(uploaded)} document(s) …"):
                try:
                    res = upload(uploaded)
                    st.success(res.get("message", "Done!"))
                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric("Pages Extracted", res.get("pages_extracted", 0))
                    col_b.metric("Passages Created", res.get("chunks_created", 0))
                    col_c.metric("Total in Knowledge Base", res.get("total_collection_chunks", 0))
                    st.rerun()
                except Exception as e:
                    st.error(f"Upload failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — POLICY LIBRARY
# ══════════════════════════════════════════════════════════════════════════════
with tab_library:
    st.write("")
    docs_list = stats.get("documents", [])

    if not docs_list:
        st.markdown("""
        <div class="info-box">
            📂 <strong>Library is empty.</strong> Upload policy PDFs to populate the knowledge base.
        </div>
        """, unsafe_allow_html=True)
    else:
        # Refresh button
        if st.button("🔄 Refresh", key="lib_refresh"):
            st.session_state.stats = get_stats()
            st.rerun()

        search_q = st.text_input("Filter policies …", placeholder="Type to search …", key="lib_search", label_visibility="collapsed")
        shown = [d for d in docs_list if not search_q or search_q.lower() in d.get("source", "").lower()]

        st.write(f"**{len(shown)} of {len(docs_list)} policies**")
        st.write("")

        for doc in shown:
            name   = doc.get("source", "Unknown")
            pages  = doc.get("indexed_pages", 0)
            pg_lst = doc.get("pages_list", [])
            pg_str = ", ".join(str(p) for p in pg_lst)

            st.markdown(f"""
            <div class="lib-card">
                <div class="lib-icon">📘</div>
                <div class="lib-info">
                    <div class="lib-name">{name}</div>
                    <div class="lib-meta">{pages} page(s) indexed &nbsp;·&nbsp; Pages: {pg_str}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
