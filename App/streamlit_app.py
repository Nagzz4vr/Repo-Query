import os
import time
import json
import requests
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────
BASE_URL = os.getenv("RAG_API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="DevDocs AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    h1, h2, h3, .stMarkdown h1 {
        font-family: 'IBM Plex Mono', monospace !important;
        letter-spacing: -0.03em;
    }
    code, pre, .stCode {
        font-family: 'IBM Plex Mono', monospace !important;
    }
    .tag {
        display: inline-block;
        background: #0f172a;
        color: #38bdf8;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        padding: 2px 8px;
        border-radius: 2px;
        margin-bottom: 4px;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .answer-box {
        background: #f8fafc;
        color: #0f172a !important;
        border-left: 3px solid #0ea5e9;
        padding: 1rem 1.25rem;
        border-radius: 0 4px 4px 0;
        font-size: 0.95rem;
        line-height: 1.7;
        white-space: pre-wrap;
    }
    .answer-box * {
        color: #0f172a !important;
    }
    .source-chip {
        display: inline-block;
        background: #e0f2fe;
        color: #0369a1;
        font-size: 0.75rem;
        font-family: 'IBM Plex Mono', monospace;
        padding: 2px 10px;
        border-radius: 20px;
        margin: 2px;
    }
    .status-ok   { color: #16a34a; font-weight: 600; }
    .status-err  { color: #dc2626; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def post(endpoint: str, payload: dict) -> dict:
    """POST to the FastAPI backend; raises on non-2xx."""
    r = requests.post(f"{BASE_URL}{endpoint}", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()


def stream_query(question: str):
    """Yield tokens from the /query/stream SSE endpoint."""
    with requests.post(
        f"{BASE_URL}/query/stream",
        json={"question": question},
        stream=True,
        timeout=120,
    ) as r:
        r.raise_for_status()
        for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
            if chunk:
                yield chunk


def reset_memory() -> dict:
    r = requests.post(f"{BASE_URL}/query/reset", timeout=30)
    r.raise_for_status()
    return r.json()


# ── Session state ─────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []   # list[dict]  role / content / sources


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<span class="tag">DevDocs AI</span>', unsafe_allow_html=True)
    st.title("Ingest")
    st.caption("Index sources into the vector store.")

    ingest_tab, settings_tab = st.tabs(["📥 Sources", "⚙️ Settings"])

    with ingest_tab:
        st.subheader("GitHub repo")
        github_repo  = st.text_input("Repo URL", placeholder="https://github.com/org/repo")
        github_token = st.text_input("Token (private repos)", type="password")

        st.subheader("PDF files")
        uploaded_pdfs = st.file_uploader(
            "Upload PDF documents",
            type=["pdf"],
            accept_multiple_files=True,
            help="PDFs are saved temporarily on the server for ingestion.",
        )

        st.subheader("URLs")
        urls_raw = st.text_area(
            "One URL per line",
            placeholder="https://docs.example.com/intro",
            height=80,
        )

        if st.button("▶ Run ingest", use_container_width=True, type="primary"):
            # Save uploaded PDFs to a temp dir and collect their paths
            pdf_paths = []
            if uploaded_pdfs:
                import tempfile, pathlib
                tmp_dir = pathlib.Path(tempfile.mkdtemp())
                for f in uploaded_pdfs:
                    dest = tmp_dir / f.name
                    dest.write_bytes(f.read())
                    pdf_paths.append(str(dest))

            payload = {
                "github_repo":  github_repo  or None,
                "github_token": github_token or None,
                "local_path":   None,
                "pdf_paths":    pdf_paths or None,
                "urls":         [l.strip() for l in urls_raw.splitlines() if l.strip()] or None,
            }

            if all(v is None for v in payload.values()):
                st.warning("Provide at least one source.")
            else:
                with st.spinner("Ingesting — this may take a few minutes for large repos…"):
                    try:
                        r = requests.post(f"{BASE_URL}/ingest/", json=payload, timeout=30)
                        r.raise_for_status()
                        job_id = r.json()["job_id"]

                        status_placeholder = st.empty()
                        while True:
                            time.sleep(5)
                            poll = requests.get(f"{BASE_URL}/ingest/status/{job_id}", timeout=10)
                            poll.raise_for_status()
                            data = poll.json()

                            if data["status"] == "done":
                                status_placeholder.success("Ingestion complete.")
                                with st.expander("Summary"):
                                    st.json(data["summary"])
                                break
                            elif data["status"] == "failed":
                                status_placeholder.error(f"Ingestion failed: {data['error']}")
                                break
                            else:
                                status_placeholder.info(f"Still running… (job: {job_id[:8]})")

                    except Exception as e:
                        st.error(str(e))

    with settings_tab:
        st.subheader("API")
        new_url = st.text_input("Backend URL", value=BASE_URL)
        if new_url != BASE_URL:
            BASE_URL = new_url          # live override for the session
            st.info("URL updated for this session.")

        st.subheader("Streaming")
        use_stream = st.toggle("Stream responses", value=True)

        st.divider()
        if st.button("🗑 Reset conversation memory", use_container_width=True):
            try:
                reset_memory()
                st.session_state.chat_history = []
                st.success("Memory reset.")
                st.rerun()
            except Exception as e:
                st.error(str(e))


# ── Main chat area ────────────────────────────────────────────────────────────
st.markdown('<span class="tag">Query</span>', unsafe_allow_html=True)
st.title("DevDocs AI")
st.caption("Ask questions about your indexed codebase or documents.")

# Render existing chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            st.markdown(
                f'<div class="answer-box">{msg["content"]}</div>',
                unsafe_allow_html=True,
            )
            if msg.get("sources"):
                st.markdown("**Sources**")
                chips = "".join(
                    f'<span class="source-chip">{s}</span>'
                    for s in msg["sources"]
                )
                st.markdown(chips, unsafe_allow_html=True)
        else:
            st.write(msg["content"])

# Chat input
if question := st.chat_input("Ask anything about your codebase…"):

    # Append and display user message
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    # Assistant response
    with st.chat_message("assistant"):
        answer_placeholder = st.empty()
        sources: list[str] = []

        if use_stream:
            # ── Streaming path ──
            accumulated = ""
            try:
                for token in stream_query(question):
                    accumulated += token
                    answer_placeholder.markdown(
                        f'<div class="answer-box">{accumulated}▌</div>',
                        unsafe_allow_html=True,
                    )
                # Final render without cursor
                answer_placeholder.markdown(
                    f'<div class="answer-box">{accumulated}</div>',
                    unsafe_allow_html=True,
                )
                answer = accumulated
            except requests.HTTPError as e:
                answer = f"❌ HTTP {e.response.status_code}: {e.response.text}"
                answer_placeholder.error(answer)
            except Exception as e:
                answer = f"❌ {e}"
                answer_placeholder.error(answer)

        else:
            # ── Non-streaming path ──
            try:
                data   = post("/query/", {"question": question})
                answer = data.get("answer", json.dumps(data, indent=2))
                sources = data.get("sources", [])
                answer_placeholder.markdown(
                    f'<div class="answer-box">{answer}</div>',
                    unsafe_allow_html=True,
                )
            except requests.HTTPError as e:
                answer = f"❌ HTTP {e.response.status_code}: {e.response.text}"
                answer_placeholder.error(answer)
            except Exception as e:
                answer = f"❌ {e}"
                answer_placeholder.error(answer)

        if sources:
            st.markdown("**Sources**")
            chips = "".join(
                f'<span class="source-chip">{s}</span>' for s in sources
            )
            st.markdown(chips, unsafe_allow_html=True)

    # Persist assistant message
    st.session_state.chat_history.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )