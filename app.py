import sys
from pathlib import Path
import tempfile
import os

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from src.agents.qa_agent import QAAgent
from src.agents.designer_agent import DesignerAgent
from src.agents.validator_agent import ValidatorAgent
from src.ingestion.ingest import ingest_document

st.set_page_config(page_title="Geotech AI Agent", page_icon="🏗️", layout="wide", initial_sidebar_state="expanded")

if "qa_agent" not in st.session_state:
    st.session_state.qa_agent = QAAgent(use_local_db=True)
if "designer_agent" not in st.session_state:
    st.session_state.designer_agent = None
if "validator_agent" not in st.session_state:
    st.session_state.validator_agent = ValidatorAgent(use_local_db=True)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "designer_messages" not in st.session_state:
    st.session_state.designer_messages = []

with st.sidebar:
    st.title("🏗️ Geotech AI Agent")
    st.caption("AI-powered geotechnical engineering assistant")
    mode = st.radio("Select Mode", ["📚 Knowledge Q&A", "📝 Report Generator", "✅ Submission Checker", "📂 Document Manager"], index=0)

if mode == "📚 Knowledge Q&A":
    st.header("📚 Knowledge Base Q&A")
    equation_mode = st.toggle(
        "Equation-focused mode",
        value=False,
        help="Prioritize critical design equations, variable definitions, and check criteria in answers.",
    )
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📎 Sources"):
                    for s in msg["sources"]:
                        st.caption(f"• {s['document']} | {s['section']} (score: {s['relevance_score']})")
    if prompt := st.chat_input("Ask about geotechnical codes and practice..."):
        user_content = f"{prompt}\n\n[Equation mode: {'ON' if equation_mode else 'OFF'}]"
        st.session_state.chat_history.append({"role": "user", "content": user_content})
        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge base..."):
                result = st.session_state.qa_agent.ask(prompt, equation_mode=equation_mode)
            st.markdown(result["answer"])
            if result["sources"]:
                with st.expander("📎 Sources"):
                    for s in result["sources"]:
                        st.caption(f"• {s['document']} | {s['section']} (score: {s['relevance_score']})")
        st.session_state.chat_history.append({"role": "assistant", "content": result["answer"], "sources": result["sources"]})

elif mode == "📝 Report Generator":
    st.header("📝 Geotechnical Report Generator")
    if st.button("🔄 Start New Report"):
        st.session_state.designer_agent = DesignerAgent(use_local_db=True)
        st.session_state.designer_messages = []
        result = st.session_state.designer_agent.start()
        st.session_state.designer_messages.append({"role": "assistant", "content": result["message"]})
        st.rerun()
    for msg in st.session_state.designer_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("questions"):
                for q in msg["questions"]:
                    st.info(f"❓ {q}")
            if msg.get("report"):
                st.divider()
                st.markdown(msg["report"])
                st.download_button("📥 Download Report (Markdown)", msg["report"], file_name="geotechnical_report.md", mime="text/markdown")
    if st.session_state.designer_agent and st.session_state.designer_agent.state.value != "complete":
        if prompt := st.chat_input("Provide project information or answer questions..."):
            st.session_state.designer_messages.append({"role": "user", "content": prompt})
            with st.spinner("Processing..."):
                result = st.session_state.designer_agent.process_input(prompt)
            payload = {"role": "assistant", "content": result.get("message", ""), "questions": result.get("questions", [])}
            if result.get("report"):
                payload["report"] = result["report"]
            st.session_state.designer_messages.append(payload)
            st.rerun()
    elif not st.session_state.designer_agent:
        st.info("Click 'Start New Report' to begin.")

elif mode == "✅ Submission Checker":
    st.header("✅ Submission Compliance Checker")
    uploaded_file = st.file_uploader("Upload a geotechnical report (PDF)", type=["pdf"])
    if uploaded_file and st.button("🔍 Run Compliance Check", type="primary"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name
        try:
            with st.spinner("Analysing report..."):
                report = st.session_state.validator_agent.validate_report(tmp_path)
                formatted = st.session_state.validator_agent.format_validation_report(report)
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Overall Status", report.overall_status.upper().replace("_", " "))
            col2.metric("✅ Passed", report.pass_count)
            col3.metric("❌ Failed", report.fail_count)
            col4.metric("⚠️ Warnings", report.warning_count)
            st.divider()
            st.markdown(formatted)
            st.download_button("📥 Download Review Report", formatted, file_name=f"review_{uploaded_file.name.replace('.pdf', '.md')}", mime="text/markdown")
        finally:
            os.unlink(tmp_path)

elif mode == "📂 Document Manager":
    st.header("📂 Knowledge Base Manager")
    docs = st.session_state.qa_agent.list_knowledge_base()
    if docs:
        for doc in docs:
            col1, col2, col3 = st.columns([3, 1, 1])
            col1.write(f"📄 **{doc['document_name']}**")
            col2.write(f"Type: {doc['document_type']}")
            col3.write(f"Chunks: {doc['chunk_count']}")
        st.divider()
        st.subheader("Inspect Stored Chunks")
        doc_options = {f"{d['document_name']} ({d['document_id']})": d["document_id"] for d in docs}
        selected_doc_label = st.selectbox("Select document", list(doc_options.keys()))
        selected_doc_id = doc_options[selected_doc_label]
        sample_limit = st.slider("Sample chunk count", min_value=1, max_value=20, value=5)
        if st.button("🔎 Show Raw Chunk Samples"):
            samples = st.session_state.qa_agent.store.get_document_chunks(selected_doc_id, limit=sample_limit)
            if not samples:
                st.warning("No chunk samples found.")
            for i, s in enumerate(samples, 1):
                st.markdown(f"**Chunk {i}**")
                st.caption(
                    f"Section: {s['metadata'].get('section_title', 'N/A')} | "
                    f"Page: {s['metadata'].get('page_number', 'N/A')} | "
                    f"Sub-chunk: {s['metadata'].get('sub_chunk_index', 'N/A')}"
                )
                st.code(s["text"][:1200], language="text")
        st.subheader("Semantic Retrieval Check")
        verify_query = st.text_input("Query to validate chunking/retrieval", value="scope of bolted end plate connections")
        if st.button("✅ Run Retrieval Check"):
            results = st.session_state.qa_agent.store.search(
                query=verify_query,
                top_k=5,
                document_id=selected_doc_id,
                score_threshold=0.0,
            )
            if not results:
                st.warning("No retrieval results for this query.")
            for i, r in enumerate(results, 1):
                st.markdown(f"**Result {i} (score: {r['score']:.3f})**")
                st.caption(
                    f"Section: {r['metadata'].get('section_title', 'N/A')} | "
                    f"Page: {r['metadata'].get('page_number', 'N/A')}"
                )
                st.code(r["text"][:1200], language="text")
    else:
        st.info("No documents ingested yet.")
    st.divider()
    uploaded = st.file_uploader("Upload a PDF document", type=["pdf"], key="ingest_upload")
    if uploaded:
        col1, col2 = st.columns(2)
        with col1:
            doc_name = st.text_input("Document Name", value=uploaded.name.replace(".pdf", ""))
            doc_id = st.text_input("Document ID", value=uploaded.name.replace(".pdf", "").lower().replace(" ", "_"))
        with col2:
            doc_type = st.selectbox("Document Type", ["code", "manual", "guideline", "report", "drawing"])
        if st.button("📥 Ingest Document", type="primary"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded.getbuffer())
                tmp_path = tmp.name
            try:
                with st.spinner(f"Processing {doc_name}..."):
                    num_chunks = ingest_document(
                        pdf_path=tmp_path,
                        document_id=doc_id,
                        document_name=doc_name,
                        document_type=doc_type,
                        use_local_db=True,
                    )
                st.success(f"Successfully ingested '{doc_name}' ({num_chunks} chunks)")
                st.rerun()
            finally:
                os.unlink(tmp_path)
