import sys
from pathlib import Path
import tempfile
import os

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from src.agents.qa_agent import QAAgent
from src.agents.designer_agent import DesignerAgent
from src.agents.validator_agent import ValidatorAgent
from src.ingestion.ingest import ingest_document, ingest_document_dual_index
from src.utils.llm_client import (
    set_runtime_llm_config,
    get_runtime_llm_config,
    list_openai_models,
    list_google_models,
)
from src.utils.session_persistence import save_session, load_session, list_sessions, delete_session

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
if "llm_models" not in st.session_state:
    st.session_state.llm_models = []
if "llm_connected" not in st.session_state:
    st.session_state.llm_connected = False
if "session_id" not in st.session_state:
    st.session_state.session_id = "default"

# Load saved session on startup (simple approach without fragment)
if "session_loaded" not in st.session_state:
    saved = load_session(st.session_state.session_id)
    if saved:
        if saved.get("chat_history") and not st.session_state.chat_history:
            st.session_state.chat_history = saved["chat_history"]
        if saved.get("designer_messages") and not st.session_state.designer_messages:
            st.session_state.designer_messages = saved["designer_messages"]
    st.session_state.session_loaded = True

with st.sidebar:
    st.title("🏗️ Geotech AI Agent")
    st.caption("AI-powered geotechnical engineering assistant")

    # Dark mode toggle
    if "theme" not in st.session_state:
        st.session_state.theme = "light"

    theme = st.toggle("🌙 Dark Mode", value=st.session_state.theme == "dark")
    st.session_state.theme = "dark" if theme else "light"

    # Apply theme
    if theme:
        st.markdown("""
            <style>
            .stApp {background-color: #1e1e1e; color: #e0e0e0;}
            .stChatMessage {background-color: #2d2d2d;}
            </style>
        """, unsafe_allow_html=True)

    st.divider()

    mode = st.radio(
        "Select Mode",
        ["📚 Knowledge Q&A", "📝 Report Generator", "✅ Submission Checker", "📂 Document Manager", "⚙️ LLM Settings"],
        index=0,
    )

if mode == "📚 Knowledge Q&A":
    st.header("📚 Knowledge Base Q&A")
    equation_mode = st.toggle(
        "Equation-focused mode",
        value=False,
        help="Prioritize critical design equations, variable definitions, and check criteria in answers.",
    )
    with st.expander("Advanced Retrieval Filters"):
        docs = st.session_state.qa_agent.list_knowledge_base()
        doc_options = {"All documents": ""}
        for d in docs:
            doc_options[f"{d['document_name']} ({d['document_id']})"] = d["document_id"]
        selected_doc_label = st.selectbox("Document", list(doc_options.keys()))
        selected_doc_id = doc_options[selected_doc_label]
        clause_filter = st.text_input("Clause ID (e.g. 6.1.5)")
        content_type_filter = st.selectbox("Content Type", ["Any", "clause_text", "table"])
        regulatory_filter = st.selectbox("Regulatory Strength", ["Any", "mandatory", "advisory", "informative"])
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📎 Sources"):
                    for s in msg["sources"]:
                        st.caption(f"• {s['document']} | {s['section']} (score: {s['relevance_score']})")
    if prompt := st.chat_input("Ask about geotechnical codes and practice..."):
        applied_filters = (
            f"doc={selected_doc_id or 'all'}, clause={clause_filter or 'any'}, "
            f"content_type={content_type_filter}, regulatory={regulatory_filter}"
        )
        user_content = (
            f"{prompt}\n\n"
            f"[Equation mode: {'ON' if equation_mode else 'OFF'}]\n"
            f"[Filters: {applied_filters}]"
        )
        st.session_state.chat_history.append({"role": "user", "content": user_content})
        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge base..."):
                result = st.session_state.qa_agent.ask(
                    prompt,
                    document_id=selected_doc_id or None,
                    clause_id=clause_filter.strip() or None,
                    content_type=None if content_type_filter == "Any" else content_type_filter,
                    regulatory_strength=None if regulatory_filter == "Any" else regulatory_filter,
                    equation_mode=equation_mode,
                )
            st.markdown(result["answer"])
            if result["sources"]:
                with st.expander("📎 Sources"):
                    for s in result["sources"]:
                        st.caption(f"• {s['document']} | {s['section']} (score: {s['relevance_score']})")
        st.session_state.chat_history.append({"role": "assistant", "content": result["answer"], "sources": result["sources"]})
        # Auto-save session
        save_session(
            st.session_state.session_id,
            st.session_state.chat_history,
            st.session_state.designer_messages,
        )

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
        doc_options = {f"{d['document_name']} ({d['document_id']})": d["document_id"] for d in docs}
        for doc in docs:
            col1, col2, col3 = st.columns([3, 1, 1])
            col1.write(f"📄 **{doc['document_name']}**")
            col2.write(f"Type: {doc['document_type']}")
            col3.write(f"Chunks: {doc['chunk_count']}")
        st.divider()
        st.subheader("Delete Document")
        delete_doc_label = st.selectbox("Select document to delete", list(doc_options.keys()), key="delete_doc_select")
        delete_confirm = st.checkbox("Confirm permanent deletion from knowledge base", key="delete_doc_confirm")
        if st.button("🗑️ Delete Document"):
            if not delete_confirm:
                st.warning("Please confirm deletion before proceeding.")
            else:
                delete_doc_id = doc_options[delete_doc_label]
                try:
                    st.session_state.qa_agent.store.delete_document(delete_doc_id)
                    st.success(f"Deleted document '{delete_doc_label}' and all associated chunks.")
                    st.session_state.chat_history = []
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to delete document: {e}")
        st.divider()
        st.subheader("Inspect Stored Chunks")
        selected_doc_label = st.selectbox("Select document", list(doc_options.keys()))
        selected_doc_id = doc_options[selected_doc_label]
        sample_limit = st.slider("Sample chunk count", min_value=1, max_value=20, value=5)
        if st.button("🔎 Show Raw Chunk Samples"):
            samples = st.session_state.qa_agent.store.get_document_chunks(selected_doc_id, limit=sample_limit)
            if not samples:
                st.warning("No chunk samples found.")
            for i, s in enumerate(samples, 1):
                st.markdown(f"**Chunk {i}**")
                page_val = s['metadata'].get('page_number') or s['metadata'].get('page_no')
                st.caption(
                    f"Section: {s['metadata'].get('section_title', 'N/A')} | "
                    f"Page: {page_val or 'N/A'} | "
                    f"Sub-chunk: {s['metadata'].get('sub_chunk_index', 'N/A')} | "
                    f"Index: {s['metadata'].get('target_index', 'N/A')}"
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
                with st.spinner(f"Processing {doc_name} with dual-index strategy..."):
                    result = ingest_document_dual_index(
                        pdf_path=tmp_path,
                        document_id=doc_id,
                        document_name=doc_name,
                        document_type=doc_type,
                        use_local_db=True,
                    )
                st.success(f"Successfully ingested '{doc_name}' - Section: {result['section_count']}, Rule: {result['rule_count']}")
                st.rerun()
            except Exception as e:
                st.error(f"Ingestion failed: {e}")
                import traceback
                with st.expander("Error details"):
                    st.code(traceback.format_exc())
            finally:
                os.unlink(tmp_path)

elif mode == "⚙️ LLM Settings":
    st.header("⚙️ LLM Provider Configuration")
    current_cfg = get_runtime_llm_config()
    default_provider = (current_cfg.get("provider") or "openai").lower()
    provider_label = st.radio(
        "Provider Selection",
        ["OpenAI", "Google"],
        index=0 if default_provider == "openai" else 1,
        horizontal=True,
    )
    provider = provider_label.lower()
    if st.session_state.get("llm_provider_ui") != provider:
        st.session_state.llm_provider_ui = provider
        st.session_state.llm_models = []
        st.session_state.llm_connected = False
    api_key = st.text_input(
        "API Key",
        type="password",
        value=current_cfg.get("api_key", ""),
        placeholder="Enter provider API key",
    )
    base_url = ""
    if provider == "openai":
        base_url = st.text_input(
            "Base URL (OpenAI Only)",
            value=current_cfg.get("base_url", "https://api.openai.com/v1") or "https://api.openai.com/v1",
            help="Use custom URL for compatible providers (e.g. OpenRouter).",
        )
    selected_model = st.selectbox(
        "Model Selection",
        options=st.session_state.llm_models if st.session_state.llm_models else ["No models loaded"],
        index=0,
    )
    if st.button("Save / Connect", type="primary"):
        if not api_key.strip():
            st.error("Please enter a valid API key.")
        else:
            try:
                if provider == "openai":
                    models = list_openai_models(api_key=api_key.strip(), base_url=base_url.strip() or "https://api.openai.com/v1")
                else:
                    models = list_google_models(api_key=api_key.strip())
                if not models:
                    st.error("No usable generative models found for this provider.")
                else:
                    st.session_state.llm_models = models
                    model_to_use = models[0]
                    set_runtime_llm_config(
                        provider=provider,
                        api_key=api_key.strip(),
                        base_url=base_url.strip(),
                        model=model_to_use,
                    )
                    st.session_state.llm_connected = True
                    st.success(f"Connected successfully. Loaded {len(models)} models.")
                    st.rerun()
            except Exception as e:
                st.session_state.llm_connected = False
                st.error(f"Connection failed: {e}")
    if st.session_state.llm_connected:
        st.info("Connection active. Select a model and click Apply Model.")
        chosen_model = st.selectbox("Available Models", options=st.session_state.llm_models, key="llm_active_model")
        if st.button("Apply Model"):
            cfg = get_runtime_llm_config()
            set_runtime_llm_config(
                provider=cfg.get("provider") or provider,
                api_key=cfg.get("api_key") or api_key.strip(),
                base_url=cfg.get("base_url") or base_url.strip(),
                model=chosen_model,
            )
            st.success(f"Model set to {chosen_model}.")
