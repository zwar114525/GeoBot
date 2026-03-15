import sys
from pathlib import Path
import tempfile
import os

sys.path.insert(0, str(Path(__file__).parent))

# ─── Startup validation (run on every device before app loads) ───
_PIP_DEPS = [
    ("streamlit", "streamlit"),
    ("qdrant_client", "qdrant-client"),
    ("loguru", "loguru"),
    ("sentence_transformers", "sentence-transformers"),
]
# Project modules that must exist (avoid deep imports to keep errors clear)
_PROJECT_MODULES = [
    "src.ingestion.ingest",
    "src.utils.llm_client",
    "src.utils.session_persistence",
]


def _ensure_ready():
    # 1. Check pip packages
    missing_pip = []
    for module, package in _PIP_DEPS:
        try:
            __import__(module)
        except ModuleNotFoundError:
            missing_pip.append(package)
    if missing_pip:
        cmd = "pip install -r requirements.txt"
        print("\n" + "=" * 60, file=sys.stderr)
        print("  MISSING DEPENDENCIES", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        print(f"\n  Missing: {', '.join(missing_pip)}", file=sys.stderr)
        print("\n  Run:  pip install -r requirements.txt", file=sys.stderr)
        print("  Or:   setup.bat (Windows) / ./setup.sh (Mac/Linux)", file=sys.stderr)
        print("=" * 60 + "\n", file=sys.stderr)
        sys.exit(1)

    # 2. Check project modules (catches incomplete repo / wrong structure)
    for mod in _PROJECT_MODULES:
        try:
            __import__(mod)
        except ModuleNotFoundError as e:
            print("\n" + "=" * 60, file=sys.stderr)
            print("  MISSING PROJECT MODULE", file=sys.stderr)
            print("=" * 60, file=sys.stderr)
            print(f"\n  Module: {mod}", file=sys.stderr)
            print(f"  Error: {e}", file=sys.stderr)
            print("\n  Ensure the repo is fully cloned and run from project root.", file=sys.stderr)
            print("  See SETUP.md for setup steps.", file=sys.stderr)
            print("=" * 60 + "\n", file=sys.stderr)
            sys.exit(1)


_ensure_ready()

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.agents.qa_agent import QAAgent
from src.agents.designer_agent import DesignerAgent
from src.agents.validator_agent import ValidatorAgent
from src.ingestion.ingest import ingest_document_dual_index
from src.programme.xer_parser import XERParser
from src.programme.chart_generator import ChartGenerator
from src.programme.programme_agent import ProgrammeAgent
from src.programme.evm_calculator import EVMCalculator
from src.utils.llm_client import (
    set_runtime_llm_config,
    get_runtime_llm_config,
    list_openai_models,
    list_google_models,
)
from src.utils.session_persistence import save_session, load_session, list_sessions, delete_session

st.set_page_config(page_title="Geotech AI Agent", page_icon="🏗️", layout="wide", initial_sidebar_state="expanded")

# Lightweight session state first
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "designer_agent" not in st.session_state:
    st.session_state.designer_agent = None
if "designer_messages" not in st.session_state:
    st.session_state.designer_messages = []
if "llm_models" not in st.session_state:
    st.session_state.llm_models = []
if "llm_connected" not in st.session_state:
    st.session_state.llm_connected = False
if "session_id" not in st.session_state:
    st.session_state.session_id = "default"
if "programme_agent" not in st.session_state:
    st.session_state.programme_agent = None
if "programme_data" not in st.session_state:
    st.session_state.programme_data = None
if "programme_chat_history" not in st.session_state:
    st.session_state.programme_chat_history = []

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
        ["📚 Knowledge Q&A", "📝 Report Generator", "✅ Submission Checker", "📂 Document Manager", "📅 Programme Manager", "⚙️ LLM Settings"],
        index=0,
    )

def _get_qa_agent():
    """Lazy-load QAAgent only when needed (avoids Qdrant lock when using Programme Manager)."""
    if "qa_agent" not in st.session_state or st.session_state.qa_agent is None:
        with st.spinner("Loading AI agents (embedding model)... First run may take 1–2 minutes."):
            st.session_state.qa_agent = QAAgent(use_local_db=True)
    return st.session_state.qa_agent

def _get_validator_agent():
    """Lazy-load ValidatorAgent only when needed."""
    if "validator_agent" not in st.session_state or st.session_state.validator_agent is None:
        with st.spinner("Loading Validator agent..."):
            st.session_state.validator_agent = ValidatorAgent(use_local_db=True)
    return st.session_state.validator_agent

if mode == "📚 Knowledge Q&A":
    col_title, col_clear = st.columns([1, 0.15])
    with col_title:
        st.header("📚 Knowledge Base Q&A")
    with col_clear:
        st.write("")
        st.write("")
        if st.button("🗑️ Clear", help="Clear chat history and conversation context so the next question is answered independently"):
            st.session_state.chat_history = []
            if st.session_state.qa_agent:
                st.session_state.qa_agent.conversation_history = []
            save_session(
                st.session_state.session_id,
                st.session_state.chat_history,
                st.session_state.designer_messages,
            )
            st.rerun()
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
    uploaded = st.file_uploader("Upload PDF documents", type=["pdf"], accept_multiple_files=True, key="ingest_upload")
    if uploaded:
        files = uploaded if isinstance(uploaded, list) else [uploaded]
        st.write(f"**{len(files)} file(s) selected**")
        col1, col2 = st.columns(2)
        with col1:
            doc_type = st.selectbox("Document Type", ["code", "manual", "guideline", "report", "drawing"])
        with col2:
            st.write("")  # alignment
            st.write("")
            if st.button("📥 Ingest All Documents", type="primary"):
                results = []
                progress_bar = st.progress(0)
                for i, f in enumerate(files):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(f.getbuffer())
                        tmp_path = tmp.name
                    try:
                        doc_name = f.name.replace(".pdf", "")
                        doc_id = doc_name.lower().replace(" ", "_")
                        with st.spinner(f"Processing {f.name} (Docling + dual-index)..."):
                            result = ingest_document_dual_index(
                                pdf_path=tmp_path,
                                document_id=doc_id,
                                document_name=doc_name,
                                document_type=doc_type,
                                use_local_db=True,
                            )
                        total = result["section_count"] + result["rule_count"]
                        results.append({"file": f.name, "result": result, "total": total, "success": True})
                    except Exception as e:
                        results.append({"file": f.name, "error": str(e), "success": False})
                    finally:
                        os.unlink(tmp_path)
                    progress_bar.progress((i + 1) / len(files))
                # Show results
                success_count = sum(1 for r in results if r["success"])
                st.success(f"Successfully ingested {success_count}/{len(files)} documents")
                for r in results:
                    if r["success"]:
                        res = r["result"]
                        st.write(f"  - {r['file']}: Section {res['section_count']}, Rule {res['rule_count']} (chunked → chunked_files/)")
                    else:
                        st.error(f"  - {r['file']}: {r['error']}")
                if success_count > 0:
                    st.rerun()

elif mode == "📅 Programme Manager":
    st.header("📅 Construction Programme Manager")

    col1, col2 = st.columns([2, 1])
    with col1:
        xer_file = st.file_uploader("Upload Primavera P6 XER Schedule File", type=["xer"])
    with col2:
        st.write("")
        st.write("")
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("Load Demo", use_container_width=True, key="load_demo"):
                st.session_state.programme_data = XERParser(use_simulated=True)
                st.session_state.programme_agent = ProgrammeAgent()
                st.session_state.programme_agent.load_schedule(
                    st.session_state.programme_data.tasks,
                    st.session_state.programme_data.resources,
                    st.session_state.programme_data.relationships
                )
                st.session_state.programme_chat_history = []
                st.rerun()
        with col_btn2:
            if st.button("Refresh", use_container_width=True, key="refresh_data"):
                if st.session_state.programme_data is not None:
                    st.session_state.programme_data = XERParser(use_simulated=True)
                    st.session_state.programme_agent = ProgrammeAgent()
                    st.session_state.programme_agent.load_schedule(
                        st.session_state.programme_data.tasks,
                        st.session_state.programme_data.resources,
                        st.session_state.programme_data.relationships
                    )
                    st.session_state.programme_chat_history = []
                    st.rerun()

    if xer_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xer") as tmp:
            tmp.write(xer_file.getbuffer())
            tmp_path = tmp.name
        try:
            with st.spinner("Parsing XER file..."):
                st.session_state.programme_data = XERParser(file_path=tmp_path)
                st.session_state.programme_agent = ProgrammeAgent()
                st.session_state.programme_agent.load_schedule(
                    st.session_state.programme_data.tasks,
                    st.session_state.programme_data.resources,
                    st.session_state.programme_data.relationships
                )
                st.session_state.programme_chat_history = []
            st.success(f"Loaded {len(st.session_state.programme_data.tasks)} tasks from XER file")
        except Exception as e:
            st.error(f"Error parsing XER file: {e}")
        finally:
            os.unlink(tmp_path)

    if st.session_state.programme_data is not None:
        summary = st.session_state.programme_data.get_schedule_summary()

        st.subheader("Schedule Overview")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Tasks", summary.get("total_tasks", 0))
        col2.metric("Completed", summary.get("completed", 0))
        col3.metric("In Progress", summary.get("in_progress", 0))
        col4.metric("Critical Tasks", summary.get("critical_tasks", 0))
        col5.metric("Budget", f"${summary.get('total_budget', 0):,.0f}")

        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
            "📊 Gantt Chart", "👷 Resources", "💰 Budget", "📈 EVM",
            "📉 S-Curve", "📋 Activity Table", "📑 Reports", "💬 Q&A"
        ])

        with tab1:
            chart_gen = ChartGenerator()

            # P6 Professional Gantt Chart - Table + Timeline layout
            gantt_fig = chart_gen.create_gantt_chart_professional(
                st.session_state.programme_data.tasks,
                relationships_df=st.session_state.programme_data.relationships
            )

            st.plotly_chart(gantt_fig, use_container_width=True)

        with tab2:
            res_fig = chart_gen.create_resource_chart(st.session_state.programme_data.tasks)
            st.plotly_chart(res_fig, use_container_width=True)
            progress_fig = chart_gen.create_progress_pie(st.session_state.programme_data.tasks)
            st.plotly_chart(progress_fig, use_container_width=True)

        with tab3:
            budget_fig = chart_gen.create_budget_chart(st.session_state.programme_data.tasks)
            st.plotly_chart(budget_fig, use_container_width=True)

        with tab4:
            st.subheader("Earned Value Management Dashboard")

            evm_calc = EVMCalculator(st.session_state.programme_data.tasks)

            st.markdown(evm_calc.get_evm_summary_html(), unsafe_allow_html=True)

            evm_fig = evm_calc.create_evm_dashboard()
            st.plotly_chart(evm_fig, use_container_width=True)

            evm_trend_fig = evm_calc.create_evm_trend_chart()
            st.plotly_chart(evm_trend_fig, use_container_width=True)

        with tab5:
            st.subheader("S-Curve: Planned Value vs Earned Value")

            s_curve_fig = chart_gen.create_s_curve(st.session_state.programme_data.tasks)
            st.plotly_chart(s_curve_fig, use_container_width=True)

        with tab6:
            st.subheader("Activity Table")

            tasks_df = st.session_state.programme_data.tasks.copy()
            tasks_df['start_date'] = pd.to_datetime(tasks_df['start_date']).dt.strftime('%Y-%m-%d')
            tasks_df['end_date'] = pd.to_datetime(tasks_df['end_date']).dt.strftime('%Y-%m-%d')
            tasks_df['duration'] = (pd.to_datetime(tasks_df['end_date']) - pd.to_datetime(tasks_df['start_date'])).dt.days
            tasks_df['critical'] = tasks_df['critical'].apply(lambda x: 'Yes' if x else 'No')

            display_cols = ['task_id', 'task_name', 'start_date', 'end_date', 'duration',
                            'percent_complete', 'total_float', 'critical']
            display_df = tasks_df[display_cols].copy()
            display_df.columns = ['Task ID', 'Task Name', 'Start', 'Finish', 'Duration',
                                  '% Complete', 'Total Float', 'Critical']

            def highlight_critical(row):
                return ['background-color: #fee2e2' if row['Critical'] == 'Yes' else '' for _ in row]

            st.dataframe(
                display_df.style.apply(highlight_critical, axis=1),
                use_container_width=True,
                hide_index=True,
                height=500
            )

            csv = display_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Export to CSV",
                data=csv,
                file_name="programme_activities.csv",
                mime="text/csv",
                use_container_width=True
            )

        with tab7:
            st.subheader("Reports")

            report_type = st.selectbox(
                "Select Report Type",
                ["4-Week Look-Ahead", "Subcontractor Coordination", "Weather Risk Analysis", "Delay Claim Analysis", "Delay Risk Prediction"]
            )

            if st.button("Generate Report"):
                with st.spinner("Generating report..."):
                    if report_type == "4-Week Look-Ahead":
                        report = st.session_state.programme_agent.generate_lookahead_report(weeks=4)
                    elif report_type == "Subcontractor Coordination":
                        report = st.session_state.programme_agent.generate_subcontractor_coordination_report()
                    elif report_type == "Weather Risk Analysis":
                        from src.programme.weather_risk import WeatherRiskAnalyzer
                        weather = WeatherRiskAnalyzer(st.session_state.programme_data.tasks)
                        report = weather.generate_risk_report(days_ahead=60)
                    elif report_type == "Delay Risk Prediction":
                        from src.programme.delay_predictor import DelayPredictor
                        predictor = DelayPredictor(st.session_state.programme_data.tasks)
                        report = predictor.generate_risk_report(top_n=15)
                    elif report_type == "Delay Claim Analysis":
                        st.info("Enter delay event details below:")
                        col1, col2 = st.columns(2)
                        with col1:
                            event_name = st.text_input("Event Name", "Owner Change Order #1")
                            cause = st.selectbox("Cause", ["Owner-initiated design change", "Force Majeure - Weather", "Contractor delay", "Permit delay", "Utility delay"])
                        with col2:
                            start_date = st.date_input("Start Date", pd.Timestamp.now())
                            end_date = st.date_input("End Date", pd.Timestamp.now() + pd.Timedelta(days=5))

                        task_ids = st.session_state.programme_data.tasks['task_id'].tolist()[:10]
                        selected_tasks = st.multiselect("Affected Activities", task_ids, default=task_ids[:3])

                        if st.button("Analyze Delay Claim"):
                            from src.programme.delay_claim import DelayClaimAnalyzer
                            analyzer = DelayClaimAnalyzer(st.session_state.programme_data.tasks)
                            delay_event = {
                                'event_name': event_name,
                                'cause': cause,
                                'start_date': str(start_date),
                                'end_date': str(end_date),
                                'affected_activities': selected_tasks
                            }
                            report = analyzer.create_delay_claim_report(delay_event)
                            st.text_area("Delay Claim Report", report, height=400)
                        else:
                            report = "Fill in delay event details and click 'Analyze Delay Claim'"
                    else:
                        report = "Select a report type."
                    if report_type != "Delay Claim Analysis":
                        st.text_area("Report Output", report, height=400)

        with tab6:
            st.subheader("Ask about your construction schedule")

            for msg in st.session_state.programme_chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            if prompt := st.chat_input("Ask about tasks, progress, critical path, delays..."):
                st.session_state.programme_chat_history.append({"role": "user", "content": prompt})
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing schedule..."):
                        result = st.session_state.programme_agent.ask(prompt)
                    st.markdown(result["answer"])
                st.session_state.programme_chat_history.append({"role": "assistant", "content": result["answer"]})
    else:
        st.info("Upload an XER file or click 'Load Demo Data' to get started.")

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
