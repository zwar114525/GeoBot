import sys
from pathlib import Path
import re

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.vectordb.qdrant_store import GeoVectorStore
from src.utils.llm_client import call_llm_with_context
from config.settings import PRIMARY_MODEL


class QAAgent:
    def __init__(self, use_local_db: bool = False):
        self.store = GeoVectorStore(use_local=use_local_db)
        self.conversation_history: list[dict] = []

    def ask(
        self,
        question: str,
        document_type: str | None = None,
        document_id: str | None = None,
        top_k: int = 8,
        model: str = PRIMARY_MODEL,
        equation_mode: bool = False,
    ) -> dict:
        chunks = self.store.search(
            query=question,
            top_k=top_k,
            document_type=document_type,
            document_id=document_id,
        )
        if equation_mode:
            equation_query = (
                f"{question} equation formula design check criterion "
                "Mpl Rd Ft Rd Fb Rd V Ed V Rd resistance capacity"
            )
            eq_chunks = self.store.search(
                query=equation_query,
                top_k=max(top_k, 12),
                document_type=document_type,
                document_id=document_id,
                score_threshold=0.0,
            )
            chunks = self._merge_chunks(chunks, eq_chunks, limit=max(top_k, 12))
        if not chunks:
            return {
                "answer": "No relevant information found in the knowledge base.",
                "sources": [],
                "query": question,
            }
        try:
            answer = call_llm_with_context(
                question=question,
                context_chunks=chunks,
                model=model,
                equation_mode=equation_mode,
            )
            if equation_mode:
                answer = self._normalize_latex_blocks(answer)
        except Exception as e:
            preview = "\n\n".join([c["text"][:350] for c in chunks[:3]])
            answer = (
                "Retrieved relevant context, but LLM response generation failed. "
                f"Error: {e}\n\n"
                "Top retrieved excerpts:\n"
                f"{preview}"
            )
        sources = []
        seen = set()
        for chunk in chunks:
            source_key = (chunk["metadata"].get("document_name", ""), chunk["metadata"].get("section_title", ""))
            if source_key not in seen:
                seen.add(source_key)
                sources.append(
                    {
                        "document": chunk["metadata"].get("document_name", "Unknown"),
                        "section": chunk["metadata"].get("section_title", "N/A"),
                        "page": chunk["metadata"].get("page_number", "N/A"),
                        "relevance_score": round(chunk["score"], 3),
                    }
                )
        self.conversation_history.append({"question": question, "answer": answer, "sources": sources})
        return {"answer": answer, "sources": sources, "query": question}

    def _normalize_latex_blocks(self, text: str) -> str:
        if not text:
            return text
        text = text.replace("<br>", "\n")
        lines = text.splitlines()
        normalized = []
        inline_pattern = re.compile(r"^\s*\[(.+)\]\s*$")
        step_pattern = re.compile(r"^\s*(###\s*Step|\d+\)\s*Step)")
        skip_var_section_until_next_heading = False
        no_equation_in_step = False
        for line in lines:
            stripped = line.strip()
            if step_pattern.match(stripped):
                no_equation_in_step = False
                skip_var_section_until_next_heading = False
                normalized.append(line)
                continue
            if skip_var_section_until_next_heading:
                if (
                    stripped.lower().startswith("**acceptance")
                    or stripped.lower().startswith("**citation")
                    or step_pattern.match(stripped)
                ):
                    skip_var_section_until_next_heading = False
                else:
                    continue
            if stripped.startswith("[Source:"):
                normalized.append(line)
                continue
            if "Equation not found in retrieved context" in stripped:
                no_equation_in_step = True
                normalized.append(line)
                continue
            if stripped.lower().startswith("**variable definitions") and no_equation_in_step:
                normalized.append(line)
                normalized.append("N/A (no equation in retrieved context).")
                skip_var_section_until_next_heading = True
                continue
            match = inline_pattern.match(stripped)
            if match:
                inner = match.group(1).strip()
                if any(token in inner for token in ("\\frac", "\\sum", "\\min", "_", "^", "=")):
                    normalized.append(f"$${inner}$$")
                    continue
            bullet_var = re.match(r"^(\s*[-*]\s*)\(([^)]+)\)\s*[-:]\s*(.*)$", line)
            if bullet_var:
                prefix, var_name, desc = bullet_var.groups()
                normalized.append(f"{prefix}${var_name.strip()}$: {desc.strip()}")
                continue
            normalized.append(line)
        normalized_text = "\n".join(normalized)
        normalized_text = normalized_text.replace("\\[", "$$").replace("\\]", "$$")
        return normalized_text

    def _merge_chunks(self, base_chunks: list[dict], extra_chunks: list[dict], limit: int) -> list[dict]:
        merged = []
        seen = set()
        for chunk in base_chunks + extra_chunks:
            key = (
                chunk.get("metadata", {}).get("document_name", ""),
                chunk.get("metadata", {}).get("section_title", ""),
                chunk.get("metadata", {}).get("page_number", ""),
                chunk.get("text", "")[:160],
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(chunk)
            if len(merged) >= limit:
                break
        return merged

    def ask_followup(self, question: str, **kwargs) -> dict:
        if self.conversation_history:
            last = self.conversation_history[-1]
            enhanced_question = (
                f"Context from previous question: {last['question']}\n"
                f"Previous answer summary: {last['answer'][:500]}\n\n"
                f"Follow-up question: {question}"
            )
            return self.ask(enhanced_question, **kwargs)
        return self.ask(question, **kwargs)

    def list_knowledge_base(self) -> list[dict]:
        return self.store.list_documents()


def interactive_qa():
    agent = QAAgent(use_local_db=True)
    print("\nGEOTECH AI Q&A AGENT\nType your question, or 'quit' to exit\n")
    while True:
        question = input("Your question: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if question.lower() == "docs":
            docs = agent.list_knowledge_base()
            for doc in docs:
                print(f"{doc['document_name']} ({doc['document_type']}) - {doc['chunk_count']} chunks")
            continue
        if not question:
            continue
        result = agent.ask(question)
        print(f"\nAnswer:\n{result['answer']}\n")
        for s in result["sources"][:5]:
            print(f"- {s['document']} | {s['section']} ({s['relevance_score']})")


if __name__ == "__main__":
    interactive_qa()
