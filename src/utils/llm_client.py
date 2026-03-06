import time
from openai import OpenAI
from loguru import logger
from config.settings import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, PRIMARY_MODEL

client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY)


def call_llm(
    prompt: str,
    system_prompt: str = "You are a senior geotechnical engineer in Hong Kong.",
    model: str = PRIMARY_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 4000,
    retries: int = 3,
) -> str:
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                extra_headers={
                    "HTTP-Referer": "https://geotech-agent.local",
                    "X-Title": "Geotech AI Agent",
                },
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.warning(f"LLM call attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(2**attempt)
            else:
                raise


def call_llm_with_context(
    question: str,
    context_chunks: list[dict],
    model: str = PRIMARY_MODEL,
    equation_mode: bool = False,
) -> str:
    context_text = "\n\n---\n\n".join(
        [
            f"[Source: {c['metadata'].get('document_name', 'Unknown')} | "
            f"Section: {c['metadata'].get('section_title', 'N/A')} | "
            f"Page: {c['metadata'].get('page_number', 'N/A')}]\n{c['text']}"
            for c in context_chunks
        ]
    )

    if equation_mode:
        system_prompt = (
            "You are a senior structural and geotechnical engineer.\n"
            "Use ONLY the provided references.\n"
            "For each design step, prioritize critical equations and checks.\n"
            "Never invent equations, clause numbers, or symbols not present in retrieved context.\n"
            "If an equation is not explicitly found in retrieved context, state: "
            "'Equation not found in retrieved context'."
        )
        user_prompt = (
            "Answer the question using the reference documents below.\n\n"
            f"REFERENCE DOCUMENTS:\n{context_text}\n\n"
            f"QUESTION: {question}\n\n"
            "Return markdown in this exact structure for each step (do not use markdown tables):\n"
            "### Step N: <title>\n"
            "**Critical equation(s)**\n"
            "$$\n<equation>\n$$\n"
            "or\n"
            "Equation not found in retrieved context\n"
            "**Variable definitions (units)**\n"
            "- $<symbol>$: <definition with units>\n"
            "**Acceptance/check criterion**\n"
            "- <criterion>\n"
            "**Citation**\n"
            "[Source: document name, Section/Clause: X.X.X, Page: N]\n\n"
            "Rules:\n"
            "- Include citations for every step.\n"
            "- If no equation exists in context for a step, write 'Equation not found in retrieved context'.\n"
            "- If no equation exists for a step, write under variable definitions: 'N/A (no equation in retrieved context)'.\n"
            "- Every equation must be in a display math block with its own lines: $$ equation $$.\n"
            "- Never use [ ... ] for equations.\n"
            "- Never use HTML tags such as <br>.\n"
            "- Do not escape dollar signs.\n"
            "Example format:\n"
            "$$\nP = \\frac{4 M_p}{m} - \\frac{Q_s}{s} + Q\n$$\n"
            "- Format citations as [Source: document name, Section/Clause: X.X.X, Page: N]."
        )
    else:
        system_prompt = (
            "You are a senior geotechnical engineer working in Hong Kong.\n"
            "You provide answers based ONLY on the provided reference documents.\n"
            "You cite specific clause numbers, section titles, and page numbers.\n"
            "If the provided context does not contain enough information to answer, "
            "say so explicitly. Never invent clause numbers or code requirements.\n"
            "Applicable codes framework: HK Code of Practice, Geoguides, BS codes, Eurocodes."
        )
        user_prompt = (
            "Based on the following reference documents, answer the question.\n\n"
            f"REFERENCE DOCUMENTS:\n{context_text}\n\n"
            f"QUESTION: {question}\n\n"
            "Provide your answer with specific citations to the source documents above.\n"
            "Format citations as [Source: document name, Section/Clause: X.X.X]."
        )
    return call_llm(user_prompt, system_prompt=system_prompt, model=model)
