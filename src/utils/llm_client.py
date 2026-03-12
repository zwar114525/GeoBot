import time
from openai import OpenAI
from loguru import logger
from config.settings import LLM_API_KEY, LLM_BASE_URL, LLM_PROVIDER, PRIMARY_MODEL
from src.utils.cache import get_llm_cache

_RUNTIME_LLM_CONFIG: dict = {}
_CLIENT_CACHE: dict = {}
_LLM_CACHE = None


def get_llm_cache():
    global _LLM_CACHE
    if _LLM_CACHE is None:
        from src.utils.cache import get_llm_cache as _get_cache
        _LLM_CACHE = _get_cache()
    return _LLM_CACHE


def set_runtime_llm_config(
    provider: str,
    api_key: str,
    base_url: str | None = None,
    model: str | None = None,
):
    _RUNTIME_LLM_CONFIG["provider"] = (provider or "").strip().lower()
    _RUNTIME_LLM_CONFIG["api_key"] = (api_key or "").strip()
    _RUNTIME_LLM_CONFIG["base_url"] = (base_url or "").strip()
    _RUNTIME_LLM_CONFIG["model"] = (model or "").strip()


def get_runtime_llm_config() -> dict:
    return dict(_RUNTIME_LLM_CONFIG)


def _resolve_config() -> dict:
    provider = (_RUNTIME_LLM_CONFIG.get("provider") or LLM_PROVIDER or "openai").lower()
    api_key = _RUNTIME_LLM_CONFIG.get("api_key") or LLM_API_KEY
    base_url = _RUNTIME_LLM_CONFIG.get("base_url") or LLM_BASE_URL
    model = _RUNTIME_LLM_CONFIG.get("model") or PRIMARY_MODEL
    return {"provider": provider, "api_key": api_key, "base_url": base_url, "model": model}


def _openai_client(base_url: str, api_key: str) -> OpenAI:
    key = f"{base_url}||{api_key}"
    client = _CLIENT_CACHE.get(key)
    if client is None:
        client = OpenAI(base_url=base_url, api_key=api_key)
        _CLIENT_CACHE[key] = client
    return client


def _openrouter_headers(base_url: str) -> dict:
    if "openrouter.ai" in (base_url or "").lower():
        return {
            "HTTP-Referer": "https://geotech-agent.local",
            "X-Title": "Geotech AI Agent",
        }
    return {}


def list_openai_models(api_key: str, base_url: str) -> list[str]:
    client = OpenAI(base_url=base_url, api_key=api_key)
    response = client.models.list()
    model_ids = sorted([m.id for m in response.data if getattr(m, "id", None)])
    return model_ids


def list_google_models(api_key: str) -> list[str]:
    try:
        import google.generativeai as genai
    except Exception as e:
        raise RuntimeError("google-generativeai SDK is not installed") from e
    genai.configure(api_key=api_key)
    models = []
    for m in genai.list_models():
        name = getattr(m, "name", "")
        methods = getattr(m, "supported_generation_methods", []) or []
        if "generateContent" not in methods:
            continue
        short = name.split("/", 1)[1] if name.startswith("models/") else name
        if "embedding" in short.lower():
            continue
        models.append(short)
    return sorted(list(dict.fromkeys(models)))


def call_llm(
    prompt: str,
    system_prompt: str = "You are a senior geotechnical engineer in Hong Kong.",
    model: str = PRIMARY_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 4000,
    retries: int = 3,
    use_cache: bool = True,
) -> str:
    """
    Call LLM with optional caching.
    
    Args:
        prompt: User prompt
        system_prompt: System prompt
        model: Model to use
        temperature: Temperature
        max_tokens: Max tokens
        retries: Number of retries
        use_cache: Whether to use cache (default True)
        
    Returns:
        LLM response text
    """
    # Check cache first
    if use_cache:
        cache = get_llm_cache()
        cached = cache.get(prompt, system_prompt, model)
        if cached is not None:
            logger.debug(f"LLM cache hit for prompt: {prompt[:50]}...")
            return cached
    
    resolved = _resolve_config()
    model_to_use = resolved["model"] or model
    base_url = resolved["base_url"]
    api_key = resolved["api_key"]
    client = _openai_client(base_url=base_url, api_key=api_key)
    headers = _openrouter_headers(base_url)
    
    for attempt in range(retries):
        try:
            request_kwargs = {}
            if headers:
                request_kwargs["extra_headers"] = headers
            response = client.chat.completions.create(
                model=model_to_use,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                **request_kwargs,
            )
            result = response.choices[0].message.content or ""
            
            # Cache the result
            if use_cache:
                cache = get_llm_cache()
                cache.set(prompt, result, system_prompt, model)
            
            return result
            
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
