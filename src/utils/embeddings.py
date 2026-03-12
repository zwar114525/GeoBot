from sentence_transformers import SentenceTransformer
from loguru import logger
from config.settings import EMBEDDING_MODEL
from src.utils.cache import get_embedding_cache

_model = None
_cache = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        _model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("Embedding model loaded successfully")
    return _model


def get_cache() -> "EmbeddingCache":
    global _cache
    if _cache is None:
        _cache = get_embedding_cache()
    return _cache


def embed_text(text: str, use_cache: bool = True) -> list[float]:
    """
    Embed a single text string with optional caching.
    
    Args:
        text: Text to embed
        use_cache: Whether to use cache (default True)
        
    Returns:
        Embedding vector
    """
    if use_cache:
        cache = get_cache()
        cached = cache.get(text)
        if cached is not None:
            return cached
    
    model = get_model()
    embedding = model.encode(text, normalize_embeddings=True)
    result = embedding.tolist()
    
    if use_cache:
        cache.set(text, result)
    
    return result


def embed_texts(texts: list[str], batch_size: int = 32, use_cache: bool = True) -> list[list[float]]:
    """
    Embed a batch of texts with optional caching.
    
    Args:
        texts: List of texts to embed
        batch_size: Batch size for encoding
        use_cache: Whether to use cache (default True)
        
    Returns:
        List of embedding vectors
    """
    if not use_cache:
        model = get_model()
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        return embeddings.tolist()
    
    # Check cache for each text
    cache = get_cache()
    results = []
    texts_to_compute = []
    indices_to_fill = []
    
    for i, text in enumerate(texts):
        cached = cache.get(text)
        if cached is not None:
            results.append(cached)
        else:
            texts_to_compute.append(text)
            indices_to_fill.append(i)
            results.append(None)  # Placeholder
    
    # Compute missing embeddings
    if texts_to_compute:
        logger.info(f"Computing {len(texts_to_compute)} new embeddings ({len(texts) - len(texts_to_compute)} from cache)")
        model = get_model()
        new_embeddings = model.encode(
            texts_to_compute,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        
        # Fill in results and cache
        for i, (idx, embedding) in enumerate(zip(indices_to_fill, new_embeddings)):
            result = embedding.tolist()
            results[idx] = result
            cache.set(texts_to_compute[i], result)
    
    return results


def embed_query(query: str, use_cache: bool = True) -> list[float]:
    """
    Embed a search query with optional caching.
    
    Args:
        query: Search query
        use_cache: Whether to use cache (default True)
        
    Returns:
        Embedding vector
    """
    model = get_model()
    instruction = "Represent this sentence for searching relevant passages: "
    full_query = instruction + query
    
    # Cache based on original query (not instruction-prefixed)
    if use_cache:
        cache = get_cache()
        cached = cache.get(query)
        if cached is not None:
            return cached
    
    embedding = model.encode(full_query, normalize_embeddings=True)
    result = embedding.tolist()
    
    if use_cache:
        cache.set(query, result)
    
    return result
