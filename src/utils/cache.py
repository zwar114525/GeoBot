"""
Caching system for embeddings and LLM responses.
Reduces API costs and improves response times.
"""
import hashlib
import json
import time
import sqlite3
from pathlib import Path
from typing import Any, Optional, Dict, List
from dataclasses import dataclass
from loguru import logger


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    created_at: float
    expires_at: float
    access_count: int = 0
    last_accessed: float = 0


class CacheDB:
    """SQLite-based cache with TTL support."""
    
    def __init__(self, cache_dir: str = "./cache", db_name: str = "cache.db"):
        """
        Initialize cache database.
        
        Args:
            cache_dir: Directory for cache files
            db_name: Database filename
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / db_name
        
        self._init_db()
        logger.info(f"Cache database initialized at {self.db_path}")
    
    def _init_db(self):
        """Initialize database schema."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL,
                access_count INTEGER DEFAULT 0,
                last_accessed REAL DEFAULT 0
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_expires ON cache(expires_at)
        """)
        
        conn.commit()
        conn.close()
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT value, expires_at FROM cache 
            WHERE key = ? AND expires_at > ?
        """, (key, time.time()))
        
        row = cursor.fetchone()
        
        if row:
            value, expires_at = row
            # Update access stats
            cursor.execute("""
                UPDATE cache 
                SET access_count = access_count + 1, last_accessed = ?
                WHERE key = ?
            """, (time.time(), key))
            conn.commit()
            conn.close()
            return json.loads(value)
        
        conn.close()
        return None
    
    def set(self, key: str, value: Any, ttl_seconds: int = 3600):
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache (must be JSON-serializable)
            ttl_seconds: Time to live in seconds
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        now = time.time()
        expires_at = now + ttl_seconds
        
        try:
            value_json = json.dumps(value)
        except (TypeError, ValueError) as e:
            logger.warning(f"Cannot cache non-serializable value: {e}")
            conn.close()
            return
        
        cursor.execute("""
            INSERT OR REPLACE INTO cache (key, value, created_at, expires_at, access_count, last_accessed)
            VALUES (?, ?, ?, ?, 0, ?)
        """, (key, value_json, now, expires_at, now))
        
        conn.commit()
        conn.close()
    
    def delete(self, key: str):
        """Delete key from cache."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("DELETE FROM cache WHERE key = ?", (key,))
        conn.commit()
        conn.close()
    
    def clear_expired(self) -> int:
        """
        Clear expired entries.
        
        Returns:
            Number of entries deleted
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM cache WHERE expires_at <= ?", (time.time(),))
        deleted = cursor.rowcount
        
        conn.commit()
        conn.close()
        
        if deleted > 0:
            logger.info(f"Cleared {deleted} expired cache entries")
        
        return deleted
    
    def clear_all(self):
        """Clear all cache entries."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("DELETE FROM cache")
        conn.commit()
        conn.close()
        logger.info("Cache cleared")
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM cache")
        total = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM cache WHERE expires_at <= ?", (time.time(),))
        expired = cursor.fetchone()[0]
        
        cursor.execute("SELECT SUM(access_count) FROM cache")
        total_accesses = cursor.fetchone()[0] or 0
        
        conn.close()
        
        return {
            "total_entries": total,
            "expired_entries": expired,
            "total_accesses": total_accesses,
            "hit_rate": "N/A",
        }


class EmbeddingCache:
    """Cache for text embeddings."""
    
    def __init__(self, cache_db: CacheDB = None, ttl_seconds: int = 86400 * 7):
        """
        Initialize embedding cache.
        
        Args:
            cache_db: CacheDB instance (creates default if None)
            ttl_seconds: Cache TTL (default 7 days)
        """
        self.db = cache_db or CacheDB()
        self.ttl = ttl_seconds
        self.prefix = "embedding:"
    
    def _make_key(self, text: str) -> str:
        """Create cache key from text."""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return f"{self.prefix}{text_hash}"
    
    def get(self, text: str) -> Optional[List[float]]:
        """
        Get cached embedding.
        
        Args:
            text: Text that was embedded
            
        Returns:
            Embedding vector or None
        """
        key = self._make_key(text)
        return self.db.get(key)
    
    def set(self, text: str, embedding: List[float]):
        """
        Cache embedding.
        
        Args:
            text: Original text
            embedding: Embedding vector
        """
        key = self._make_key(text)
        self.db.set(key, embedding, ttl_seconds=self.ttl)
    
    def get_or_compute(self, text: str, compute_fn) -> List[float]:
        """
        Get from cache or compute and cache.
        
        Args:
            text: Text to embed
            compute_fn: Function to compute embedding
            
        Returns:
            Embedding vector
        """
        cached = self.get(text)
        if cached is not None:
            logger.debug(f"Embedding cache hit for text: {text[:50]}...")
            return cached
        
        logger.debug(f"Embedding cache miss, computing...")
        embedding = compute_fn(text)
        self.set(text, embedding)
        return embedding


class LLMResponseCache:
    """Cache for LLM responses."""
    
    def __init__(self, cache_db: CacheDB = None, ttl_seconds: int = 86400 * 30):
        """
        Initialize LLM response cache.
        
        Args:
            cache_db: CacheDB instance
            ttl_seconds: Cache TTL (default 30 days)
        """
        self.db = cache_db or CacheDB()
        self.ttl = ttl_seconds
        self.prefix = "llm:"
    
    def _make_key(self, prompt: str, system_prompt: str = "", model: str = "") -> str:
        """Create cache key from LLM parameters."""
        key_data = f"{prompt}|{system_prompt}|{model}"
        key_hash = hashlib.sha256(key_data.encode()).hexdigest()
        return f"{self.prefix}{key_hash}"
    
    def get(self, prompt: str, system_prompt: str = "", model: str = "") -> Optional[str]:
        """
        Get cached LLM response.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt
            model: Model name
            
        Returns:
            Cached response or None
        """
        key = self._make_key(prompt, system_prompt, model)
        return self.db.get(key)
    
    def set(self, prompt: str, response: str, system_prompt: str = "", model: str = ""):
        """
        Cache LLM response.
        
        Args:
            prompt: User prompt
            response: LLM response
            system_prompt: System prompt
            model: Model name
        """
        key = self._make_key(prompt, system_prompt, model)
        self.db.set(key, response, ttl_seconds=self.ttl)
    
    def get_or_generate(self, prompt: str, generate_fn, system_prompt: str = "", model: str = "") -> str:
        """
        Get from cache or generate and cache.
        
        Args:
            prompt: User prompt
            generate_fn: Function to generate response
            system_prompt: System prompt
            model: Model name
            
        Returns:
            LLM response
        """
        cached = self.get(prompt, system_prompt, model)
        if cached is not None:
            logger.debug(f"LLM cache hit for prompt: {prompt[:50]}...")
            return cached
        
        logger.debug(f"LLM cache miss, generating...")
        response = generate_fn(prompt)
        self.set(prompt, response, system_prompt, model)
        return response


class SearchCache:
    """Cache for search results."""
    
    def __init__(self, cache_db: CacheDB = None, ttl_seconds: int = 3600):
        """
        Initialize search cache.
        
        Args:
            cache_db: CacheDB instance
            ttl_seconds: Cache TTL (default 1 hour)
        """
        self.db = cache_db or CacheDB()
        self.ttl = ttl_seconds
        self.prefix = "search:"
    
    def _make_key(self, query: str, filters: Dict = None) -> str:
        """Create cache key from search parameters."""
        filter_str = json.dumps(filters or {}, sort_keys=True)
        key_data = f"{query}|{filter_str}"
        key_hash = hashlib.md5(key_data.encode()).hexdigest()
        return f"{self.prefix}{key_hash}"
    
    def get(self, query: str, filters: Dict = None) -> Optional[List[Dict]]:
        """Get cached search results."""
        key = self._make_key(query, filters)
        return self.db.get(key)
    
    def set(self, query: str, results: List[Dict], filters: Dict = None):
        """Cache search results."""
        key = self._make_key(query, filters)
        self.db.set(key, results, ttl_seconds=self.ttl)


# Global cache instances
_embedding_cache: Optional[EmbeddingCache] = None
_llm_cache: Optional[LLMResponseCache] = None
_search_cache: Optional[SearchCache] = None


def get_embedding_cache() -> EmbeddingCache:
    """Get or create global embedding cache."""
    global _embedding_cache
    if _embedding_cache is None:
        _embedding_cache = EmbeddingCache()
    return _embedding_cache


def get_llm_cache() -> LLMResponseCache:
    """Get or create global LLM cache."""
    global _llm_cache
    if _llm_cache is None:
        _llm_cache = LLMResponseCache()
    return _llm_cache


def get_search_cache() -> SearchCache:
    """Get or create global search cache."""
    global _search_cache
    if _search_cache is None:
        _search_cache = SearchCache()
    return _search_cache


def clear_all_caches():
    """Clear all caches."""
    get_embedding_cache().db.clear_all()
    get_llm_cache().db.clear_all()
    get_search_cache().db.clear_all()
    logger.info("All caches cleared")


def get_cache_stats() -> Dict:
    """Get statistics for all caches."""
    return {
        "embeddings": get_embedding_cache().db.get_stats(),
        "llm": get_llm_cache().db.get_stats(),
        "search": get_search_cache().db.get_stats(),
    }


def cleanup_expired_cache():
    """Clean up expired cache entries."""
    get_embedding_cache().db.clear_expired()
    get_llm_cache().db.clear_expired()
    get_search_cache().db.clear_expired()
