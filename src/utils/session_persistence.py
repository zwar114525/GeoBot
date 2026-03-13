"""
Session persistence for GeoBot.
Saves and loads chat history and session state to/from JSON files.
"""
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any
from loguru import logger


# Default session storage path (relative to project root / cwd)
SESSION_DIR = Path("./logs/sessions")
SESSION_DIR.mkdir(parents=True, exist_ok=True)


def get_session_path(session_id: str = "default") -> Path:
    """Get the path for a session file."""
    return SESSION_DIR / f"session_{session_id}.json"


def save_session(
    session_id: str,
    chat_history: List[Dict[str, Any]],
    designer_messages: Optional[List[Dict[str, Any]]] = None,
    llm_config: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Save session state to file.

    Args:
        session_id: Unique session identifier
        chat_history: Q&A chat history
        designer_messages: Report generator messages
        llm_config: LLM configuration
        metadata: Additional metadata

    Returns:
        True if successful
    """
    try:
        session_path = get_session_path(session_id)
        data = {
            "session_id": session_id,
            "saved_at": datetime.now().isoformat(),
            "chat_history": chat_history or [],
            "designer_messages": designer_messages or [],
            "llm_config": llm_config or {},
            "metadata": metadata or {},
        }
        with open(session_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Session saved: {session_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to save session: {e}")
        return False


def load_session(session_id: str) -> Optional[Dict[str, Any]]:
    """
    Load session state from file.

    Args:
        session_id: Unique session identifier

    Returns:
        Session data dict or None if not found
    """
    try:
        session_path = get_session_path(session_id)
        if not session_path.exists():
            return None
        with open(session_path, encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Session loaded: {session_id}")
        return data
    except Exception as e:
        logger.error(f"Failed to load session: {e}")
        return None


def list_sessions() -> List[Dict[str, Any]]:
    """
    List all saved sessions.

    Returns:
        List of session info dicts
    """
    sessions = []
    try:
        for session_file in SESSION_DIR.glob("session_*.json"):
            try:
                with open(session_file, encoding="utf-8") as f:
                    data = json.load(f)
                sessions.append({
                    "session_id": data.get("session_id", session_file.stem.replace("session_", "")),
                    "saved_at": data.get("saved_at", ""),
                    "chat_count": len(data.get("chat_history", [])),
                    "designer_count": len(data.get("designer_messages", [])),
                })
            except Exception:
                continue
        sessions.sort(key=lambda x: x.get("saved_at", ""), reverse=True)
    except Exception as e:
        logger.error(f"Failed to list sessions: {e}")
    return sessions


def delete_session(session_id: str) -> bool:
    """
    Delete a saved session.

    Args:
        session_id: Unique session identifier

    Returns:
        True if successful
    """
    try:
        session_path = get_session_path(session_id)
        if session_path.exists():
            session_path.unlink()
            logger.info(f"Session deleted: {session_id}")
            return True
        return False
    except Exception as e:
        logger.error(f"Failed to delete session: {e}")
        return False


def auto_save_enabled() -> bool:
    """Check if auto-save is enabled via environment variable."""
    return os.getenv("GEOBOT_AUTO_SAVE", "true").lower() == "true"


def auto_save_session(
    session_id: str,
    chat_history: List[Dict[str, Any]],
    designer_messages: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Auto-save session if enabled."""
    if auto_save_enabled():
        save_session(session_id, chat_history, designer_messages)
