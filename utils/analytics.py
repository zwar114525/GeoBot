"""
Monitoring and analytics dashboard for GeoBot.
Track usage metrics, retrieval quality, and user satisfaction.
"""
import json
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from loguru import logger


@dataclass
class UsageEvent:
    """Represents a usage event."""
    event_id: str
    event_type: str  # query, answer, document_ingest, report_generate, validation
    timestamp: str
    user_id: str = ""
    session_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    duration_ms: int = 0
    success: bool = True


@dataclass
class RetrievalMetric:
    """Retrieval quality metric."""
    query: str
    results_count: int
    avg_score: float
    max_score: float
    min_score: float
    has_relevant_result: bool = True
    user_feedback: Optional[int] = None  # 1-5 rating


class AnalyticsDB:
    """SQLite-based analytics database."""
    
    def __init__(self, db_path: str = "./logs/analytics.db"):
        """
        Initialize analytics database.
        
        Args:
            db_path: Path to database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Usage events table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS usage_events (
                event_id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                user_id TEXT,
                session_id TEXT,
                metadata TEXT,
                duration_ms INTEGER,
                success INTEGER
            )
        """)
        
        # Retrieval metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS retrieval_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                results_count INTEGER,
                avg_score REAL,
                max_score REAL,
                min_score REAL,
                has_relevant_result INTEGER,
                user_feedback INTEGER
            )
        """)
        
        # User feedback table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                event_id TEXT,
                query TEXT,
                rating INTEGER,
                feedback_text TEXT
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON usage_events(event_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON usage_events(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_retrieval_timestamp ON retrieval_metrics(timestamp)")
        
        conn.commit()
        conn.close()
    
    def log_event(self, event: UsageEvent):
        """Log a usage event."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO usage_events 
            (event_id, event_type, timestamp, user_id, session_id, metadata, duration_ms, success)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            event.event_id,
            event.event_type,
            event.timestamp,
            event.user_id,
            event.session_id,
            json.dumps(event.metadata),
            event.duration_ms,
            1 if event.success else 0,
        ))
        
        conn.commit()
        conn.close()
    
    def log_retrieval(self, metric: RetrievalMetric):
        """Log retrieval metric."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO retrieval_metrics
            (query, timestamp, results_count, avg_score, max_score, min_score, has_relevant_result, user_feedback)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            metric.query,
            datetime.now().isoformat(),
            metric.results_count,
            metric.avg_score,
            metric.max_score,
            metric.min_score,
            1 if metric.has_relevant_result else 0,
            metric.user_feedback,
        ))
        
        conn.commit()
        conn.close()
    
    def log_feedback(self, event_id: str, query: str, rating: int, feedback_text: str = ""):
        """Log user feedback."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO user_feedback (timestamp, event_id, query, rating, feedback_text)
            VALUES (?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            event_id,
            query,
            rating,
            feedback_text,
        ))
        
        conn.commit()
        conn.close()
    
    def get_usage_stats(self, days: int = 7) -> Dict:
        """Get usage statistics for the last N days."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        
        # Total events by type
        cursor.execute("""
            SELECT event_type, COUNT(*) as count, AVG(duration_ms) as avg_duration
            FROM usage_events
            WHERE timestamp > ?
            GROUP BY event_type
        """, (cutoff,))
        events_by_type = {row[0]: {"count": row[1], "avg_duration_ms": row[2]} for row in cursor.fetchall()}
        
        # Success rate
        cursor.execute("""
            SELECT 
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as success,
                COUNT(*) as total
            FROM usage_events
            WHERE timestamp > ?
        """, (cutoff,))
        row = cursor.fetchone()
        success_rate = (row[0] / row[1] * 100) if row[1] > 0 else 0
        
        # Unique users/sessions
        cursor.execute("""
            SELECT COUNT(DISTINCT user_id), COUNT(DISTINCT session_id)
            FROM usage_events
            WHERE timestamp > ?
        """, (cutoff,))
        row = cursor.fetchone()
        
        conn.close()
        
        return {
            "period_days": days,
            "events_by_type": events_by_type,
            "success_rate": success_rate,
            "unique_users": row[0],
            "unique_sessions": row[1],
        }
    
    def get_retrieval_stats(self, days: int = 7) -> Dict:
        """Get retrieval quality statistics."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        
        # Average scores
        cursor.execute("""
            SELECT AVG(avg_score), AVG(max_score), AVG(results_count)
            FROM retrieval_metrics
            WHERE timestamp > ?
        """, (cutoff,))
        row = cursor.fetchone()
        
        # Feedback distribution
        cursor.execute("""
            SELECT rating, COUNT(*) as count
            FROM retrieval_metrics
            WHERE timestamp > ? AND user_feedback IS NOT NULL
            GROUP BY rating
        """, (cutoff,))
        feedback_dist = {str(row[0]): row[1] for row in cursor.fetchall()}
        
        conn.close()
        
        return {
            "period_days": days,
            "avg_score": row[0] or 0,
            "avg_max_score": row[1] or 0,
            "avg_results": row[2] or 0,
            "feedback_distribution": feedback_dist,
        }
    
    def get_popular_queries(self, days: int = 7, limit: int = 20) -> List[Dict]:
        """Get most popular queries."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        
        cursor.execute("""
            SELECT query, COUNT(*) as count, AVG(avg_score) as avg_score
            FROM retrieval_metrics
            WHERE timestamp > ?
            GROUP BY query
            ORDER BY count DESC
            LIMIT ?
        """, (cutoff, limit))
        
        results = [{"query": row[0], "count": row[1], "avg_score": row[2]} for row in cursor.fetchall()]
        conn.close()
        
        return results
    
    def get_recent_feedback(self, limit: int = 20) -> List[Dict]:
        """Get recent user feedback."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT timestamp, event_id, query, rating, feedback_text
            FROM user_feedback
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))
        
        results = [
            {"timestamp": row[0], "event_id": row[1], "query": row[2], "rating": row[3], "feedback": row[4]}
            for row in cursor.fetchall()
        ]
        conn.close()
        
        return results


class GeoBotAnalytics:
    """Main analytics interface for GeoBot."""
    
    def __init__(self, db_path: str = "./logs/analytics.db"):
        """Initialize analytics."""
        self.db = AnalyticsDB(db_path)
        self._current_session = self._generate_session_id()
    
    def _generate_session_id(self) -> str:
        """Generate session ID."""
        import uuid
        return str(uuid.uuid4())[:8]
    
    def _generate_event_id(self) -> str:
        """Generate event ID."""
        import uuid
        return str(uuid.uuid4())
    
    def log_query(
        self,
        query: str,
        results: List[Dict],
        duration_ms: int = 0,
        user_id: str = "",
    ):
        """Log a search query."""
        event = UsageEvent(
            event_id=self._generate_event_id(),
            event_type="query",
            timestamp=datetime.now().isoformat(),
            user_id=user_id,
            session_id=self._current_session,
            metadata={"query": query},
            duration_ms=duration_ms,
        )
        self.db.log_event(event)
        
        # Log retrieval metrics
        if results:
            scores = [r.get("score", 0) for r in results]
            metric = RetrievalMetric(
                query=query,
                results_count=len(results),
                avg_score=sum(scores) / len(scores) if scores else 0,
                max_score=max(scores) if scores else 0,
                min_score=min(scores) if scores else 0,
            )
            self.db.log_retrieval(metric)
    
    def log_answer(
        self,
        query: str,
        has_sources: bool,
        duration_ms: int = 0,
        user_id: str = "",
    ):
        """Log an answer generation."""
        event = UsageEvent(
            event_id=self._generate_event_id(),
            event_type="answer",
            timestamp=datetime.now().isoformat(),
            user_id=user_id,
            session_id=self._current_session,
            metadata={"query": query, "has_sources": has_sources},
            duration_ms=duration_ms,
            success=has_sources,
        )
        self.db.log_event(event)
    
    def log_document_ingest(
        self,
        document_name: str,
        chunks_created: int,
        duration_ms: int,
        user_id: str = "",
    ):
        """Log document ingestion."""
        event = UsageEvent(
            event_id=self._generate_event_id(),
            event_type="document_ingest",
            timestamp=datetime.now().isoformat(),
            user_id=user_id,
            session_id=self._current_session,
            metadata={"document_name": document_name, "chunks_created": chunks_created},
            duration_ms=duration_ms,
        )
        self.db.log_event(event)
    
    def log_report_generate(
        self,
        project_name: str,
        sections: int,
        duration_ms: int,
        user_id: str = "",
    ):
        """Log report generation."""
        event = UsageEvent(
            event_id=self._generate_event_id(),
            event_type="report_generate",
            timestamp=datetime.now().isoformat(),
            user_id=user_id,
            session_id=self._current_session,
            metadata={"project_name": project_name, "sections": sections},
            duration_ms=duration_ms,
        )
        self.db.log_event(event)
    
    def log_validation(
        self,
        document_name: str,
        pass_count: int,
        fail_count: int,
        duration_ms: int,
        user_id: str = "",
    ):
        """Log validation check."""
        event = UsageEvent(
            event_id=self._generate_event_id(),
            event_type="validation",
            timestamp=datetime.now().isoformat(),
            user_id=user_id,
            session_id=self._current_session,
            metadata={
                "document_name": document_name,
                "pass_count": pass_count,
                "fail_count": fail_count,
            },
            duration_ms=duration_ms,
        )
        self.db.log_event(event)
    
    def submit_feedback(
        self,
        query: str,
        rating: int,
        feedback_text: str = "",
        event_id: str = "",
    ):
        """Submit user feedback."""
        self.db.log_feedback(event_id, query, rating, feedback_text)
    
    def get_dashboard_data(self, days: int = 7) -> Dict:
        """Get all data for dashboard."""
        return {
            "usage_stats": self.db.get_usage_stats(days),
            "retrieval_stats": self.db.get_retrieval_stats(days),
            "popular_queries": self.db.get_popular_queries(days),
            "recent_feedback": self.db.get_recent_feedback(),
        }


# Global analytics instance
_analytics: Optional[GeoBotAnalytics] = None


def get_analytics() -> GeoBotAnalytics:
    """Get or create global analytics instance."""
    global _analytics
    if _analytics is None:
        _analytics = GeoBotAnalytics()
    return _analytics
