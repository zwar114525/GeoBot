"""
Graph Store for persistent cross-reference edges.

This module provides persistent storage for the cross-reference graph
used by the Hybrid Hierarchical Graph Retrieval system.
"""
import json
import os
from typing import Dict, List, Optional
from pathlib import Path
from loguru import logger


class GraphStore:
    """
    Graph edge storage with persistence.

    Stores relationships between chunks as edges, enabling
    graph expansion during retrieval.
    """

    def __init__(self, storage_path: str = "data/graph_edges.json"):
        self.storage_path = storage_path
        # Ensure directory exists
        Path(storage_path).parent.mkdir(parents=True, exist_ok=True)
        self.edges: Dict[str, List[str]] = self._load_edges()

    def _load_edges(self) -> Dict[str, List[str]]:
        """Load graph edges from disk."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Failed to load graph edges, starting fresh: {self.storage_path}")
                return {}
        return {}

    def _save_edges(self):
        """Save graph edges to disk."""
        with open(self.storage_path, 'w', encoding='utf-8') as f:
            json.dump(self.edges, f, indent=2, ensure_ascii=False)

    def add_edge(self, source_id: str, target_id: str):
        """Add a directed edge (source -> target)."""
        if source_id not in self.edges:
            self.edges[source_id] = []
        if target_id not in self.edges[source_id]:
            self.edges[source_id].append(target_id)
            self._save_edges()

    def add_edges_from_chunk(self, chunk_id: str, references: List[str]):
        """Batch add edges from a chunk's references field."""
        for ref_id in references:
            self.add_edge(chunk_id, ref_id)

    def get_references(self, chunk_id: str) -> List[str]:
        """Get all chunks referenced by this chunk (outgoing edges)."""
        return self.edges.get(chunk_id, [])

    def get_referenced_by(self, chunk_id: str) -> List[str]:
        """Get all chunks that reference this chunk (incoming edges)."""
        return [
            source_id
            for source_id, targets in self.edges.items()
            if chunk_id in targets
        ]

    def has_references(self, chunk_id: str) -> bool:
        """Check if chunk has any outgoing references."""
        return len(self.get_references(chunk_id)) > 0

    def clear(self):
        """Clear all edges."""
        self.edges = {}
        self._save_edges()

    def get_stats(self) -> Dict:
        """Get graph statistics."""
        return {
            "total_nodes": len(self.edges),
            "total_edges": sum(len(v) for v in self.edges.values())
        }

    def rebuild_from_chunks(self, chunks: List[Dict]):
        """
        Rebuild graph edges from a list of chunk dictionaries.

        Args:
            chunks: List of chunk dicts with 'chunk_id' and 'references' fields
        """
        self.clear()
        for chunk in chunks:
            chunk_id = chunk.get("chunk_id") or chunk.get("id")
            references = chunk.get("references", [])
            if chunk_id and references:
                self.add_edges_from_chunk(chunk_id, references)
        logger.info(f"Rebuilt graph with {self.get_stats()}")
