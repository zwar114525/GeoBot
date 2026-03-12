"""
Report versioning system for tracking revisions.
"""
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field, asdict
from loguru import logger


@dataclass
class ReportVersion:
    """Represents a version of a report."""
    version_id: str
    report_id: str
    version_number: int
    content_hash: str
    content: str
    created_at: str
    created_by: str = ""
    changes: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReportDiff:
    """Difference between two report versions."""
    old_version: int
    new_version: int
    sections_added: List[str] = field(default_factory=list)
    sections_removed: List[str] = field(default_factory=list)
    sections_modified: List[str] = field(default_factory=list)
    parameter_changes: Dict[str, Any] = field(default_factory=dict)


class ReportVersioning:
    """Manage report versions and track changes."""
    
    def __init__(self, storage_dir: str = "./outputs/versions"):
        """
        Initialize versioning system.
        
        Args:
            storage_dir: Directory for version storage
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.storage_dir / "version_index.json"
        self._load_index()
    
    def _load_index(self):
        """Load version index from disk."""
        if self.index_file.exists():
            with open(self.index_file, 'r') as f:
                self.index = json.load(f)
        else:
            self.index = {"reports": {}}
            self._save_index()
    
    def _save_index(self):
        """Save version index to disk."""
        with open(self.index_file, 'w') as f:
            json.dump(self.index, f, indent=2)
    
    def _compute_hash(self, content: str) -> str:
        """Compute content hash."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _generate_version_id(self, report_id: str, version_number: int) -> str:
        """Generate unique version ID."""
        return f"{report_id}_v{version_number}"
    
    def create_version(
        self,
        report_id: str,
        content: str,
        created_by: str = "",
        changes: str = "",
        metadata: Dict[str, Any] = None,
    ) -> ReportVersion:
        """
        Create a new version of a report.
        
        Args:
            report_id: Unique report identifier
            content: Report content (markdown)
            created_by: User who created the version
            changes: Description of changes from previous version
            metadata: Additional metadata
            
        Returns:
            Created ReportVersion
        """
        # Get next version number
        if report_id not in self.index["reports"]:
            version_number = 1
        else:
            version_number = self.index["reports"][report_id]["latest_version"] + 1
        
        version = ReportVersion(
            version_id=self._generate_version_id(report_id, version_number),
            report_id=report_id,
            version_number=version_number,
            content_hash=self._compute_hash(content),
            content=content,
            created_at=datetime.now().isoformat(),
            created_by=created_by,
            changes=changes,
            metadata=metadata or {},
        )
        
        # Save version file
        version_file = self.storage_dir / f"{version.version_id}.json"
        with open(version_file, 'w') as f:
            json.dump(asdict(version), f, indent=2)
        
        # Update index
        if report_id not in self.index["reports"]:
            self.index["reports"][report_id] = {
                "versions": [],
                "latest_version": 0,
                "created_at": version.created_at,
            }
        
        self.index["reports"][report_id]["versions"].append(version.version_id)
        self.index["reports"][report_id]["latest_version"] = version_number
        self.index["reports"][report_id]["updated_at"] = version.created_at
        
        self._save_index()
        logger.info(f"Created version {version_number} for report {report_id}")
        
        return version
    
    def get_version(self, report_id: str, version_number: int = None) -> Optional[ReportVersion]:
        """
        Get a specific version of a report.
        
        Args:
            report_id: Report identifier
            version_number: Version number (latest if None)
            
        Returns:
            ReportVersion or None
        """
        if report_id not in self.index["reports"]:
            return None
        
        if version_number is None:
            version_number = self.index["reports"][report_id]["latest_version"]
        
        version_id = self._generate_version_id(report_id, version_number)
        version_file = self.storage_dir / f"{version_id}.json"
        
        if not version_file.exists():
            return None
        
        with open(version_file, 'r') as f:
            data = json.load(f)
        
        return ReportVersion(**data)
    
    def get_all_versions(self, report_id: str) -> List[ReportVersion]:
        """Get all versions of a report."""
        if report_id not in self.index["reports"]:
            return []
        
        versions = []
        for version_id in self.index["reports"][report_id]["versions"]:
            version_file = self.storage_dir / f"{version_id}.json"
            if version_file.exists():
                with open(version_file, 'r') as f:
                    data = json.load(f)
                versions.append(ReportVersion(**data))
        
        return sorted(versions, key=lambda v: v.version_number)
    
    def compare_versions(
        self,
        report_id: str,
        old_version: int,
        new_version: int = None,
    ) -> ReportDiff:
        """
        Compare two versions of a report.
        
        Args:
            report_id: Report identifier
            old_version: Old version number
            new_version: New version number (latest if None)
            
        Returns:
            ReportDiff object
        """
        if new_version is None:
            new_version = self.index["reports"][report_id]["latest_version"]
        
        old_v = self.get_version(report_id, old_version)
        new_v = self.get_version(report_id, new_version)
        
        if not old_v or not new_v:
            return ReportDiff(old_version=old_version, new_version=new_version or 0)
        
        # Simple diff based on sections
        old_sections = self._extract_sections(old_v.content)
        new_sections = self._extract_sections(new_v.content)
        
        diff = ReportDiff(
            old_version=old_version,
            new_version=new_version,
            sections_added=[s for s in new_sections if s not in old_sections],
            sections_removed=[s for s in old_sections if s not in new_sections],
            sections_modified=[s for s in old_sections if s in new_sections 
                             and self._section_content_changed(old_v.content, new_v.content, s)],
        )
        
        return diff
    
    def _extract_sections(self, content: str) -> List[str]:
        """Extract section headings from markdown content."""
        import re
        sections = re.findall(r'^##\s+(.+)$', content, re.MULTILINE)
        return [s.strip() for s in sections]
    
    def _section_content_changed(self, old_content: str, new_content: str, section: str) -> bool:
        """Check if content of a specific section changed."""
        import re
        pattern = rf'##\s+{re.escape(section)}\n(.*?)(?=##\s|$)'
        
        old_match = re.search(pattern, old_content, re.DOTALL)
        new_match = re.search(pattern, new_content, re.DOTALL)
        
        if not old_match or not new_match:
            return True
        
        return old_match.group(1).strip() != new_match.group(1).strip()
    
    def get_version_history(self, report_id: str) -> List[Dict]:
        """Get version history summary."""
        if report_id not in self.index["reports"]:
            return []
        
        history = []
        for version_id in self.index["reports"][report_id]["versions"]:
            version = self.get_version(report_id, int(version_id.split('_v')[-1]))
            if version:
                history.append({
                    "version_number": version.version_number,
                    "created_at": version.created_at,
                    "created_by": version.created_by,
                    "changes": version.changes,
                    "content_hash": version.content_hash,
                })
        
        return history


def create_versioning(storage_dir: str = "./outputs/versions") -> ReportVersioning:
    """Factory function to create versioning system."""
    return ReportVersioning(storage_dir)
